import torch
import torch.nn as nn
import torch.ao.quantization as ao_quantization
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
import numpy as np
import copy
import sys
import argparse
import re
import wandb
import os
import random

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from lib.wear_train import WEAR_Dataset as WearDataset
from lib.model import SeparableConvCNN, PrunedSeparableConvCNN

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class QuantizableSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def make_conv1d_from_bn(bn: nn.BatchNorm1d) -> nn.Conv1d:
    channels = bn.num_features
    conv = nn.Conv1d(channels, channels, kernel_size=1, groups=channels, bias=True)
    eps = bn.eps
    mu = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    scale = gamma / torch.sqrt(var + eps)
    shift = beta - mu * scale
    conv.weight.data = scale.view(channels, 1, 1).clone()
    conv.bias.data = shift.clone()
    return conv

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ExportableSeparableConvCNN(nn.Module):
    def __init__(self, num_classes=8, num_channels=3, 
                 b2=64, b3=128, b4=128, ptq_dtype=torch.qint8):
        super().__init__()
        
        self.quant_stub = ao_quantization.QuantStub()
        self.dequant_stub = ao_quantization.DeQuantStub()
        
        self.sep_conv1 = QuantizableSeparableConv1d(num_channels, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.bn1_conv = nn.Conv1d(32, 32, kernel_size=1, groups=32, bias=True) 
        self.pool1 = nn.MaxPool1d(2)
        
        self.sep_conv2 = QuantizableSeparableConv1d(32, b2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2_conv = nn.Conv1d(b2, b2, kernel_size=1, groups=b2, bias=True)
        self.pool2 = nn.MaxPool1d(2)
        
        self.sep_conv3 = QuantizableSeparableConv1d(b2, b3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3_conv = nn.Conv1d(b3, b3, kernel_size=1, groups=b3, bias=True)
        self.pool3 = nn.MaxPool1d(2)
        
        self.sep_conv4 = QuantizableSeparableConv1d(b3, b4, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4_conv = nn.Conv1d(b4, b4, kernel_size=1, groups=b4, bias=True)
        self.pool4 = nn.MaxPool1d(2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = Flatten()
        self.dropout = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(b4, 64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.quant_stub(x)
        
        x = self.sep_conv1(x)
        x = self.relu1(x)
        x = self.bn1_conv(x)
        x = self.pool1(x)
        
        x = self.sep_conv2(x)
        x = self.relu2(x)
        x = self.bn2_conv(x)
        x = self.pool2(x)
        
        x = self.sep_conv3(x)
        x = self.relu3(x)
        x = self.bn3_conv(x)
        x = self.pool3(x)
        
        x = self.sep_conv4(x)
        x = self.relu4(x)
        x = self.bn4_conv(x)
        x = self.pool4(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        
        x = self.dequant_stub(x)
        return x

def evaluate_quantized(model, data_loader, device, ptq_mode, orig_quant_stub=None, bin_mask=None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            if bin_mask is not None:
                inputs = inputs * bin_mask.unsqueeze(0).to(inputs.device)
            inputs, labels = inputs.to(device), labels.to(device)
            
            if ptq_mode == 3 and orig_quant_stub is not None:
                inputs = torch.quantize_per_tensor(inputs, orig_quant_stub.scale, orig_quant_stub.zero_point, orig_quant_stub.dtype)
                
            outputs = model(inputs)
            
            if ptq_mode == 3:
                outputs = outputs.dequantize()
                
            _, predicted = outputs.max(1)
            
            bs = labels.size(0)
            total += bs
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100. * correct / max(total, 1)
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100.0
    return {"accuracy": accuracy, "f1_macro": f1_macro}

def extract_hard_bin_mask(val_subject):
    log_path = project_root / "log" / "wear_loso_five_stage_results_fft.txt"
    if not log_path.exists():
        return None
    with open(log_path, "r") as f:
        content = f.read()
        
    blocks = content.split("==================================================")
    for block in blocks:
        if f"Fold Val Subject {val_subject}:" in block:
            m = re.search(r"Hard Bin Mask:\s*\[([\d\.\,\s]+)\]", block)
            if m:
                mask_str = m.group(1)
                mask = [float(x.strip()) for x in mask_str.split(",")]
                return torch.tensor(mask, dtype=torch.float32)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--ptq_mode", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()

    set_seed(42)

    root_path = project_root / "wear"
    all_train_subjects = list(range(18))
    test_eval_subjects = list(range(18, 24))

    if args.subjects is not None:
        val_subjects = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]
    else:
        val_subjects = all_train_subjects

    device = torch.device('cpu') 
    num_classes = 8
    num_channels = 6 # Assuming 6 channels for wear
    freq_bins = 51

    fold_results = []
    g = torch.Generator()
    g.manual_seed(42)

    test_dataset = WearDataset(root_path, split='test', subject_ids=test_eval_subjects)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for val_subject in val_subjects:
        train_subjects = [s for s in all_train_subjects if s != val_subject]
        
        if args.stage == 1:
            model_path = project_root / "models" / f"wear_best_model_five_stage_subject{val_subject}_val_stage1.pth"
        elif args.stage == 3:
            model_path = project_root / "models" / f"wear_best_model_five_stage_subject{val_subject}_val_stage3_pruned_input.pth"
        elif args.stage == 5:
            model_path = project_root / "models" / f"wear_best_model_five_stage_subject{val_subject}_val_stage5_compact.pth"

        print(f"\n{'='*50}\nPTQ Fold | Model: Val Subject {val_subject} | Stage {args.stage} | Mode {args.ptq_mode}\n{'='*50}")

        if not model_path.exists():
            print(f"ERROR: Could not find trained model at {model_path}. Skipping.")
            continue

        state_dict = torch.load(model_path, map_location="cpu")
        
        # Infer dimensions safely
        b2 = state_dict['sep_conv2.pointwise.weight'].shape[0] if 'sep_conv2.pointwise.weight' in state_dict else 64
        b3 = state_dict['sep_conv3.pointwise.weight'].shape[0] if 'sep_conv3.pointwise.weight' in state_dict else 128
        b4 = state_dict['sep_conv4.pointwise.weight'].shape[0] if 'sep_conv4.pointwise.weight' in state_dict else 128

        if args.stage in [1, 3]:
            original_model = SeparableConvCNN(num_classes=num_classes, num_channels=num_channels)
        else:
            original_model = PrunedSeparableConvCNN(num_classes=num_classes, num_channels=num_channels, 
                                                    block2_channels=b2, block3_channels=b3, block4_channels=b4)
            
        original_model.load_state_dict(state_dict)
        original_model.eval()

        bin_mask = None
        if args.stage == 3:
            bin_mask = extract_hard_bin_mask(val_subject)

        exportable_model = ExportableSeparableConvCNN(num_classes=num_classes, num_channels=num_channels, 
                                                      b2=b2, b3=b3, b4=b4)
        model_state = exportable_model.state_dict()
        for key in state_dict.keys():
            if key in model_state:
                model_state[key].copy_(state_dict[key])
        exportable_model.load_state_dict(model_state)

        exportable_model.bn1_conv.load_state_dict(make_conv1d_from_bn(original_model.bn1).state_dict())
        exportable_model.bn2_conv.load_state_dict(make_conv1d_from_bn(original_model.bn2).state_dict())
        exportable_model.bn3_conv.load_state_dict(make_conv1d_from_bn(original_model.bn3).state_dict())
        exportable_model.bn4_conv.load_state_dict(make_conv1d_from_bn(original_model.bn4).state_dict())
        exportable_model.eval()

        calib_subjects = train_subjects[:3] 
        calib_dataset = WearDataset(root_path, split='train', subject_ids=calib_subjects)
        # Using a subset of 50 samples per class for stratified calibration to ensure min/max cover range
        targets = np.array([calib_dataset.labels[i] for i in range(len(calib_dataset))])
        calib_indices = []
        for c in range(num_classes):
            idx_c = np.where(targets == c)[0]
            if len(idx_c) > 0:
                np.random.shuffle(idx_c)
                calib_indices.extend(idx_c[:50])
        
        calib_subset = torch.utils.data.Subset(calib_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=True, generator=g)

        torch.backends.quantized.engine = 'qnnpack'
        
        if args.ptq_mode in [1, 2]:
            exportable_model.qconfig = ao_quantization.QConfig(
                activation=ao_quantization.FakeQuantize.with_args(observer=ao_quantization.MinMaxObserver, quant_min=-32768, quant_max=32767, dtype=torch.qint32),
                weight=ao_quantization.FakeQuantize.with_args(observer=ao_quantization.MinMaxObserver, quant_min=-128, quant_max=127, dtype=torch.qint8)
            )
        else:
            exportable_model.qconfig = ao_quantization.get_default_qconfig('qnnpack')

        torch.ao.quantization.fuse_modules(exportable_model, [
            ['sep_conv1.pointwise', 'relu1'], 
            ['sep_conv2.pointwise', 'relu2'],
            ['sep_conv3.pointwise', 'relu3'],
            ['sep_conv4.pointwise', 'relu4'],
            ['fc1', 'relu5'],
        ], inplace=True) 

        ao_quantization.prepare(exportable_model, inplace=True)

        with torch.no_grad():
            for inputs, _ in calib_loader:
                if bin_mask is not None:
                    inputs = inputs * bin_mask.unsqueeze(0).to(inputs.device)
                exportable_model(inputs) 

        quantized_model = exportable_model
        if args.ptq_mode == 3:
            quantized_model = ao_quantization.convert(exportable_model, inplace=True)
        
        orig_quant_stub = None
        if args.ptq_mode == 3:
            orig_quant_stub = quantized_model.quant_stub
            quantized_model.quant_stub = nn.Identity()
            quantized_model.dequant_stub = nn.Identity()

        q_metrics = evaluate_quantized(quantized_model, test_loader, device, args.ptq_mode, orig_quant_stub, bin_mask)
        q_acc = q_metrics['accuracy']
        q_f1 = q_metrics['f1_macro']

        print(f" -> Quantized Test Acc : {q_acc:.2f}%")
        print(f" -> Quantized Test F1  : {q_f1:.2f}%")

        fold_results.append({
            'val_subject': val_subject,
            'acc_int': q_acc,
            'f1_int': q_f1
        })
        
        out_prefix = f"best_model_ptq_stage{args.stage}_mode{args.ptq_mode}"
        torch.save(quantized_model.state_dict(), project_root / "models" / f"{out_prefix}_val_{val_subject}.pth")

    if len(fold_results) == 0:
        return

    acc_int_list = [res['acc_int'] for res in fold_results]
    f1_int_list = [res['f1_int'] for res in fold_results]

    mean_int_acc, std_int_acc = np.mean(acc_int_list), np.std(acc_int_list)
    mean_int_f1, std_int_f1 = np.mean(f1_int_list), np.std(f1_int_list)

    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"wear_ptq_results_stage{args.stage}_mode{args.ptq_mode}.txt"
    log_path = log_dir / log_name

    with open(log_path, "w") as f:
        f.write(f"Quantized Test Accuracy: {mean_int_acc:.2f}% +- {std_int_acc:.2f}%\n")
        f.write(f"Quantized Test F1-Macro: {mean_int_f1:.2f}% +- {std_int_f1:.2f}%\n\n")
        f.write("Detailed Fold Results:\n")
        for res in fold_results:
            f.write(f"Fold Val {res['val_subject']}: Acc = {res['acc_int']:.2f}%, F1 = {res['f1_int']:.2f}%\n")

    try:
        wandb.login()
        run = wandb.init(
            project="thesis",
            name=f"wear-ptq-log-stage{args.stage}-mode{args.ptq_mode}",
            job_type="ptq_results_upload",
            reinit=True,
            config={
                "training_type": "ptq",
                "stage": args.stage,
                "ptq_mode": args.ptq_mode
            }
        )
        artifact = wandb.Artifact(
            name=f"wear-ptq-results-stage{args.stage}-mode{args.ptq_mode}",
            type="results-log",
        )
        artifact.add_file(str(log_path))
        run.log_artifact(artifact)
        run.finish()
    except Exception as e:
        print(f"WandB Logging failed: {e}")

if __name__ == "__main__":
    main()
