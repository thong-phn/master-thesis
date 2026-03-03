# pytorch --> mct
import torch
import model_compression_toolkit as mct
from torch.utils.data import DataLoader

from train import MyDataset
from model import SeparableConvCNN

num_channels = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_path = './models/best_model_subject1_val.pth'

# Load the best model
model = SeparableConvCNN(num_channels=num_channels).to(device)
state = torch.load(pretrain_path, map_location=device)
model.load_state_dict(state, strict=False)
print(f"Loaded pretrain weights from {[pretrain_path]}")
model.eval()

# Prepare representative dataset from training set: 100 * mini-batch = 100*64
rep_dataset = MyDataset(
    root_path='./uci-har',
    split='train',
    use_gyro=True,
)

rep_batch_size = 64
num_rep_batches = 100

rep_loader = DataLoader(
    rep_dataset,
    batch_size=rep_batch_size,
    shuffle=True,
    drop_last=True,
)


def representative_data_gen():
    produced_batches = 0

    while produced_batches < num_rep_batches:
        for fft_mag, _ in rep_loader:
            yield [fft_mag.to(device)]
            produced_batches += 1
            if produced_batches >= num_rep_batches:
                break

quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_data_gen
)
quantized_model.eval()

# Evaluate model
test_dataset = MyDataset(
    root_path='./uci-har',
    split='test',
    use_gyro=True,
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

q_device = next(quantized_model.parameters()).device
test_correct = 0
test_total = 0

with torch.no_grad():
    for fft_mag, labels in test_loader:
        fft_mag = fft_mag.to(q_device)
        labels = labels.to(q_device)

        outputs = quantized_model(fft_mag)
        preds = torch.argmax(outputs, dim=1)

        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()

test_acc = 100.0 * test_correct / max(test_total, 1)
print(f"Quantized model test accuracy: {test_acc:.2f}%")
