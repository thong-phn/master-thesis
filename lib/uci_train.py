import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.fftpack import dct
from lib.ml_lib import _val_one_epoch, stage1_pipeline, stage2_channel_gumbel_pruning_pipeline, _load_weights_to_gumbel_model


# Dataset 
class UCIHAR_Dataset(Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        subject_ids=None,
        preprocessing='fft',
    ):
        """
        Load UCI-HAR data, compute FFT
        Args:
            root_path: Path to dataset
            split: 'train' or 'test'
            subject_ids: List of subject IDs to filter (optional)
        """
        self.root_path = Path(root_path)
        self.split_path = self.root_path / split
        self.inertial_path = self.split_path/"Inertial Signals"
        self.preprocessing = preprocessing

        # Load Y (label) and subjects
        path_to_y_file = self.split_path/f"y_{split}.txt"
        path_to_subject_file = self.split_path/f"subject_{split}.txt"

        all_labels = np.loadtxt(path_to_y_file, dtype=int) - 1 # 0-indexed [0, 1, 2, 3, 4, 5]
        all_subjects = np.loadtxt(path_to_subject_file, dtype=int)
        
        if self.preprocessing == 'no': # time domain
            # Load raw accelerometer data (with gravity)
            accel_files = {       
                "X": f"total_acc_x_{split}.txt",
                "Y": f"total_acc_y_{split}.txt",
                "Z": f"total_acc_z_{split}.txt",
            }
        else: # frequency domain
            # Load processed accelerometer data (low pass filter, remove gravity)
            accel_files = {       
                "X": f"body_acc_x_{split}.txt",
                "Y": f"body_acc_y_{split}.txt",
                "Z": f"body_acc_z_{split}.txt",
            }
        signals = []
        for axis in ["X", "Y", "Z"]:
            data = np.loadtxt(self.inertial_path/accel_files[axis])
            signals.append(data)

        # Load gyro data
        gyro_files = {
            "X": f"body_gyro_x_{split}.txt",
            "Y": f"body_gyro_y_{split}.txt",
            "Z": f"body_gyro_z_{split}.txt",
        }
        for axis in ["X", "Y", "Z"]:
            data = np.loadtxt(self.inertial_path/gyro_files[axis])
            signals.append(data)

        all_signals = np.stack(signals, axis=1) # Stack to shape (samples, num_channels, 128) 

        self.preprocessing = preprocessing

        # Return
        if subject_ids is None:
            self.labels = all_labels
            self.signals = all_signals
            self.subjects = all_subjects
        else:
            mask = np.isin(all_subjects, subject_ids)
            self.labels = all_labels[mask]
            self.signals = all_signals[mask]
            self.subjects = all_subjects[mask]


    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _compute_fft_magnitude(signal):
        fft_vals = np.fft.rfft(signal, axis=-1)
        mag = np.abs(fft_vals) / signal.shape[-1]

        # One-sided amplitude scaling
        if signal.shape[-1] % 2 == 0:
            mag[..., 1:-1] *= 2
        else:
            mag[..., 1:] *= 2
        
        return mag

    @staticmethod
    def _compute_ihw(signal, fixed_point_scale=1024):
        # Keep fixed-point precision, then emulate signed int16 arithmetic like C.
        int_signal = np.rint(signal * fixed_point_scale).astype(np.int64)

        def to_int16_np(val):
            val = np.bitwise_and(val, 0xFFFF)
            return np.where(val > 32767, val - 65536, val).astype(np.int64)

        length = int_signal.shape[-1]
        half_len = length // 2

        even = to_int16_np(int_signal[..., 0:2 * half_len:2])
        odd = to_int16_np(int_signal[..., 1:2 * half_len:2])

        # Detail: d = to_int16(odd - even)
        detail = to_int16_np(odd - even)
        # Approximation: a = to_int16(even + (d >> 1))
        approx = to_int16_np(even + np.right_shift(detail, 1))

        ihw = np.zeros_like(int_signal, dtype=np.int64)
        ihw[..., :half_len] = approx
        ihw[..., half_len:half_len + half_len] = detail

        # Preserve odd tail sample if present.
        if length % 2 == 1:
            ihw[..., -1] = to_int16_np(int_signal[..., -1])

        # Return raw integer coefficients for MCU-style integer pipeline simulation.
        return ihw.astype(np.int16)

    @staticmethod
    def _compute_dct(signal):
        dct_vals = dct(signal, type=2, axis=-1, norm='ortho')
        return np.abs(dct_vals)

    def __getitem__(self, idx):
        signal = self.signals[idx]  # (6, window_size)

        if self.preprocessing == 'dct':
            mag = self._compute_dct(signal)
        elif self.preprocessing == 'ihw':
            mag = self._compute_ihw(signal)
        elif self.preprocessing == 'no':
            mag = signal
        elif self.preprocessing == 'fft':
            mag = self._compute_fft_magnitude(signal)
        else:
            raise ValueError(f"Unsupported preprocessing: {self.preprocessing}")

        return torch.FloatTensor(mag), torch.LongTensor([self.labels[idx]])[0]
        
# Training function
def train_loso(root_path, model_class, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    Args:
        root_path: path to UCI-HAR
        model_class
        train_subjects
        val_subjects
        wandb_run:
        **train_kwargs
    """
    # Hyperparameters 
    epochs = train_kwargs.get('epochs', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    sparsity_weight = train_kwargs.get('sparsity_weight', 0.01)
    model_path = Path(train_kwargs.get('model_path', './models/best_model.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessing = train_kwargs.get('preprocessing', 'fft')

    # Create dataset and dataloader
    # Exp-5: log1p only, no z-score normalization
    train_dataset = UCIHAR_Dataset(
        root_path,
        split='train',
        subject_ids=train_subjects,
        preprocessing=preprocessing,
    )

    val_dataset = UCIHAR_Dataset(
        root_path,
        split='train',
        subject_ids=val_subjects,
        preprocessing=preprocessing,
    )
    test_dataset = UCIHAR_Dataset(
        root_path,
        split='test',
        subject_ids=None,
        preprocessing=preprocessing,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")



    # Training loop configuration
    freq_bins = train_dataset[0][0].shape[-1]
    model = model_class(freq_bins=freq_bins).to(device)
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_epoch = 0 # 
    epochs_no_improve = 0 # early stopping

    print("-"*50)
    # Training loop
    for epoch in range(epochs):
        # Exp-G4: Enable tau annealing
        if hasattr(model, 'set_tau'):
            model.set_tau(epoch, epochs)
        
        # Train one epoch
        train_loss_sum = 0.0 # sum of training loss  
        train_correct = 0 # no. of training samples predicted correctly
        train_total = 0 # no. of training samples used
        
        model.train()

        for fft_mag, labels in train_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)

            outputs = model(fft_mag) # 1. forward 
            loss = criterion(outputs, labels) # 2. loss
            
            # Exp-G3/Combined: Progressive up to Lower Weight (Exp 2 + Exp 3)
            # L1 penalty on fraction of bins kept
            if hasattr(model, 'mask_l1') and model.mask_l1 is not None:
                # sparsity_weight = min(0.005, epoch * 0.005 / 20)
                # sparsity_weight = 0.01
                loss = loss + sparsity_weight * model.mask_l1
            
            optimizer.zero_grad() # 3. backward: zero_grad
            loss.backward() # cal gradient
            optimizer.step() # update step

            train_loss_sum += loss.item() * labels.size(0) # loss.item() is the average loss of the batch -> recover loss of the batch
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss_sum/train_total
        train_acc = train_correct/train_total * 100.0

        # Val one epoch
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # no need to track grad in val
            for fft_mag, labels in val_dataloader:
                fft_mag, labels = fft_mag.to(device), labels.to(device)
                
                outputs = model(fft_mag)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = 100. * val_correct / max(val_total, 1)
        scheduler.step(val_loss) # Step LR scheduler on val loss
        
        # Save best model based on val_loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            best_epoch = epoch+1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Exp-G1: Print mask statistics if model has masking
        mask_info = ""
        if hasattr(model, 'mask_l1') and model.mask_l1 is not None:
            mask_fraction = model.mask_l1.item()
            tau_info = ""
            if hasattr(model, 'current_tau'):
                tau_info = f' (tau={model.current_tau:.2f})'
            mask_info = f'; Mask: {mask_fraction:.2%}' + tau_info
        
        print(f'Epoch [{epoch+1}/{epochs}]: '
              f'Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; '
              f'Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}' + mask_info)
        
        if wandb_run is not None: # tracking
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_val_loss,
                "lr": optimizer.param_groups[0]["lr"],  # actual LR
                # "epochs_no_improve": epochs_no_improve, # early stopping
            })

        if epochs_no_improve >= patience: # early stopping
            print(f"Early Stopping: Epoch [{epoch+1}/{epochs}] (patience={patience}, min_delta={min_delta}).")
            break


    # Test with best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loss_sum, test_correct, test_total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for fft_mag, labels in test_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)
            outputs = model(fft_mag)
            loss = criterion(outputs, labels)

            bs = labels.size(0)
            test_loss_sum += loss.item() * bs
            _, preds = outputs.max(1)
            test_total += bs
            test_correct += preds.eq(labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss_sum / max(test_total, 1)
    test_acc = 100.0 * test_correct / max(test_total, 1)
    test_f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    print("-"*50)
    print(f"Summary:")
    print(f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1 Macro: {test_f1_macro:.4f}")
    
    # Exp-G1: Print final learned mask if available
    if hasattr(model, 'last_mask') and model.last_mask is not None:
        final_mask = model.last_mask.cpu().numpy()
        bins_kept = (final_mask > 0.5).sum()
        total_bins = len(final_mask)
        print(f"\nFinal Mask Statistics:")
        print(f"  Bins kept: {bins_kept}/{total_bins} ({bins_kept/total_bins:.1%})")
        print(f"  All mask values: {final_mask}")
    else:
        final_mask = None

    if wandb_run is not None: # tracking
        wandb_run.log({
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1_macro": test_f1_macro,
        })

    return {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1_macro": test_f1_macro,
        "model_path": str(model_path),
        "final_mask": final_mask,
    }


def _resolve_device(device):
    if isinstance(device, str):
        return torch.device(device)
    return device


class SlicedUCIDataset(Dataset):
    """Wrap UCIHAR_Dataset and keep only selected frequency bins (hard slicing)."""

    def __init__(self, base_dataset, keep_indices):
        self.base_dataset = base_dataset
        self.keep_indices = torch.as_tensor(keep_indices, dtype=torch.long)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        x = x.index_select(-1, self.keep_indices)
        return x, y


def _count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def _copy_batchnorm_subset(src_bn, dst_bn, keep_indices=None):
    with torch.no_grad():
        if keep_indices is None:
            dst_bn.load_state_dict(src_bn.state_dict())
            return

        idx = keep_indices.to(dtype=torch.long, device=src_bn.weight.device)
        dst_bn.weight.copy_(src_bn.weight[idx])
        dst_bn.bias.copy_(src_bn.bias[idx])
        dst_bn.running_mean.copy_(src_bn.running_mean[idx])
        dst_bn.running_var.copy_(src_bn.running_var[idx])
        dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)


def _copy_separable_conv_subset(src_conv, dst_conv, keep_in_indices=None, keep_out_indices=None):
    with torch.no_grad():
        if keep_in_indices is None:
            keep_in_indices = torch.arange(src_conv.depthwise.weight.shape[0], device=src_conv.depthwise.weight.device)
        if keep_out_indices is None:
            keep_out_indices = torch.arange(src_conv.pointwise.weight.shape[0], device=src_conv.pointwise.weight.device)

        keep_in_indices = keep_in_indices.to(dtype=torch.long, device=src_conv.depthwise.weight.device)
        keep_out_indices = keep_out_indices.to(dtype=torch.long, device=src_conv.pointwise.weight.device)

        dst_conv.depthwise.weight.copy_(src_conv.depthwise.weight[keep_in_indices])
        dst_conv.pointwise.weight.copy_(src_conv.pointwise.weight[keep_out_indices][:, keep_in_indices, :])


def _copy_linear_input_subset(src_linear, dst_linear, keep_input_indices=None):
    with torch.no_grad():
        if keep_input_indices is None:
            dst_linear.load_state_dict(src_linear.state_dict())
            return

        keep_input_indices = keep_input_indices.to(dtype=torch.long, device=src_linear.weight.device)
        dst_linear.weight.copy_(src_linear.weight[:, keep_input_indices])
        dst_linear.bias.copy_(src_linear.bias)


def _build_pruned_channel_model_from_stage2(stage2_model, num_classes, dropout, device):
    hard_masks = stage2_model.get_hard_masks()

    keep_indices = {}
    for block_name in ("block2", "block3", "block4"):
        mask = hard_masks[block_name].detach().cpu()
        indices = torch.where(mask > 0.5)[0]
        if indices.numel() == 0:
            raise ValueError(f"Stage 4 mask for {block_name} pruned all channels; Stage 5 cannot proceed.")
        keep_indices[block_name] = indices

    from lib.model import PrunedSeparableConvCNN

    pruned_model = PrunedSeparableConvCNN(
        num_classes=num_classes,
        num_channels=6,
        block2_channels=int(keep_indices["block2"].numel()),
        block3_channels=int(keep_indices["block3"].numel()),
        block4_channels=int(keep_indices["block4"].numel()),
        dropout=dropout,
    ).to(device)

    with torch.no_grad():
        _copy_separable_conv_subset(stage2_model.sep_conv1, pruned_model.sep_conv1)
        pruned_model.bn1.load_state_dict(stage2_model.bn1.state_dict())

        _copy_separable_conv_subset(
            stage2_model.sep_conv2,
            pruned_model.sep_conv2,
            keep_in_indices=None,
            keep_out_indices=keep_indices["block2"],
        )
        _copy_batchnorm_subset(stage2_model.bn2, pruned_model.bn2, keep_indices["block2"])

        _copy_separable_conv_subset(
            stage2_model.sep_conv3,
            pruned_model.sep_conv3,
            keep_in_indices=keep_indices["block2"],
            keep_out_indices=keep_indices["block3"],
        )
        _copy_batchnorm_subset(stage2_model.bn3, pruned_model.bn3, keep_indices["block3"])

        _copy_separable_conv_subset(
            stage2_model.sep_conv4,
            pruned_model.sep_conv4,
            keep_in_indices=keep_indices["block3"],
            keep_out_indices=keep_indices["block4"],
        )
        _copy_batchnorm_subset(stage2_model.bn4, pruned_model.bn4, keep_indices["block4"])

        _copy_linear_input_subset(stage2_model.fc1, pruned_model.fc1, keep_indices["block4"])
        pruned_model.fc2.load_state_dict(stage2_model.fc2.state_dict())

    return pruned_model, keep_indices


def _load_matching_weights(target_model, source_state_dict):
    """Load only parameters whose names and shapes match between source and target models."""
    target_state = target_model.state_dict()
    loaded = 0
    skipped = 0

    for name, src_param in source_state_dict.items():
        tgt_param = target_state.get(name)
        if tgt_param is not None and tgt_param.shape == src_param.shape:
            target_state[name] = src_param
            loaded += 1
        else:
            skipped += 1

    target_model.load_state_dict(target_state)
    print(f"  Partially loaded weights: loaded={loaded}, skipped={skipped}")
    return target_model


def _evaluate_classifier(model, dataloader, criterion, device):
    avg_loss, acc, f1, _, _ = _val_one_epoch(model, dataloader, criterion, device)
    return avg_loss, acc, f1


def _save_confusion_matrix_artifact(
    y_true,
    y_pred,
    output_dir,
    stage_name,
    model_name,
    wandb_run=None,
    artifact_name=None,
    preprocessing=None,
    class_names=None,
):
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = [
            'walking',
            'walking_upstairs',
            'walking_downstairs',
            'sitting',
            'standing',
            'laying',
        ]
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if wandb_run is not None:
        import wandb

        safe_stage_name = stage_name.replace(' ', '_').lower()
        safe_model_name = model_name.replace(' ', '_').lower()
        log_key_prefix = f"{safe_stage_name}_{safe_model_name}"
        try:
            wandb_run.log({
                f"{log_key_prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.asarray(y_true).astype(int).tolist(),
                    preds=np.asarray(y_pred).astype(int).tolist(),
                    class_names=class_names,
                )
            })
        except Exception as e:
            print(f"Warning: failed to log W&B confusion matrix chart: {e}")

    return {
        'matrix': cm,
    }


def _get_hard_bin_mask_from_model(model):
    with torch.no_grad():
        hard = torch.softmax(model.bin_logits, dim=-1).argmax(dim=-1).float()
    return hard


def train_loso_uci_multi_stage(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    Five-stage LOSO training for UCI-HAR dataset:
    Stage 1: Train SeparableConvCNN on full input.
    Stage 2: Load stage-1 weights into GumbelMaskSeparableConvCNN and learn input-bin pruning.
    Stage 3: Apply hard bin mask to data and retrain SeparableConvCNN.
    Stage 4: Load stage-3 weights into GumbelChannelPruningCNN and train channel pruning on pruned input.
    Stage 5: Physically prune channels and fine-tune compact model.
    """
    from lib.model import SeparableConvCNN, GumbelMaskSeparableConvCNN, GumbelChannelPruningCNN

    epochs_stage1 = train_kwargs.get('epochs_stage1', 60)
    epochs_stage2 = train_kwargs.get('epochs_stage2', 60)
    epochs_stage3 = train_kwargs.get('epochs_stage3', 60)
    epochs_stage4 = train_kwargs.get('epochs_stage4', 60)
    epochs_stage5 = train_kwargs.get('epochs_stage5', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = _resolve_device(train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    preprocessing = train_kwargs.get('preprocessing', 'fft')
    dropout = train_kwargs.get('dropout', 0.4)
    tau_start = train_kwargs.get('tau_start', 10.0)
    tau_end = train_kwargs.get('tau_end', 1.0)
    performance_mode = bool(train_kwargs.get('performance', False))
    log_every_n_epochs = max(1, int(train_kwargs.get('log_every_n_epochs', 5)))

    sparsity_weight_bin = train_kwargs.get('sparsity_weight_bin', train_kwargs.get('sparsity_weight', 0.01))
    sparsity_weight_channel = train_kwargs.get('sparsity_weight_channel', train_kwargs.get('sparsity_weight', 0.01))

    stage2_backbone_lr_factor = train_kwargs.get('stage2_backbone_lr_factor', 0.1)
    stage4_backbone_lr_factor = train_kwargs.get('stage4_backbone_lr_factor', 0.1)
    stage5_loaded_lr_factor = train_kwargs.get('stage5_loaded_lr_factor', 0.1)

    model_path = Path(train_kwargs.get('model_path', './models/best_uci_model_multi_stage.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_model_path_arg = train_kwargs.get('stage1_model_path')
    if stage1_model_path_arg is not None:
        stage1_model_path = Path(stage1_model_path_arg).expanduser()
    else:
        stage1_model_path = model_path.parent / f"{model_path.stem}_stage1.pth"
    stage1_model_path.parent.mkdir(parents=True, exist_ok=True)

    stage2_model_path_arg = train_kwargs.get('stage2_model_path')
    if stage2_model_path_arg is not None:
        stage2_model_path = Path(stage2_model_path_arg).expanduser()
    else:
        stage2_model_path = model_path.parent / f"{model_path.stem}_stage2_bin.pth"
    stage2_model_path.parent.mkdir(parents=True, exist_ok=True)

    stage3_model_path_arg = train_kwargs.get('stage3_model_path')
    if stage3_model_path_arg is not None:
        stage3_model_path = Path(stage3_model_path_arg).expanduser()
    else:
        stage3_model_path = model_path.parent / f"{model_path.stem}_stage3_pruned_input.pth"
    stage3_model_path.parent.mkdir(parents=True, exist_ok=True)

    stage4_model_path_arg = train_kwargs.get('stage4_model_path')
    if stage4_model_path_arg is not None:
        stage4_model_path = Path(stage4_model_path_arg).expanduser()
    else:
        stage4_model_path = model_path.parent / f"{model_path.stem}_stage4_channel.pth"
    stage4_model_path.parent.mkdir(parents=True, exist_ok=True)

    stage5_model_path_arg = train_kwargs.get('stage5_model_path')
    if stage5_model_path_arg is not None:
        stage5_model_path = Path(stage5_model_path_arg).expanduser()
    else:
        stage5_model_path = model_path.parent / f"{model_path.stem}_stage5_compact.pth"
    stage5_model_path.parent.mkdir(parents=True, exist_ok=True)

    use_pretrained_stage1 = stage1_model_path_arg is not None
    use_pretrained_stage2 = stage2_model_path_arg is not None
    use_pretrained_stage3 = stage3_model_path_arg is not None
    use_pretrained_stage4 = stage4_model_path_arg is not None
    use_pretrained_stage5 = stage5_model_path_arg is not None

    if use_pretrained_stage1 and not stage1_model_path.exists():
        raise FileNotFoundError(f"Provided stage1 model path does not exist: {stage1_model_path}")
    if use_pretrained_stage2 and not stage2_model_path.exists():
        raise FileNotFoundError(f"Provided stage2 model path does not exist: {stage2_model_path}")
    if use_pretrained_stage3 and not stage3_model_path.exists():
        raise FileNotFoundError(f"Provided stage3 model path does not exist: {stage3_model_path}")
    if use_pretrained_stage4 and not stage4_model_path.exists():
        raise FileNotFoundError(f"Provided stage4 model path does not exist: {stage4_model_path}")
    if use_pretrained_stage5 and not stage5_model_path.exists():
        raise FileNotFoundError(f"Provided stage5 model path does not exist: {stage5_model_path}")

    train_dataset = UCIHAR_Dataset(root_path, split='train', subject_ids=train_subjects, preprocessing=preprocessing)
    val_dataset = UCIHAR_Dataset(root_path, split='train', subject_ids=val_subjects, preprocessing=preprocessing)
    test_dataset = UCIHAR_Dataset(root_path, split='test', subject_ids=None, preprocessing=preprocessing)

    cpu_count = os.cpu_count() or 1
    if performance_mode:
        num_workers = max(0, min(6, cpu_count - 2))
        prefetch_factor = 2
        pin_memory = device.type == 'cuda'
        persistent_workers = num_workers > 0
    else:
        num_workers = 0
        prefetch_factor = 2
        pin_memory = False
        persistent_workers = False

    common_loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        common_loader_kwargs['persistent_workers'] = persistent_workers
        common_loader_kwargs['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Performance mode: {performance_mode}")
    print(
        f"DataLoader settings: workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers if num_workers > 0 else False}, "
        f"prefetch_factor={prefetch_factor if num_workers > 0 else 'n/a'}"
    )

    freq_bins = train_dataset[0][0].shape[-1]
    class_counts = np.bincount(train_dataset.labels.astype(int), minlength=6).astype(np.float32)
    class_counts = np.clip(class_counts, 1.0, None)
    class_weights = class_counts.sum() / class_counts
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.detach().cpu().numpy().round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # STAGE 1
    print("\n" + "=" * 60)
    if use_pretrained_stage1:
        print("STAGE 1: Loading pretrained SeparableConvCNN")
        print(f"Checkpoint: {stage1_model_path}")
    else:
        print("STAGE 1: Training SeparableConvCNN")
    print("=" * 60)

    model_stage1 = SeparableConvCNN(num_classes=6, num_channels=6, freq_bins=freq_bins, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model_stage1.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    stage1_result = stage1_pipeline(
        model=model_stage1,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=stage1_model_path,
        num_epochs=epochs_stage1,
        patience=patience,
        min_delta=min_delta,
        log_every_n_epochs=log_every_n_epochs,
        wandb_run=wandb_run,
        use_pretrained=use_pretrained_stage1,
        device=device,
    )

    model_stage1.load_state_dict(torch.load(stage1_model_path, map_location=device))
    _, _, _, stage1_y_true, stage1_y_pred = _val_one_epoch(model_stage1, test_loader, criterion, device)
    _save_confusion_matrix_artifact(
        y_true=stage1_y_true,
        y_pred=stage1_y_pred,
        output_dir=None,
        stage_name='Stage 1',
        model_name='SeparableConvCNN',
        wandb_run=wandb_run,
        artifact_name=f"{model_path.stem}-stage1-confusion-matrix".replace('.', '_'),
        preprocessing=preprocessing,
    )

    stage1_best_val_loss = stage1_result['best_val_loss']
    stage1_best_epoch = stage1_result['best_epoch']
    stage1_test_loss = stage1_result['test_loss']
    stage1_test_acc = stage1_result['test_acc']
    stage1_test_f1 = stage1_result['test_f1']

    # STAGE 2
    print("\n" + "=" * 60)
    if use_pretrained_stage2:
        print("STAGE 2: Loading pretrained GumbelMaskSeparableConvCNN (input-bin pruning)")
        print(f"Checkpoint: {stage2_model_path}")
    else:
        print("STAGE 2: Training GumbelMaskSeparableConvCNN (input-bin pruning)")
    print("=" * 60)

    model_stage2 = GumbelMaskSeparableConvCNN(
        num_classes=6,
        num_channels=6,
        freq_bins=freq_bins,
        dropout=dropout,
        tau_start=tau_start,
        tau_end=tau_end,
    ).to(device)

    stage2_best_val_loss = None
    stage2_best_epoch = None

    if use_pretrained_stage2:
        model_stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
    else:
        print("\nLoading Stage 1 weights into Stage 2 model:")
        _load_weights_to_gumbel_model(model_stage2, torch.load(stage1_model_path, map_location=device))

        stage2_backbone_lr = lr * stage2_backbone_lr_factor
        gumbel_params = [model_stage2.bin_logits]
        gumbel_ids = {id(p) for p in gumbel_params}
        backbone_params = [p for p in model_stage2.parameters() if id(p) not in gumbel_ids]

        optimizer = torch.optim.Adam(
            [
                {'params': backbone_params, 'lr': stage2_backbone_lr},
                {'params': gumbel_params, 'lr': lr},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

        stage2_best_val_loss = float('inf')
        stage2_best_epoch = 0
        no_improve = 0

        for epoch in range(epochs_stage2):
            model_stage2.train()
            if hasattr(model_stage2, 'set_tau'):
                model_stage2.set_tau(epoch, epochs_stage2)

            train_loss_sum, train_correct, train_total = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model_stage2(x)
                loss = criterion(out, y)
                if model_stage2.mask_l1 is not None:
                    loss = loss + sparsity_weight_bin * model_stage2.mask_l1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                train_loss_sum += loss.item() * bs
                _, pred = out.max(1)
                train_total += bs
                train_correct += pred.eq(y).sum().item()

            train_loss = train_loss_sum / max(train_total, 1)
            train_acc = 100.0 * train_correct / max(train_total, 1)
            val_loss, val_acc, _ = _evaluate_classifier(model_stage2, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < stage2_best_val_loss - min_delta:
                stage2_best_val_loss = val_loss
                stage2_best_epoch = epoch + 1
                torch.save(model_stage2.state_dict(), stage2_model_path)
                no_improve = 0
            else:
                no_improve += 1

            mask_info = f"; Mask: {model_stage2.mask_l1.item():.2%}" if model_stage2.mask_l1 is not None else ""
            should_log_epoch = ((epoch + 1) % log_every_n_epochs == 0) or ((epoch + 1) == epochs_stage2)
            if should_log_epoch:
                print(
                    f"Epoch [{epoch+1}/{epochs_stage2}]: "
                    f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                    f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}" + mask_info
                )

            if wandb_run is not None and should_log_epoch:
                wandb_run.log({
                    'stage': 2,
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_loss': stage2_best_val_loss,
                    'lr_backbone': optimizer.param_groups[0]['lr'],
                    'lr_gumbel_bin': optimizer.param_groups[1]['lr'],
                    'mask_l1': model_stage2.mask_l1.item() if model_stage2.mask_l1 is not None else None,
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage2}] (patience={patience}).")
                break

    if not use_pretrained_stage2:
        model_stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
    stage2_test_loss, stage2_test_acc, stage2_test_f1 = _evaluate_classifier(model_stage2, test_loader, criterion, device)
    hard_bin_mask = _get_hard_bin_mask_from_model(model_stage2).detach().cpu()
    bin_keep_ratio = hard_bin_mask.mean().item()

    print('-' * 50)
    print('Stage 2 Summary:')
    if stage2_best_val_loss is not None:
        print(f"Best Val Loss: {stage2_best_val_loss:.4f} at Epoch {stage2_best_epoch}")
    else:
        print('Best Val Loss: not available (loaded pretrained stage2 checkpoint)')
    print(f"Test Loss: {stage2_test_loss:.4f} | Test Acc: {stage2_test_acc:.2f}% | Test F1 Macro: {stage2_test_f1:.4f}")
    print(f"Hard input bins kept: {(hard_bin_mask > 0.5).sum().item()}/{hard_bin_mask.numel()} ({bin_keep_ratio:.1%})")

    # STAGE 3
    print("\n" + "=" * 60)
    if use_pretrained_stage3:
        print('STAGE 3: Loading pretrained SeparableConvCNN on pruned input')
        print(f'Checkpoint: {stage3_model_path}')
    else:
        print('STAGE 3: Retraining SeparableConvCNN on pruned input')
    print('=' * 60)

    keep_indices = torch.where(hard_bin_mask > 0.5)[0]
    if keep_indices.numel() == 0:
        raise ValueError('Stage 2 pruned all bins; cannot run Stage 3 with empty input.')

    pruned_train_dataset = SlicedUCIDataset(train_dataset, keep_indices)
    pruned_val_dataset = SlicedUCIDataset(val_dataset, keep_indices)
    pruned_test_dataset = SlicedUCIDataset(test_dataset, keep_indices)

    pruned_train_loader = DataLoader(pruned_train_dataset, shuffle=True, **common_loader_kwargs)
    pruned_val_loader = DataLoader(pruned_val_dataset, shuffle=False, **common_loader_kwargs)
    pruned_test_loader = DataLoader(pruned_test_dataset, shuffle=False, **common_loader_kwargs)

    pruned_freq_bins = int(keep_indices.numel())
    model_stage3 = SeparableConvCNN(num_classes=6, num_channels=6, freq_bins=pruned_freq_bins, dropout=dropout).to(device)
    stage3_best_val_loss = None
    stage3_best_epoch = None

    if use_pretrained_stage3:
        model_stage3.load_state_dict(torch.load(stage3_model_path, map_location=device))
    else:
        stage2_state = torch.load(stage2_model_path, map_location=device)
        _load_matching_weights(model_stage3, stage2_state)

        optimizer = torch.optim.Adam(model_stage3.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        stage3_best_val_loss = float('inf')
        stage3_best_epoch = 0
        no_improve = 0

        for epoch in range(epochs_stage3):
            model_stage3.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            for x, y in pruned_train_loader:
                x, y = x.to(device), y.to(device)
                out = model_stage3(x)
                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                train_loss_sum += loss.item() * bs
                _, pred = out.max(1)
                train_total += bs
                train_correct += pred.eq(y).sum().item()

            train_loss = train_loss_sum / max(train_total, 1)
            train_acc = 100.0 * train_correct / max(train_total, 1)
            val_loss, val_acc, _ = _evaluate_classifier(model_stage3, pruned_val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < stage3_best_val_loss - min_delta:
                stage3_best_val_loss = val_loss
                stage3_best_epoch = epoch + 1
                torch.save(model_stage3.state_dict(), stage3_model_path)
                no_improve = 0
            else:
                no_improve += 1

            should_log_epoch = ((epoch + 1) % log_every_n_epochs == 0) or ((epoch + 1) == epochs_stage3)
            if should_log_epoch:
                print(
                    f"Epoch [{epoch+1}/{epochs_stage3}]: "
                    f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                    f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}"
                )

            if wandb_run is not None and should_log_epoch:
                wandb_run.log({
                    'stage': 3,
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_loss': stage3_best_val_loss,
                    'lr': optimizer.param_groups[0]['lr'],
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage3}] (patience={patience}).")
                break

        model_stage3.load_state_dict(torch.load(stage3_model_path, map_location=device))

    stage3_test_loss, stage3_test_acc, stage3_test_f1, stage3_y_true, stage3_y_pred = _val_one_epoch(
        model_stage3, pruned_test_loader, criterion, device
    )

    _save_confusion_matrix_artifact(
        y_true=stage3_y_true,
        y_pred=stage3_y_pred,
        output_dir=None,
        stage_name='Stage 3',
        model_name='SeparableConvCNN',
        wandb_run=wandb_run,
        artifact_name=f"{model_path.stem}-stage3-confusion-matrix".replace('.', '_'),
        preprocessing=preprocessing,
    )

    print('-' * 50)
    print('Stage 3 Summary:')
    if stage3_best_val_loss is not None:
        print(f"Best Val Loss: {stage3_best_val_loss:.4f} at Epoch {stage3_best_epoch}")
    else:
        print('Best Val Loss: not available (loaded pretrained stage3 checkpoint)')
    print(f"Test Loss: {stage3_test_loss:.4f} | Test Acc: {stage3_test_acc:.2f}% | Test F1 Macro: {stage3_test_f1:.4f}")

    # STAGE 4
    print("\n" + '=' * 60)
    if use_pretrained_stage4:
        print('STAGE 4: Loading pretrained GumbelChannelPruningCNN on pruned input')
        print(f'Checkpoint: {stage4_model_path}')
    else:
        print('STAGE 4: Training GumbelChannelPruningCNN on pruned input')
    print('=' * 60)

    model_stage4 = GumbelChannelPruningCNN(
        num_classes=6,
        num_channels=6,
        freq_bins=pruned_freq_bins,
        dropout=dropout,
        tau_start=tau_start,
        tau_end=tau_end,
    ).to(device)

    stage4_result = stage2_channel_gumbel_pruning_pipeline(
        model=model_stage4,
        train_loader=pruned_train_loader,
        val_loader=pruned_val_loader,
        test_loader=pruned_test_loader,
        criterion=criterion,
        checkpoint_path=stage4_model_path,
        stage1_checkpoint_path=stage3_model_path,
        lr=lr,
        backbone_lr_factor=stage4_backbone_lr_factor,
        sparsity_weight=sparsity_weight_channel,
        num_epochs=epochs_stage4,
        patience=patience,
        min_delta=min_delta,
        log_every_n_epochs=log_every_n_epochs,
        wandb_run=wandb_run,
        use_pretrained=use_pretrained_stage4,
        device=device,
    )

    stage4_best_val_loss = stage4_result['best_val_loss']
    stage4_best_epoch = stage4_result['best_epoch']
    stage4_test_loss = stage4_result['test_loss']
    stage4_test_acc = stage4_result['test_acc']
    stage4_test_f1 = stage4_result['test_f1']
    hard_channel_masks = stage4_result['hard_masks']
    pruning_stats = stage4_result['pruning_stats']

    final_channel_mask = {
        'block2': hard_channel_masks['block2'].detach().cpu().numpy(),
        'block3': hard_channel_masks['block3'].detach().cpu().numpy(),
        'block4': hard_channel_masks['block4'].detach().cpu().numpy(),
    }

    # STAGE 5
    print("\n" + '=' * 60)
    if use_pretrained_stage5:
        print('STAGE 5: Loading pretrained compact final model')
        print(f'Checkpoint: {stage5_model_path}')
    else:
        print('STAGE 5: Building compact pruned model and fine-tuning')
    print('=' * 60)

    stage5_model, keep_channel_indices = _build_pruned_channel_model_from_stage2(
        stage2_model=model_stage4,
        num_classes=6,
        dropout=dropout,
        device=device,
    )

    dense_reference_model = SeparableConvCNN(num_classes=6, num_channels=6, freq_bins=freq_bins, dropout=dropout)
    dense_param_count = _count_parameters(dense_reference_model)
    pruned_param_count = _count_parameters(stage5_model)
    param_reduction_pct = (1.0 - pruned_param_count / max(dense_param_count, 1)) * 100.0

    print(
        f"Stage 5 model size: {pruned_param_count:,} params vs dense {dense_param_count:,} params "
        f"({param_reduction_pct:.2f}% reduction)"
    )

    stage5_best_val_loss = None
    stage5_best_epoch = None

    if use_pretrained_stage5:
        stage5_model.load_state_dict(torch.load(stage5_model_path, map_location=device))
    else:
        loaded_params = list(stage5_model.parameters())
        optimizer = torch.optim.Adam([{'params': loaded_params, 'lr': lr * stage5_loaded_lr_factor}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

        stage5_best_val_loss = float('inf')
        stage5_best_epoch = 0
        no_improve = 0

        for epoch in range(epochs_stage5):
            stage5_model.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            for x, y in pruned_train_loader:
                x, y = x.to(device), y.to(device)
                out = stage5_model(x)
                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                train_loss_sum += loss.item() * bs
                _, pred = out.max(1)
                train_total += bs
                train_correct += pred.eq(y).sum().item()

            train_loss = train_loss_sum / max(train_total, 1)
            train_acc = 100.0 * train_correct / max(train_total, 1)

            val_loss, val_acc, _ = _evaluate_classifier(stage5_model, pruned_val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < stage5_best_val_loss - min_delta:
                stage5_best_val_loss = val_loss
                stage5_best_epoch = epoch + 1
                torch.save(stage5_model.state_dict(), stage5_model_path)
                no_improve = 0
            else:
                no_improve += 1

            should_log_epoch = ((epoch + 1) % log_every_n_epochs == 0) or ((epoch + 1) == epochs_stage5)
            if should_log_epoch:
                print(
                    f"Epoch [{epoch+1}/{epochs_stage5}]: "
                    f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                    f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}"
                )

            if wandb_run is not None and should_log_epoch:
                wandb_run.log({
                    'stage': 5,
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_loss': stage5_best_val_loss,
                    'lr_loaded_weights': optimizer.param_groups[0]['lr'],
                    'model_param_count_pruned': pruned_param_count,
                    'model_param_reduction_pct': param_reduction_pct,
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage5}] (patience={patience}).")
                break

        stage5_model.load_state_dict(torch.load(stage5_model_path, map_location=device))

    stage5_test_loss, stage5_test_acc, stage5_test_f1, stage5_y_true, stage5_y_pred = _val_one_epoch(
        stage5_model, pruned_test_loader, criterion, device
    )

    _save_confusion_matrix_artifact(
        y_true=stage5_y_true,
        y_pred=stage5_y_pred,
        output_dir=None,
        stage_name='Stage 5',
        model_name='PrunedSeparableConvCNN',
        wandb_run=wandb_run,
        artifact_name=f"{model_path.stem}-stage5-confusion-matrix".replace('.', '_'),
        preprocessing=preprocessing,
    )

    if wandb_run is not None:
        wandb_run.log({
            'stage5_test_loss': stage5_test_loss,
            'stage5_test_acc': stage5_test_acc,
            'stage5_test_f1': stage5_test_f1,
            'model_param_count_dense': dense_param_count,
            'model_param_count_pruned': pruned_param_count,
            'model_param_reduction_pct': param_reduction_pct,
            'hard_bin_mask': hard_bin_mask.tolist(),
            'final_channel_mask_block2': final_channel_mask['block2'].tolist(),
            'final_channel_mask_block3': final_channel_mask['block3'].tolist(),
            'final_channel_mask_block4': final_channel_mask['block4'].tolist(),
        })

    print('\n' + '=' * 60)
    print('FIVE-STAGE TRAINING COMPLETE')
    print('=' * 60)
    print(f"Stage 1 (SeparableConvCNN): Test Acc: {stage1_test_acc:.2f}% | F1: {stage1_test_f1:.4f}")
    print(f"Stage 2 (Input Bin Pruning): Test Acc: {stage2_test_acc:.2f}% | F1: {stage2_test_f1:.4f}")
    print(f"Stage 3 (Pruned Input Retrain): Test Acc: {stage3_test_acc:.2f}% | F1: {stage3_test_f1:.4f}")
    print(f"Stage 4 (Channel Pruning on Pruned Input): Test Acc: {stage4_test_acc:.2f}% | F1: {stage4_test_f1:.4f}")
    print(f"Stage 5 (Compact Model): Test Acc: {stage5_test_acc:.2f}% | F1: {stage5_test_f1:.4f}")

    return {
        'stage1': {
            'model': 'SeparableConvCNN',
            'best_val_loss': stage1_best_val_loss,
            'best_epoch': stage1_best_epoch,
            'test_loss': stage1_test_loss,
            'test_acc': stage1_test_acc,
            'test_f1_macro': stage1_test_f1,
            'model_path': str(stage1_model_path),
            'loaded_from_checkpoint': use_pretrained_stage1,
        },
        'stage2': {
            'model': 'GumbelMaskSeparableConvCNN',
            'best_val_loss': stage2_best_val_loss,
            'best_epoch': stage2_best_epoch,
            'test_loss': stage2_test_loss,
            'test_acc': stage2_test_acc,
            'test_f1_macro': stage2_test_f1,
            'model_path': str(stage2_model_path),
            'hard_bin_mask': hard_bin_mask.numpy(),
            'bin_keep_ratio': bin_keep_ratio,
            'loaded_from_checkpoint': use_pretrained_stage2,
        },
        'stage3': {
            'model': 'SeparableConvCNN',
            'best_val_loss': stage3_best_val_loss,
            'best_epoch': stage3_best_epoch,
            'test_loss': stage3_test_loss,
            'test_acc': stage3_test_acc,
            'test_f1_macro': stage3_test_f1,
            'model_path': str(stage3_model_path),
            'loaded_from_checkpoint': use_pretrained_stage3,
        },
        'stage4': {
            'model': 'GumbelChannelPruningCNN',
            'best_val_loss': stage4_best_val_loss,
            'best_epoch': stage4_best_epoch,
            'test_loss': stage4_test_loss,
            'test_acc': stage4_test_acc,
            'test_f1_macro': stage4_test_f1,
            'model_path': str(stage4_model_path),
            'final_mask': final_channel_mask,
            'pruning_stats': pruning_stats,
            'loaded_from_checkpoint': use_pretrained_stage4,
        },
        'stage5': {
            'model': 'PrunedSeparableConvCNN',
            'best_val_loss': stage5_best_val_loss,
            'best_epoch': stage5_best_epoch,
            'test_loss': stage5_test_loss,
            'test_acc': stage5_test_acc,
            'test_f1_macro': stage5_test_f1,
            'model_path': str(stage5_model_path),
            'loaded_from_checkpoint': use_pretrained_stage5,
            'dense_param_count': dense_param_count,
            'pruned_param_count': pruned_param_count,
            'param_reduction_pct': param_reduction_pct,
            'block2_keep': int(keep_channel_indices['block2'].numel()),
            'block3_keep': int(keep_channel_indices['block3'].numel()),
            'block4_keep': int(keep_channel_indices['block4'].numel()),
        },
    }




    
