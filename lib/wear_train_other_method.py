import torch
import torch.nn as nn
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.fftpack import dct
from scipy.signal import butter, filtfilt
from lib.model import SeparableConvCNN, RandomChannelPruningCNN, PrunedSeparableConvCNN


# Label mapping: 18 fine-grained WEAR labels → 8 merged classes
LABEL_MAP = {
    'jogging': 0,
    'jogging (sidesteps)': 0,
    'jogging (skipping)': 0,
    'jogging (butt-kicks)': 0,
    'jogging (rotating arms)': 0,
    'stretching (lunging)': 1,
    'stretching (hamstrings)': 1,
    'stretching (triceps)': 1,
    'stretching (shoulders)': 1,
    'stretching (lumbar rotation)': 1,
    'lunges': 2,
    'lunges (complex)': 2,
    'sit-ups': 3,
    'sit-ups (complex)': 3,
    'push-ups': 4,
    'push-ups (complex)': 4,
    'burpees': 5,
    'bench-dips': 6,
    'null': 7,
}

#####################################################
# DATASET 
#####################################################

def _remove_gravity(signal, cutoff=0.3, fs=50, order=3):
    """High-pass filter to remove gravity (same as UCI-HAR preprocessing)."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, signal, axis=0)

def _load_and_window_subject_csv(file_path, window_size=100, step_size=50):
    """
    Load a single subject's CSV, extract left_arm_acc_x/y/z and label,
    map labels via LABEL_MAP, and apply a sliding window.

    Args:
        file_path: Path to the subject's CSV file
        window_size: samples per window (default 100 = 2s at 50Hz)
        step_size: sliding step (default 50 = 50% overlap)

    Returns:
        signals: np.ndarray of shape (num_windows, 6, window_size)
                 (lx, ly, lz, rx, ry, rz)
        labels:  np.ndarray of shape (num_windows,)
    """
    acc_data = []
    mapped_labels = []

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)

        try:
            lx_idx = headers.index('left_arm_acc_x')
            ly_idx = headers.index('left_arm_acc_y')
            lz_idx = headers.index('left_arm_acc_z')
            rx_idx = headers.index('right_arm_acc_x')
            ry_idx = headers.index('right_arm_acc_y')
            rz_idx = headers.index('right_arm_acc_z')
            lbl_idx = headers.index('label')
        except ValueError as e:
            print(f"Error finding columns in {file_path}: {e}")
            return np.array([]), np.array([])

        col_indices = [lx_idx, ly_idx, lz_idx, rx_idx, ry_idx, rz_idx]

        for row in reader:
            # Skip rows with missing accelerometer values
            if any(not row[i].strip() for i in col_indices):
                continue

            acc_data.append([float(row[i]) for i in col_indices])

            lbl_str = row[lbl_idx].strip()
            if lbl_str in LABEL_MAP:
                mapped_labels.append(LABEL_MAP[lbl_str])
            else:
                mapped_labels.append(-1)

    acc_data = np.array(acc_data, dtype=np.float32)
    mapped_labels = np.array(mapped_labels, dtype=np.int64)

    # -----------------------------
    # Step 1: Filter gravity (0.3Hz high-pass)
    # -----------------------------
    if len(acc_data) > 0:
        acc_data = _remove_gravity(acc_data)

        # Step 2: Min-Max normalization per channel to [-1, 1]
        # This standardizes the data to be bounded in [-1, 1] similar to UCI-HAR,
        # preventing large inputs from destabilizing the Gumbel mask gradients.
        # min_vals = np.min(acc_data, axis=0)
        # max_vals = np.max(acc_data, axis=0)
        # range_vals = max_vals - min_vals
        # range_vals[range_vals == 0] = 1.0
        # acc_data = 2.0 * (acc_data - min_vals) / range_vals - 1.0

    num_samples = len(acc_data)

    windows_signals = []
    windows_labels = []

    # Sliding window
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size

        window_signal = acc_data[start:end]            # (window_size, 6)
        window_label_seq = mapped_labels[start:end]    # (window_size,)

        # Majority-vote label (offset by +1 to support -1 in bincount)
        counts = np.bincount(window_label_seq + 1)
        mode_idx = counts.argmax()
        mode_label = mode_idx - 1

        # Discard windows where majority label is unknown
        if mode_label == -1:
            continue

        windows_signals.append(window_signal.T)  # (6, window_size)
        windows_labels.append(mode_label)

    return np.array(windows_signals, dtype=np.float32), np.array(windows_labels, dtype=np.int64)

class WEAR_Dataset(Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        subject_ids=None,
        preprocessing='fft',
        window_size=100,
        step_size=50,
    ):
        """
        Load WEAR data for specific subjects, apply sliding window, then FFT/DCT/IHW/no.

        Args:
            root_path: Path to WEAR dataset root (containing train/ and test/ subdirs)
            split: 'train' or 'test'
            subject_ids: List of subject IDs to load (None = all in split)
            preprocessing: 'fft', 'dct', 'ihw', or 'no'
            window_size: samples per window (default 100 = 2s at 50Hz)
            step_size: sliding step (default 50 = 50% overlap)
        """
        self.root_path = Path(root_path)
        self.split_path = self.root_path / split
        self.preprocessing = preprocessing

        # Determine which subjects to load
        if subject_ids is None:
            subject_file = self.split_path / f"subject_{split}.txt"
            subject_ids = sorted(np.loadtxt(subject_file, dtype=int).tolist())
            if isinstance(subject_ids, int):
                subject_ids = [subject_ids]

        all_signals = []
        all_labels = []
        all_subjects = []

        for sbj_id in subject_ids:
            file_path = self.split_path / f"sbj_{sbj_id}.csv"
            if not file_path.exists():
                print(f"Warning: {file_path} not found. Skipping.")
                continue

            signals, labels = _load_and_window_subject_csv(
                file_path, window_size=window_size, step_size=step_size
            )

            if len(signals) > 0:
                all_signals.append(signals)
                all_labels.append(labels)
                all_subjects.extend([sbj_id] * len(labels))

        if len(all_signals) > 0:
            self.signals = np.concatenate(all_signals, axis=0)   # (N, 6, window_size)
            self.labels = np.concatenate(all_labels, axis=0)     # (N,)
            self.subjects = np.array(all_subjects)
        else:
            self.signals = np.array([])
            self.labels = np.array([])
            self.subjects = np.array([])

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
    def _compute_dct(signal):
        dct_vals = dct(signal, type=2, axis=-1, norm='ortho')
        return np.abs(dct_vals)

    @staticmethod
    def _compute_ihw(signal, fixed_point_scale=1024):
        # Fixed-point scaling keeps sub-integer precision while using integer Haar lifting.
        current = np.rint(signal * fixed_point_scale).astype(np.int64)
        details = []

        # Apply dyadic levels while the current length is even.
        while current.shape[-1] > 1 and current.shape[-1] % 2 == 0:
            even = current[..., 0::2]
            odd = current[..., 1::2]

            detail = odd - even
            approx = even + (detail // 2)

            details.append(detail)
            current = approx

        coeffs = [current] + details[::-1]
        ihw = np.concatenate(coeffs, axis=-1)
        return ihw.astype(np.float32) / float(fixed_point_scale)

    def __getitem__(self, idx):
        signal = self.signals[idx]  # (3, window_size)

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

#####################################################
# TRAINING 
#####################################################
def _load_stage1_weights_to_stage2_model(stage2_model, stage1_state_dict):
    """
    Load weights from SeparableConvCNN (stage 1) into the next model (stage 2) selectively.
    
    Args:
        stage2_model: Model to load weights into
        stage1_state_dict: State dict from trained SeparableConvCNN model
    """
    # Get model current state dict
    stage2_state_dict = stage2_model.state_dict()
    
    # Load all weights that exist in both models
    for name, param in stage1_state_dict.items():
        if name in stage2_state_dict and stage2_state_dict[name].shape==param.shape:
            stage2_state_dict[name] = param
            print(f"  Loaded: {name}")
        else:
            print(f"  Skipped: {name}")
    
    # Load the state dict into model
    stage2_model.load_state_dict(stage2_state_dict)
    return stage2_model

def train_loso_wear_two_stage_random_pruning(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    LOSO training for WEAR dataset with channel-pruning stage 2:
    Stage 1: SeparableConvCNN (no mask)
    Stage 2: GumbelChannelPruningCNN (mask channels after conv blocks)
    
    """
    # ARGS
    preprocessing = train_kwargs.get('preprocessing', 'fft')
    epochs_stage1 = train_kwargs.get('epochs_stage1', 60)
    epochs_stage2 = train_kwargs.get('epochs_stage2', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    pruning_ratio = train_kwargs.get('pruning_ratio', 0.3)
    mask_seed = train_kwargs.get('mask_seed', 42)
    model_path = Path(train_kwargs.get('model_path', './models/best_wear_model_two_stage_channel.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dropout = train_kwargs.get('dropout', 0.4)
    stage2_backbone_lr_factor = train_kwargs.get('stage2_backbone_lr_factor', 0.1)
    
    stage1_model_path_arg = train_kwargs.get('stage1_model_path')
    if stage1_model_path_arg is not None:
        stage1_model_path = Path(stage1_model_path_arg).expanduser()
    else:
        stage1_model_path = model_path.parent / f"{model_path.stem}_stage1.pth"
    stage1_model_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_model_path = model_path
    use_pretrained_stage1 = stage1_model_path_arg is not None

    if use_pretrained_stage1 and not stage1_model_path.exists():
        raise FileNotFoundError(f"Provided stage1 model path does not exist: {stage1_model_path}")

    # INIT DATASET
    train_dataset = WEAR_Dataset(
        root_path,
        split='train',
        subject_ids=train_subjects,
        preprocessing=preprocessing,
    )
    val_dataset = WEAR_Dataset(
        root_path,
        split='train',
        subject_ids=val_subjects,
        preprocessing=preprocessing,
    )
    test_dataset = WEAR_Dataset(
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

    freq_bins = train_dataset[0][0].shape[-1]
    
    # STAGE 1
    print("\n" + "=" * 60)
    if use_pretrained_stage1:
        print("STAGE 1: Loading pretrained SeparableConvCNN")
        print(f"Checkpoint: {stage1_model_path}")
    else:
        print("STAGE 1: Training SeparableConvCNN (without Gumbel mask)")
    print("=" * 60)

    model_stage1 = SeparableConvCNN(
        num_classes=8,
        num_channels=6,
        freq_bins=freq_bins,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    stage1_best_val_loss = None
    stage1_best_epoch = None

    if not use_pretrained_stage1:
        optimizer_stage1 = torch.optim.Adam(model_stage1.parameters(), lr=lr)
        scheduler_stage1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_stage1, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

        stage1_best_val_loss = float('inf')
        stage1_best_epoch = 0
        epochs_no_improve = 0

        print("-" * 50)
        for epoch in range(epochs_stage1):
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0
            model_stage1.train()

            for fft_mag, labels in train_dataloader:
                fft_mag, labels = fft_mag.to(device), labels.to(device)
                outputs = model_stage1(fft_mag)
                loss = criterion(outputs, labels)

                optimizer_stage1.zero_grad()
                loss.backward()
                optimizer_stage1.step()

                train_loss_sum += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total * 100.0

            model_stage1.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for fft_mag, labels in val_dataloader:
                    fft_mag, labels = fft_mag.to(device), labels.to(device)
                    outputs = model_stage1(fft_mag)
                    loss = criterion(outputs, labels)

                    val_loss_sum += loss.item() * labels.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = 100.0 * val_correct / max(val_total, 1)
            scheduler_stage1.step(val_loss)

            if val_loss < stage1_best_val_loss - min_delta:
                stage1_best_val_loss = val_loss
                torch.save(model_stage1.state_dict(), stage1_model_path)
                stage1_best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(
                f"Epoch [{epoch+1}/{epochs_stage1}]: "
                f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "stage": 1,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": stage1_best_val_loss,
                    "lr": optimizer_stage1.param_groups[0]["lr"],
                })

            if epochs_no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage1}] (patience={patience}).")
                break

    model_stage1.load_state_dict(torch.load(stage1_model_path, map_location=device))
    model_stage1.eval()

    test_loss_sum, test_correct, test_total = 0.0, 0, 0
    all_preds_stage1 = []
    all_labels_stage1 = []
    with torch.no_grad():
        for fft_mag, labels in test_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)
            outputs = model_stage1(fft_mag)
            loss = criterion(outputs, labels)

            bs = labels.size(0)
            test_loss_sum += loss.item() * bs
            _, preds = outputs.max(1)
            test_total += bs
            test_correct += preds.eq(labels).sum().item()
            all_preds_stage1.extend(preds.cpu().numpy())
            all_labels_stage1.extend(labels.cpu().numpy())

    test_loss_stage1 = test_loss_sum / max(test_total, 1)
    test_acc_stage1 = 100.0 * test_correct / max(test_total, 1)
    test_f1_stage1 = f1_score(all_labels_stage1, all_preds_stage1, average='macro')

    print("-" * 50)
    print("Stage 1 Summary:")
    if stage1_best_val_loss is not None:
        print(f"Best Val Loss: {stage1_best_val_loss:.4f} at Epoch {stage1_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage1 checkpoint)")
    print(f"Test Loss: {test_loss_stage1:.4f} | Test Acc: {test_acc_stage1:.2f}% | Test F1 Macro: {test_f1_stage1:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "stage1_checkpoint_path": str(stage1_model_path),
            "stage1_loaded_from_checkpoint": use_pretrained_stage1,
            "stage1_test_loss": test_loss_stage1,
            "stage1_test_acc": test_acc_stage1,
            "stage1_test_f1": test_f1_stage1,
        })

    # STAGE 2
    print("\n" + "=" * 60)
    print("STAGE 2: Training RandomChannelPruningCNN (channel masks after blocks)")
    print("=" * 60)

    model_stage2 = RandomChannelPruningCNN(
        num_classes=8,
        num_channels=6,
        freq_bins=freq_bins,
        dropout=dropout,
        pruning_ratio=pruning_ratio,
        mask_seed=mask_seed,
    ).to(device)

    print("\nLoading Stage 1 weights into Stage 2 model:")
    stage1_state_dict = torch.load(stage1_model_path, map_location=device)
    _load_stage1_weights_to_stage2_model(model_stage2, stage1_state_dict)

    stage2_backbone_lr = lr * stage2_backbone_lr_factor
    stage2_named_params = list(model_stage2.named_parameters())
    gumbel_params = [p for n, p in stage2_named_params if n.startswith('chan_logits_')]
    gumbel_param_ids = {id(p) for p in gumbel_params}
    backbone_params = [p for _, p in stage2_named_params if id(p) not in gumbel_param_ids]

    optimizer_stage2 = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": stage2_backbone_lr},
            {"params": gumbel_params, "lr": lr},
        ]
    )
    scheduler_stage2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage2, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    print(
        f"Stage 2 LR setup -> backbone: {stage2_backbone_lr:.2e}, "
        f"gumbel(chan_logits): {lr:.2e}"
    )

    stage2_best_val_loss = float('inf')
    stage2_best_epoch = 0
    epochs_no_improve = 0

    print("-" * 50)
    for epoch in range(epochs_stage2):
        if hasattr(model_stage2, 'set_tau'):
            model_stage2.set_tau(epoch, epochs_stage2)

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        model_stage2.train()

        for fft_mag, labels in train_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)
            outputs = model_stage2(fft_mag)
            loss = criterion(outputs, labels)

            if hasattr(model_stage2, 'get_sparsity_loss'):
                loss = loss + sparsity_weight * model_stage2.get_sparsity_loss()

            optimizer_stage2.zero_grad()
            loss.backward()
            optimizer_stage2.step()

            train_loss_sum += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total * 100.0

        model_stage2.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for fft_mag, labels in val_dataloader:
                fft_mag, labels = fft_mag.to(device), labels.to(device)
                outputs = model_stage2(fft_mag)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        scheduler_stage2.step(val_loss)

        if val_loss < stage2_best_val_loss - min_delta:
            stage2_best_val_loss = val_loss
            torch.save(model_stage2.state_dict(), stage2_model_path)
            stage2_best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        mask_info = ""
        if hasattr(model_stage2, 'get_sparsity_loss'):
            mask_fraction = model_stage2.get_sparsity_loss().item()
            tau_info = f" (tau={model_stage2.current_tau:.2f})" if hasattr(model_stage2, 'current_tau') else ""
            mask_info = f"; Soft On-Prob: {mask_fraction:.2%}" + tau_info

        print(
            f"Epoch [{epoch+1}/{epochs_stage2}]: "
            f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
            f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}" + mask_info
        )

        if wandb_run is not None:
            log_payload = {
                "stage": 2,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": stage2_best_val_loss,
                "lr_backbone": optimizer_stage2.param_groups[0]["lr"],
                "lr_gumbel": optimizer_stage2.param_groups[1]["lr"],
            }
            if hasattr(model_stage2, 'get_pruning_stats'):
                log_payload.update(model_stage2.get_pruning_stats())
            wandb_run.log(log_payload)

        if epochs_no_improve >= patience:
            print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage2}] (patience={patience}).")
            break

    model_stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
    model_stage2.eval()

    test_loss_sum, test_correct, test_total = 0.0, 0, 0
    all_preds_stage2 = []
    all_labels_stage2 = []
    with torch.no_grad():
        for fft_mag, labels in test_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)
            outputs = model_stage2(fft_mag)
            loss = criterion(outputs, labels)

            bs = labels.size(0)
            test_loss_sum += loss.item() * bs
            _, preds = outputs.max(1)
            test_total += bs
            test_correct += preds.eq(labels).sum().item()
            all_preds_stage2.extend(preds.cpu().numpy())
            all_labels_stage2.extend(labels.cpu().numpy())

    test_loss_stage2 = test_loss_sum / max(test_total, 1)
    test_acc_stage2 = 100.0 * test_correct / max(test_total, 1)
    test_f1_stage2 = f1_score(all_labels_stage2, all_preds_stage2, average='macro')

    print("-" * 50)
    print("Stage 2 Summary:")
    print(f"Best Val Loss: {stage2_best_val_loss:.4f} at Epoch {stage2_best_epoch}")
    print(f"Test Loss: {test_loss_stage2:.4f} | Test Acc: {test_acc_stage2:.2f}% | Test F1 Macro: {test_f1_stage2:.4f}")

    if hasattr(model_stage2, 'get_hard_masks'):
        hard_masks = model_stage2.get_hard_masks()
        final_mask = {
            "block2": hard_masks["block2"].detach().cpu().numpy(),
            "block3": hard_masks["block3"].detach().cpu().numpy(),
            "block4": hard_masks["block4"].detach().cpu().numpy(),
        }
    else:
        final_mask = {
            "block2": model_stage2.last_mask_2.detach().cpu().numpy(),
            "block3": model_stage2.last_mask_3.detach().cpu().numpy(),
            "block4": model_stage2.last_mask_4.detach().cpu().numpy(),
        }

    if hasattr(model_stage2, 'get_pruning_stats'):
        pruning_stats = model_stage2.get_pruning_stats()
        
        c2 = int(model_stage2.mask2.sum().item())
        c3 = int(model_stage2.mask3.sum().item())
        c4 = int(model_stage2.mask4.sum().item())
        
        pruned_model = PrunedSeparableConvCNN(
            num_classes=8,
            num_channels=6,
            block2_channels=c2,
            block3_channels=c3,
            block4_channels=c4,
        )
        params_before = sum(p.numel() for p in model_stage1.parameters() if p.requires_grad)
        params_after = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        pruning_stats['params_before_pruning'] = params_before
        pruning_stats['params_after_pruning'] = params_after

        print("\nFinal Pruning Statistics:")
        for k, v in pruning_stats.items():
            if 'params' in k:
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.2f}")
    else:
        pruning_stats = None

    if wandb_run is not None:
        log_payload = {
            "stage2_test_loss": test_loss_stage2,
            "stage2_test_acc": test_acc_stage2,
            "stage2_test_f1": test_f1_stage2,
        }
        if pruning_stats is not None:
            log_payload.update(pruning_stats)
        wandb_run.log(log_payload)

    print("\n" + "=" * 60)
    print("TWO-STAGE CHANNEL PRUNING TRAINING COMPLETE")
    print("=" * 60)
    print(f"Stage 1 (SeparableConvCNN): Test Acc: {test_acc_stage1:.2f}% | F1: {test_f1_stage1:.4f}")
    print(f"Stage 2 (GumbelChannelPruningCNN): Test Acc: {test_acc_stage2:.2f}% | F1: {test_f1_stage2:.4f}")
    print(f"Improvement: {test_acc_stage2 - test_acc_stage1:.2f}%")

    return {
        "stage1": {
            "model": "SeparableConvCNN",
            "best_val_loss": stage1_best_val_loss,
            "best_epoch": stage1_best_epoch,
            "test_loss": test_loss_stage1,
            "test_acc": test_acc_stage1,
            "test_f1_macro": test_f1_stage1,
            "model_path": str(stage1_model_path),
            "loaded_from_checkpoint": use_pretrained_stage1,
        },
        "stage2": {
            "model": "GumbelChannelPruningCNN",
            "best_val_loss": stage2_best_val_loss,
            "best_epoch": stage2_best_epoch,
            "test_loss": test_loss_stage2,
            "test_acc": test_acc_stage2,
            "test_f1_macro": test_f1_stage2,
            "model_path": str(stage2_model_path),
            "final_mask": final_mask,
            "pruning_stats": pruning_stats,
        },
    }


def get_l1_masks(model, pruning_ratio=0.3):
    """
    Computes L1-norm based masks for block 2, 3, and 4 pointwise convolutions.
    Returns float masks (1.0 for keep, 0.0 for pruned).
    """
    masks = {}
    for block_num, conv_layer in [(2, model.sep_conv2), (3, model.sep_conv3), (4, model.sep_conv4)]:
        weight = conv_layer.pointwise.weight.data # shape: (out_channels, in_channels, 1)
        l1_norm = weight.abs().sum(dim=(1, 2))
        num_channels = l1_norm.shape[0]
        keep_count = max(1, int(round(num_channels * (1.0 - pruning_ratio))))
        _, top_indices = torch.topk(l1_norm, keep_count)
        mask = torch.zeros(num_channels, device=weight.device)
        mask[top_indices] = 1.0
        masks[f'block{block_num}'] = mask
    return masks

def copy_weights_pruned(base_model, pruned_model, masks):
    """
    Physically copy weights from base_model to pruned_model using the provided masks.
    """
    idx2 = masks['block2'].nonzero(as_tuple=True)[0]
    idx3 = masks['block3'].nonzero(as_tuple=True)[0]
    idx4 = masks['block4'].nonzero(as_tuple=True)[0]
    
    state_desc = pruned_model.state_dict()
    base_desc = base_model.state_dict()
    
    # 1. Stem and block 1 - No pruning
    state_desc['sep_conv1.depthwise.weight'].copy_(base_desc['sep_conv1.depthwise.weight'])
    state_desc['sep_conv1.pointwise.weight'].copy_(base_desc['sep_conv1.pointwise.weight'])
    state_desc['bn1.weight'].copy_(base_desc['bn1.weight'])
    state_desc['bn1.bias'].copy_(base_desc['bn1.bias'])
    state_desc['bn1.running_mean'].copy_(base_desc['bn1.running_mean'])
    state_desc['bn1.running_var'].copy_(base_desc['bn1.running_var'])
    state_desc['bn1.num_batches_tracked'].copy_(base_desc['bn1.num_batches_tracked'])

    # 2. Block 2 - Prune output channels based on mask2
    state_desc['sep_conv2.depthwise.weight'].copy_(base_desc['sep_conv2.depthwise.weight'])
    state_desc['sep_conv2.pointwise.weight'].copy_(base_desc['sep_conv2.pointwise.weight'][idx2, :, :])
    state_desc['bn2.weight'].copy_(base_desc['bn2.weight'][idx2])
    state_desc['bn2.bias'].copy_(base_desc['bn2.bias'][idx2])
    state_desc['bn2.running_mean'].copy_(base_desc['bn2.running_mean'][idx2])
    state_desc['bn2.running_var'].copy_(base_desc['bn2.running_var'][idx2])
    state_desc['bn2.num_batches_tracked'].copy_(base_desc['bn2.num_batches_tracked'])

    # 3. Block 3 - Prune input channels based on mask2, output channels based on mask3
    state_desc['sep_conv3.depthwise.weight'].copy_(base_desc['sep_conv3.depthwise.weight'][idx2, :, :])
    state_desc['sep_conv3.pointwise.weight'].copy_(base_desc['sep_conv3.pointwise.weight'][idx3][:, idx2, :])
    state_desc['bn3.weight'].copy_(base_desc['bn3.weight'][idx3])
    state_desc['bn3.bias'].copy_(base_desc['bn3.bias'][idx3])
    state_desc['bn3.running_mean'].copy_(base_desc['bn3.running_mean'][idx3])
    state_desc['bn3.running_var'].copy_(base_desc['bn3.running_var'][idx3])
    state_desc['bn3.num_batches_tracked'].copy_(base_desc['bn3.num_batches_tracked'])

    # 4. Block 4 - Prune input channels based on mask3, output channels based on mask4
    state_desc['sep_conv4.depthwise.weight'].copy_(base_desc['sep_conv4.depthwise.weight'][idx3, :, :])
    state_desc['sep_conv4.pointwise.weight'].copy_(base_desc['sep_conv4.pointwise.weight'][idx4][:, idx3, :])
    state_desc['bn4.weight'].copy_(base_desc['bn4.weight'][idx4])
    state_desc['bn4.bias'].copy_(base_desc['bn4.bias'][idx4])
    state_desc['bn4.running_mean'].copy_(base_desc['bn4.running_mean'][idx4])
    state_desc['bn4.running_var'].copy_(base_desc['bn4.running_var'][idx4])
    state_desc['bn4.num_batches_tracked'].copy_(base_desc['bn4.num_batches_tracked'])

    # 5. FC layers
    state_desc['fc1.weight'].copy_(base_desc['fc1.weight'][:, idx4])
    state_desc['fc1.bias'].copy_(base_desc['fc1.bias'])
    state_desc['fc2.weight'].copy_(base_desc['fc2.weight'])
    state_desc['fc2.bias'].copy_(base_desc['fc2.bias'])

    pruned_model.load_state_dict(state_desc)


def train_loso_wear_three_stage_static_pruning(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    preprocessing = train_kwargs.get('preprocessing', 'fft')
    epochs_stage1 = train_kwargs.get('epochs_stage1', 60)
    epochs_stage3 = train_kwargs.get('epochs_stage3', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    pruning_ratio = train_kwargs.get('pruning_ratio', 0.3)
    model_path = Path(train_kwargs.get('model_path', './models/best_wear_model_three_stage_static.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dropout = train_kwargs.get('dropout', 0.4)
    
    stage1_model_path_arg = train_kwargs.get('stage1_model_path')
    if stage1_model_path_arg is not None:
        stage1_model_path = Path(stage1_model_path_arg).expanduser()
    else:
        stage1_model_path = model_path.parent / f"{model_path.stem}_stage1.pth"
    stage1_model_path.parent.mkdir(parents=True, exist_ok=True)
    stage3_model_path = model_path
    use_pretrained_stage1 = stage1_model_path_arg is not None

    if use_pretrained_stage1 and not stage1_model_path.exists():
        raise FileNotFoundError(f"Provided stage1 model path does not exist: {stage1_model_path}")

    # INIT DATASET
    train_dataset = WEAR_Dataset(root_path, split='train', subject_ids=train_subjects, preprocessing=preprocessing)
    val_dataset = WEAR_Dataset(root_path, split='train', subject_ids=val_subjects, preprocessing=preprocessing)
    test_dataset = WEAR_Dataset(root_path, split='test', subject_ids=None, preprocessing=preprocessing)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    freq_bins = train_dataset[0][0].shape[-1]
    
    # STAGE 1
    print("\n" + "=" * 60)
    if use_pretrained_stage1:
        print("STAGE 1: Loading pretrained SeparableConvCNN")
    else:
        print("STAGE 1: Training SeparableConvCNN (without masks)")
    print("=" * 60)

    model_stage1 = SeparableConvCNN(num_classes=8, num_channels=6, freq_bins=freq_bins, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    stage1_best_val_loss = None
    stage1_best_epoch = None

    if not use_pretrained_stage1:
        optimizer_stage1 = torch.optim.Adam(model_stage1.parameters(), lr=lr)
        scheduler_stage1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_stage1, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        stage1_best_val_loss = float('inf')
        stage1_best_epoch = 0
        epochs_no_improve = 0
        for epoch in range(epochs_stage1):
            train_loss_sum, train_correct, train_total = 0.0, 0, 0
            model_stage1.train()
            for fft_mag, labels in train_dataloader:
                fft_mag, labels = fft_mag.to(device), labels.to(device)
                outputs = model_stage1(fft_mag)
                loss = criterion(outputs, labels)
                optimizer_stage1.zero_grad()
                loss.backward()
                optimizer_stage1.step()
                train_loss_sum += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total * 100.0

            model_stage1.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for fft_mag, labels in val_dataloader:
                    fft_mag, labels = fft_mag.to(device), labels.to(device)
                    outputs = model_stage1(fft_mag)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * labels.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = 100.0 * val_correct / max(val_total, 1)
            scheduler_stage1.step(val_loss)

            if val_loss < stage1_best_val_loss - min_delta:
                stage1_best_val_loss = val_loss
                torch.save(model_stage1.state_dict(), stage1_model_path)
                stage1_best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if wandb_run is not None:
                wandb_run.log({
                    "stage": 1,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": stage1_best_val_loss,
                    "lr": optimizer_stage1.param_groups[0]["lr"],
                })

            if epochs_no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage1}] (patience={patience}).")
                break

    model_stage1.load_state_dict(torch.load(stage1_model_path, map_location=device))
    model_stage1.eval()

    test_loss_sum, test_correct, test_total = 0.0, 0, 0
    all_preds_stage1 = []
    all_labels_stage1 = []
    with torch.no_grad():
        for fft_mag, labels in test_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)
            outputs = model_stage1(fft_mag)
            loss = criterion(outputs, labels)
            bs = labels.size(0)
            test_loss_sum += loss.item() * bs
            _, preds = outputs.max(1)
            test_total += bs
            test_correct += preds.eq(labels).sum().item()
            all_preds_stage1.extend(preds.cpu().numpy())
            all_labels_stage1.extend(labels.cpu().numpy())

    test_loss_stage1 = test_loss_sum / max(test_total, 1)
    test_acc_stage1 = 100.0 * test_correct / max(test_total, 1)
    test_f1_stage1 = f1_score(all_labels_stage1, all_preds_stage1, average='macro')

    print("-" * 50)
    print("Stage 1 Summary:")
    print(f"Test Loss: {test_loss_stage1:.4f} | Test Acc: {test_acc_stage1:.2f}% | Test F1 Macro: {test_f1_stage1:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "stage1_test_acc": test_acc_stage1,
            "stage1_test_f1": test_f1_stage1,
        })

    # STAGE 2
    print("\n" + "=" * 60)
    print("STAGE 2: Pruning weights based on L1 norm")
    print("=" * 60)
    masks = get_l1_masks(model_stage1, pruning_ratio)
    
    # Analyze masks
    c2 = int(masks['block2'].sum().item())
    c3 = int(masks['block3'].sum().item())
    c4 = int(masks['block4'].sum().item())
    
    print(f"Kept channels per block: Block2: {c2}/64, Block3: {c3}/128, Block4: {c4}/128")

    # STAGE 3
    print("\n" + "=" * 60)
    print("STAGE 3: Fine tuning physically shrunk PrunedSeparableConvCNN")
    print("=" * 60)
    model_stage3 = PrunedSeparableConvCNN(
        num_classes=8, num_channels=6,
        block2_channels=c2, block3_channels=c3, block4_channels=c4,
        dropout=dropout
    ).to(device)

    print("\nCopying unpruned weights into Stage 3 physical model...")
    copy_weights_pruned(model_stage1, model_stage3, masks)

    optimizer_stage3 = torch.optim.Adam(model_stage3.parameters(), lr=lr)
    scheduler_stage3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage3, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    stage3_best_val_loss = float('inf')
    stage3_best_epoch = 0
    epochs_no_improve = 0

    print("-" * 50)
    for epoch in range(epochs_stage3):
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        model_stage3.train()
        for fft_mag, labels in train_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)
            outputs = model_stage3(fft_mag)
            loss = criterion(outputs, labels)
            optimizer_stage3.zero_grad()
            loss.backward()
            optimizer_stage3.step()
            train_loss_sum += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total * 100.0

        model_stage3.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for fft_mag, labels in val_dataloader:
                fft_mag, labels = fft_mag.to(device), labels.to(device)
                outputs = model_stage3(fft_mag)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        scheduler_stage3.step(val_loss)

        if val_loss < stage3_best_val_loss - min_delta:
            stage3_best_val_loss = val_loss
            torch.save(model_stage3.state_dict(), stage3_model_path)
            stage3_best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch [{epoch+1}/{epochs_stage3}]: Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}")

        if wandb_run is not None:
            wandb_run.log({
                "stage": 3,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": stage3_best_val_loss,
                "lr": optimizer_stage3.param_groups[0]["lr"],
            })

        if epochs_no_improve >= patience:
            print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage3}] (patience={patience}).")
            break

    model_stage3.load_state_dict(torch.load(stage3_model_path, map_location=device))
    model_stage3.eval()

    test_loss_sum, test_correct, test_total = 0.0, 0, 0
    all_preds_stage3 = []
    all_labels_stage3 = []
    with torch.no_grad():
        for fft_mag, labels in test_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)
            outputs = model_stage3(fft_mag)
            loss = criterion(outputs, labels)
            bs = labels.size(0)
            test_loss_sum += loss.item() * bs
            _, preds = outputs.max(1)
            test_total += bs
            test_correct += preds.eq(labels).sum().item()
            all_preds_stage3.extend(preds.cpu().numpy())
            all_labels_stage3.extend(labels.cpu().numpy())

    test_loss_stage3 = test_loss_sum / max(test_total, 1)
    test_acc_stage3 = 100.0 * test_correct / max(test_total, 1)
    test_f1_stage3 = f1_score(all_labels_stage3, all_preds_stage3, average='macro')

    print("-" * 50)
    print("Stage 3 Summary:")
    print(f"Best Val Loss: {stage3_best_val_loss:.4f} at Epoch {stage3_best_epoch}")
    print(f"Test Loss: {test_loss_stage3:.4f} | Test Acc: {test_acc_stage3:.2f}% | Test F1 Macro: {test_f1_stage3:.4f}")

    params_before = sum(p.numel() for p in model_stage1.parameters() if p.requires_grad)
    params_after = sum(p.numel() for p in model_stage3.parameters() if p.requires_grad)
    pruning_stats = {
        'params_before_pruning': params_before,
        'params_after_pruning': params_after,
        'channels_block2': c2,
        'channels_block3': c3,
        'channels_block4': c4
    }

    if wandb_run is not None:
        log_payload = {
            "stage3_test_acc": test_acc_stage3,
            "stage3_test_f1": test_f1_stage3,
        }
        log_payload.update(pruning_stats)
        wandb_run.log(log_payload)

    return {
        "stage1": {
            "model": "SeparableConvCNN",
            "test_acc": test_acc_stage1,
            "test_f1_macro": test_f1_stage1,
        },
        "stage3": {
            "model": "PrunedSeparableConvCNN",
            "test_acc": test_acc_stage3,
            "test_f1_macro": test_f1_stage3,
            "pruning_stats": pruning_stats,
        },
    }