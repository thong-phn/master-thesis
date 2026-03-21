import torch
import torch.nn as nn
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.fftpack import dct
from scipy.signal import butter, filtfilt


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


def remove_gravity(signal, cutoff=0.3, fs=50, order=3):
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
        acc_data = remove_gravity(acc_data)

        # Step 2: Min-Max normalization per channel to [-1, 1]
        # This standardizes the data to be bounded in [-1, 1] similar to UCI-HAR,
        # preventing large inputs from destabilizing the Gumbel mask gradients.
        min_vals = np.min(acc_data, axis=0)
        max_vals = np.max(acc_data, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        acc_data = 2.0 * (acc_data - min_vals) / range_vals - 1.0

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


# Dataset
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
        Load WEAR data for specific subjects, apply sliding window, then FFT/DCT/no.

        Args:
            root_path: Path to WEAR dataset root (containing train/ and test/ subdirs)
            split: 'train' or 'test'
            subject_ids: List of subject IDs to load (None = all in split)
            preprocessing: 'fft', 'dct', or 'no'
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

    def __getitem__(self, idx):
        signal = self.signals[idx]  # (3, window_size)

        if self.preprocessing == 'dct':
            mag = self._compute_dct(signal)
        elif self.preprocessing == 'no':
            mag = signal
        elif self.preprocessing == 'fft':
            mag = self._compute_fft_magnitude(signal)
        else:
            mag = None

        return torch.FloatTensor(mag), torch.LongTensor([self.labels[idx]])[0]


# Training function
def train_loso_wear(root_path, model_class, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    LOSO training for WEAR dataset.

    Args:
        root_path: path to WEAR dataset root
        model_class: model constructor (e.g. GumbelMaskSeparableConvCNN)
        train_subjects: list of subject IDs for training
        val_subjects: list of subject IDs for validation
        wandb_run: optional wandb run for logging
        **train_kwargs: epochs, lr, batch_size, device, patience, min_delta,
                        sparsity_weight, model_path, preprocessing
    """
    # Hyperparameters
    epochs = train_kwargs.get('epochs', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    sparsity_weight = train_kwargs.get('sparsity_weight', 0.01)
    model_path = Path(train_kwargs.get('model_path', './models/best_wear_model.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessing = train_kwargs.get('preprocessing', 'fft')

    # Create datasets
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

    # Training loop configuration
    freq_bins = train_dataset[0][0].shape[-1]
    
    # Check if model supports tau parameters
    tau_start = train_kwargs.get('tau_start', 5.0)
    tau_end = train_kwargs.get('tau_end', 1.0)
    dropout = train_kwargs.get('dropout', 0.4)
    if model_class.__name__ == 'GumbelMaskSeparableConvCNN':
        model = model_class(num_classes=8, num_channels=6, freq_bins=freq_bins, tau_start=tau_start, tau_end=tau_end, dropout=dropout).to(device)
    else:
        model = model_class(num_classes=8, num_channels=6, freq_bins=freq_bins, dropout=dropout).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    print("-" * 50)
    # Training loop
    for epoch in range(epochs):
        # Tau annealing (GumbelMask)
        if hasattr(model, 'set_tau'):
            model.set_tau(epoch, epochs)

        # Train one epoch
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        model.train()

        for fft_mag, labels in train_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)

            outputs = model(fft_mag)
            loss = criterion(outputs, labels)

            # L1 penalty on mask fraction (GumbelMask)
            if hasattr(model, 'mask_l1') and model.mask_l1 is not None:
                loss = loss + sparsity_weight * model.mask_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total * 100.0

        # Val one epoch
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
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
        scheduler.step(val_loss)

        # Save best model based on val_loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print mask statistics if model has masking
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

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if epochs_no_improve >= patience:
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

    print("-" * 50)
    print(f"Summary:")
    print(f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1 Macro: {test_f1_macro:.4f}")

    # Print final learned mask if available
    if hasattr(model, 'last_mask') and model.last_mask is not None:
        final_mask = model.last_mask.cpu().numpy()
        bins_kept = (final_mask > 0.5).sum()
        total_bins = len(final_mask)
        print(f"\nFinal Mask Statistics:")
        print(f"  Bins kept: {bins_kept}/{total_bins} ({bins_kept/total_bins:.1%})")
        print(f"  All mask values: {final_mask}")
    else:
        final_mask = None

    if wandb_run is not None:
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
