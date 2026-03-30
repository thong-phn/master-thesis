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
    tau_start = train_kwargs.get('tau_start', 20.0)
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


def _load_stage1_weights_to_gumbel_model(gumbel_model, stage1_state_dict):
    """
    Load weights from SeparableConvCNN (stage 1) into GumbelMaskSeparableConvCNN (stage 2).
    
    The Gumbel model has an additional 'bin_logits' parameter that doesn't exist in the
    base SeparableConvCNN model, so we selectively load weights.
    
    Args:
        gumbel_model: GumbelMaskSeparableConvCNN model to load weights into
        stage1_state_dict: State dict from trained SeparableConvCNN model
    """
    gumbel_state_dict = gumbel_model.state_dict()
    
    # Load all weights that exist in both models
    for name, param in stage1_state_dict.items():
        if name in gumbel_state_dict:
            gumbel_state_dict[name] = param
            print(f"  Loaded: {name}")
        else:
            print(f"  Skipped (not in Gumbel model): {name}")
    
    # The bin_logits will be initialized randomly (not loaded from stage 1)
    print(f"  Kept random initialization: bin_logits (Gumbel-specific parameter)")
    
    gumbel_model.load_state_dict(gumbel_state_dict)
    return gumbel_model


def train_loso_wear_two_stage(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    Two-stage LOSO training for WEAR dataset:
    Stage 1: Train base model without Gumbel mask and save best weights
    Stage 2: Load stage 1 weights into corresponding Gumbel model and train with Gumbel mask
    
    Args:
        root_path: path to WEAR dataset root
        train_subjects: list of subject IDs for training
        val_subjects: list of subject IDs for validation
        wandb_run: optional wandb run for logging
        **train_kwargs: epochs, lr, batch_size, device, patience, min_delta,
                        sparsity_weight, model_path, preprocessing, etc.
    """
    from lib.model import (
        SeparableConvCNN,
        GumbelMaskSeparableConvCNN,
    )
    
    # Hyperparameters
    epochs_stage1 = train_kwargs.get('epochs_stage1', 30)
    epochs_stage2 = train_kwargs.get('epochs_stage2', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    sparsity_weight = train_kwargs.get('sparsity_weight', 0.1)
    model_path = Path(train_kwargs.get('model_path', './models/best_wear_model_two_stage.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing = train_kwargs.get('preprocessing', 'fft')
    
    # Stage 1 and 2 specific model paths
    stage1_model_path_arg = train_kwargs.get('stage1_model_path')
    if stage1_model_path_arg is not None:
        stage1_model_path = Path(stage1_model_path_arg).expanduser()
    else:
        stage1_model_path = model_path.parent / f"{model_path.stem}_stage1.pth"
    stage1_model_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_model_path = model_path
    use_pretrained_stage1 = stage1_model_path_arg is not None

    if use_pretrained_stage1 and not stage1_model_path.exists():
        raise FileNotFoundError(
            f"Provided stage1 model path does not exist: {stage1_model_path}"
        )
    
    dropout = train_kwargs.get('dropout', 0.4)
    tau_start = train_kwargs.get('tau_start', 5.0)
    tau_end = train_kwargs.get('tau_end', 1.0)
    stage2_backbone_lr_factor = train_kwargs.get('stage2_backbone_lr_factor', 0.1)
    model_family = train_kwargs.get('model', 'Separable')

    model_registry = {
        'Separable': (SeparableConvCNN, GumbelMaskSeparableConvCNN, 'SeparableConvCNN', 'GumbelMaskSeparableConvCNN'),
        'DeepConvLSTM': (DeepConvLSTM, GumbelMaskDeepConvLSTM, 'DeepConvLSTM', 'GumbelMaskDeepConvLSTM'),
    }
    if model_family not in model_registry:
        raise ValueError(f"Unsupported model family: {model_family}")

    stage1_class, stage2_class, stage1_name, stage2_name = model_registry[model_family]
    
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

    freq_bins = train_dataset[0][0].shape[-1]
    
    # ============================================================
    # STAGE 1: Train SeparableConvCNN (no Gumbel mask)
    # ============================================================
    print("\n" + "="*60)
    if use_pretrained_stage1:
        print(f"STAGE 1: Loading pretrained {stage1_name}")
        print(f"Checkpoint: {stage1_model_path}")
    else:
        print(f"STAGE 1: Training {stage1_name} (without Gumbel mask)")
    print("="*60)
    
    model_stage1 = stage1_class(
        num_classes=8, 
        num_channels=6, 
        freq_bins=freq_bins, 
        dropout=dropout
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
        # Training loop for stage 1
        for epoch in range(epochs_stage1):
            # Train one epoch
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

            # Val one epoch
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
            val_acc = 100. * val_correct / max(val_total, 1)
            scheduler_stage1.step(val_loss)

            # Save best model based on val_loss
            if val_loss < stage1_best_val_loss - min_delta:
                stage1_best_val_loss = val_loss
                torch.save(model_stage1.state_dict(), stage1_model_path)
                stage1_best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f'Epoch [{epoch+1}/{epochs_stage1}]: '
                  f'Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; '
                  f'Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}')

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

    # Load best stage 1 model for evaluation
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
    print(f"Stage 1 Summary:")
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

    # ============================================================
    # STAGE 2: Train GumbelMaskSeparableConvCNN with loaded weights
    # ============================================================
    print("\n" + "="*60)
    print(f"STAGE 2: Training {stage2_name} (with Gumbel mask)")
    print("="*60)
    
    model_stage2 = stage2_class(
        num_classes=8,
        num_channels=6,
        freq_bins=freq_bins,
        dropout=dropout,
        tau_start=tau_start,
        tau_end=tau_end
    ).to(device)
    
    # Load stage 1 weights into stage 2 model
    print("\nLoading Stage 1 weights into Stage 2 model:")
    stage1_state_dict = torch.load(stage1_model_path, map_location=device)
    _load_stage1_weights_to_gumbel_model(model_stage2, stage1_state_dict)
    
    # Use higher LR for Gumbel logits and reduced LR for all other parameters.
    stage2_backbone_lr = lr * stage2_backbone_lr_factor
    gumbel_params = [model_stage2.bin_logits]
    gumbel_param_ids = {id(p) for p in gumbel_params}
    backbone_params = [p for p in model_stage2.parameters() if id(p) not in gumbel_param_ids]

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
        f"gumbel(bin_logits): {lr:.2e}"
    )

    stage2_best_val_loss = float('inf')
    stage2_best_epoch = 0
    epochs_no_improve = 0

    print("-" * 50)
    # Training loop for stage 2
    for epoch in range(epochs_stage2):
        # Tau annealing (GumbelMask)
        if hasattr(model_stage2, 'set_tau'):
            model_stage2.set_tau(epoch, epochs_stage2)

        # Train one epoch
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        model_stage2.train()

        for fft_mag, labels in train_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)

            outputs = model_stage2(fft_mag)
            loss = criterion(outputs, labels)

            # L1 penalty on mask fraction (GumbelMask)
            if hasattr(model_stage2, 'mask_l1') and model_stage2.mask_l1 is not None:
                loss = loss + sparsity_weight * model_stage2.mask_l1

            optimizer_stage2.zero_grad()
            loss.backward()
            optimizer_stage2.step()

            train_loss_sum += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total * 100.0

        # Val one epoch
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
        val_acc = 100. * val_correct / max(val_total, 1)
        scheduler_stage2.step(val_loss)

        # Save best model based on val_loss
        if val_loss < stage2_best_val_loss - min_delta:
            stage2_best_val_loss = val_loss
            torch.save(model_stage2.state_dict(), stage2_model_path)
            stage2_best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print mask statistics if model has masking
        mask_info = ""
        if hasattr(model_stage2, 'mask_l1') and model_stage2.mask_l1 is not None:
            mask_fraction = model_stage2.mask_l1.item()
            tau_info = ""
            if hasattr(model_stage2, 'current_tau'):
                tau_info = f' (tau={model_stage2.current_tau:.2f})'
            mask_info = f'; Mask: {mask_fraction:.2%}' + tau_info

        print(f'Epoch [{epoch+1}/{epochs_stage2}]: '
              f'Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; '
              f'Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}' + mask_info)

        if wandb_run is not None:
            wandb_run.log({
                "stage": 2,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": stage2_best_val_loss,
                "lr_backbone": optimizer_stage2.param_groups[0]["lr"],
                "lr_gumbel": optimizer_stage2.param_groups[1]["lr"],
            })

        if epochs_no_improve >= patience:
            print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage2}] (patience={patience}).")
            break

    # Load best stage 2 model for evaluation
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
    print(f"Stage 2 Summary:")
    print(f"Best Val Loss: {stage2_best_val_loss:.4f} at Epoch {stage2_best_epoch}")
    print(f"Test Loss: {test_loss_stage2:.4f} | Test Acc: {test_acc_stage2:.2f}% | Test F1 Macro: {test_f1_stage2:.4f}")

    # Print final learned mask if available
    if hasattr(model_stage2, 'last_mask') and model_stage2.last_mask is not None:
        final_mask = model_stage2.last_mask.cpu().numpy()
        bins_kept = (final_mask > 0.5).sum()
        total_bins = len(final_mask)
        print(f"\nFinal Mask Statistics:")
        print(f"  Bins kept: {bins_kept}/{total_bins} ({bins_kept/total_bins:.1%})")
        print(f"  All mask values: {final_mask}")
    else:
        final_mask = None

    if wandb_run is not None:
        wandb_run.log({
            "stage2_test_loss": test_loss_stage2,
            "stage2_test_acc": test_acc_stage2,
            "stage2_test_f1": test_f1_stage2,
        })
    
    print("\n" + "="*60)
    print("TWO-STAGE TRAINING COMPLETE")
    print("="*60)
    print(f"Stage 1 ({stage1_name}): Test Acc: {test_acc_stage1:.2f}% | F1: {test_f1_stage1:.4f}")
    print(f"Stage 2 ({stage2_name}): Test Acc: {test_acc_stage2:.2f}% | F1: {test_f1_stage2:.4f}")
    print(f"Improvement: {test_acc_stage2 - test_acc_stage1:.2f}%")

    return {
        "stage1": {
            "model": stage1_name,
            "best_val_loss": stage1_best_val_loss,
            "best_epoch": stage1_best_epoch,
            "test_loss": test_loss_stage1,
            "test_acc": test_acc_stage1,
            "test_f1_macro": test_f1_stage1,
            "model_path": str(stage1_model_path),
            "loaded_from_checkpoint": use_pretrained_stage1,
        },
        "stage2": {
            "model": stage2_name,
            "best_val_loss": stage2_best_val_loss,
            "best_epoch": stage2_best_epoch,
            "test_loss": test_loss_stage2,
            "test_acc": test_acc_stage2,
            "test_f1_macro": test_f1_stage2,
            "model_path": str(stage2_model_path),
            "final_mask": final_mask,
        }
    }


def _load_stage1_weights_to_channel_gumbel_model(channel_model, stage1_state_dict):
    """
    Load stage-1 SeparableConvCNN weights into channel-pruning stage-2 model.

    The stage-2 model adds chan_logits_* parameters, which are intentionally
    left with random initialization.
    """
    channel_state_dict = channel_model.state_dict()

    for name, param in stage1_state_dict.items():
        if name in channel_state_dict:
            channel_state_dict[name] = param
            print(f"  Loaded: {name}")
        else:
            print(f"  Skipped (not in stage-2 channel model): {name}")

    print("  Kept random initialization: chan_logits_2, chan_logits_3, chan_logits_4")
    channel_model.load_state_dict(channel_state_dict)
    return channel_model


def train_loso_wear_two_stage_channel(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    LOSO training for WEAR dataset with channel-pruning stage 2:
    Stage 1: SeparableConvCNN (no mask)
    Stage 2: GumbelChannelPruningCNN (mask channels after conv blocks)
    
    """
    from lib.model import SeparableConvCNN, GumbelChannelPruningCNN

    epochs_stage1 = train_kwargs.get('epochs_stage1', 30)
    epochs_stage2 = train_kwargs.get('epochs_stage2', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    sparsity_weight = train_kwargs.get('sparsity_weight', 0.01)
    model_path = Path(train_kwargs.get('model_path', './models/best_wear_model_two_stage_channel.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing = train_kwargs.get('preprocessing', 'fft')

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

    dropout = train_kwargs.get('dropout', 0.4)
    tau_start = train_kwargs.get('tau_start', 5.0)
    tau_end = train_kwargs.get('tau_end', 1.0)
    stage2_backbone_lr_factor = train_kwargs.get('stage2_backbone_lr_factor', 0.1)

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

    print("\n" + "=" * 60)
    print("STAGE 2: Training GumbelChannelPruningCNN (channel masks after blocks)")
    print("=" * 60)

    model_stage2 = GumbelChannelPruningCNN(
        num_classes=8,
        num_channels=6,
        freq_bins=freq_bins,
        dropout=dropout,
        tau_start=tau_start,
        tau_end=tau_end,
    ).to(device)

    print("\nLoading Stage 1 weights into Stage 2 model:")
    stage1_state_dict = torch.load(stage1_model_path, map_location=device)
    _load_stage1_weights_to_channel_gumbel_model(model_stage2, stage1_state_dict)

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
        print("\nFinal Pruning Statistics:")
        for k, v in pruning_stats.items():
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


# class MaskedWEARDataset(Dataset):
#     """Wrap WEAR_Dataset and apply a fixed per-bin mask to each sample."""

#     def __init__(self, base_dataset, bin_mask):
#         self.base_dataset = base_dataset
#         self.bin_mask = torch.as_tensor(bin_mask, dtype=torch.float32)

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         x, y = self.base_dataset[idx]
#         x = x * self.bin_mask.unsqueeze(0)
#         return x, y


class SlicedWEARDataset(Dataset):
    """Wrap WEAR_Dataset and keep only selected frequency bins (hard slicing)."""

    def __init__(self, base_dataset, keep_indices):
        self.base_dataset = base_dataset
        self.keep_indices = torch.as_tensor(keep_indices, dtype=torch.long)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        x = x.index_select(-1, self.keep_indices)
        return x, y


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
    model.eval()
    loss_sum, total, correct = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            bs = y.size(0)
            loss_sum += loss.item() * bs
            _, pred = out.max(1)
            total += bs
            correct += pred.eq(y).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0
    return avg_loss, acc, f1


def _get_hard_bin_mask_from_model(model):
    with torch.no_grad():
        hard = torch.softmax(model.bin_logits, dim=-1).argmax(dim=-1).float()
    return hard

def train_loso_wear_two_stage_input_pruning(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    Two-stage LOSO training for WEAR dataset:
    Stage 1: Train SeparableConvCNN on full input.
    Stage 2: Load stage-1 weights into GumbelMaskSeparableConvCNN and learn input-bin pruning.
    """
    from lib.model import SeparableConvCNN, GumbelMaskSeparableConvCNN

    epochs_stage1 = train_kwargs.get('epochs_stage1', 60)
    epochs_stage2 = train_kwargs.get('epochs_stage2', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    preprocessing = train_kwargs.get('preprocessing', 'fft')
    dropout = train_kwargs.get('dropout', 0.4)
    tau_start = train_kwargs.get('tau_start', 10.0)
    tau_end = train_kwargs.get('tau_end', 1.0)
    sparsity_weight_bin = train_kwargs.get('sparsity_weight_bin', train_kwargs.get('sparsity_weight', 0.01))
    stage2_backbone_lr_factor = train_kwargs.get('stage2_backbone_lr_factor', 0.1)

    model_path = Path(train_kwargs.get('model_path', './models/best_wear_model_three_stage.pth'))
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

    use_pretrained_stage1 = stage1_model_path_arg is not None
    use_pretrained_stage2 = stage2_model_path_arg is not None

    if use_pretrained_stage1 and not stage1_model_path.exists():
        raise FileNotFoundError(f"Provided stage1 model path does not exist: {stage1_model_path}")
    if use_pretrained_stage2 and not stage2_model_path.exists():
        raise FileNotFoundError(f"Provided stage2 model path does not exist: {stage2_model_path}")

    train_dataset = WEAR_Dataset(root_path, split='train', subject_ids=train_subjects, preprocessing=preprocessing)
    val_dataset = WEAR_Dataset(root_path, split='train', subject_ids=val_subjects, preprocessing=preprocessing)
    test_dataset = WEAR_Dataset(root_path, split='test', subject_ids=None, preprocessing=preprocessing)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    freq_bins = train_dataset[0][0].shape[-1]
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # STAGE 1
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage1:
        print("STAGE 1: Loading pretrained SeparableConvCNN")
        print(f"Checkpoint: {stage1_model_path}")
    else:
        print("STAGE 1: Training SeparableConvCNN")
    print("=" * 60)

    model_stage1 = SeparableConvCNN(num_classes=8, num_channels=6, freq_bins=freq_bins, dropout=dropout).to(device)
    stage1_best_val_loss = None
    stage1_best_epoch = None

    if not use_pretrained_stage1:
        optimizer = torch.optim.Adam(model_stage1.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
        stage1_best_val_loss = float('inf')
        stage1_best_epoch = 0
        no_improve = 0

        for epoch in range(epochs_stage1):
            model_stage1.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model_stage1(x)
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

            val_loss, val_acc, _ = _evaluate_classifier(model_stage1, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < stage1_best_val_loss - min_delta:
                stage1_best_val_loss = val_loss
                stage1_best_epoch = epoch + 1
                torch.save(model_stage1.state_dict(), stage1_model_path)
                no_improve = 0
            else:
                no_improve += 1

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
                    "lr": optimizer.param_groups[0]["lr"],
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage1}] (patience={patience}).")
                break

    model_stage1.load_state_dict(torch.load(stage1_model_path, map_location=device))
    stage1_test_loss, stage1_test_acc, stage1_test_f1 = _evaluate_classifier(model_stage1, test_loader, criterion, device)
    print("-" * 50)
    print("Stage 1 Summary:")
    if stage1_best_val_loss is not None:
        print(f"Best Val Loss: {stage1_best_val_loss:.4f} at Epoch {stage1_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage1 checkpoint)")
    print(f"Test Loss: {stage1_test_loss:.4f} | Test Acc: {stage1_test_acc:.2f}% | Test F1 Macro: {stage1_test_f1:.4f}")

    # ============================================================
    # STAGE 2
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage2:
        print("STAGE 2: Loading pretrained GumbelMaskSeparableConvCNN (input-bin pruning)")
        print(f"Checkpoint: {stage2_model_path}")
    else:
        print("STAGE 2: Training GumbelMaskSeparableConvCNN (input-bin pruning)")
    print("=" * 60)

    model_stage2 = GumbelMaskSeparableConvCNN(
        num_classes=8,
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
        _load_stage1_weights_to_gumbel_model(model_stage2, torch.load(stage1_model_path, map_location=device))

        stage2_backbone_lr = lr * stage2_backbone_lr_factor
        gumbel_params = [model_stage2.bin_logits]
        gumbel_ids = {id(p) for p in gumbel_params}
        backbone_params = [p for p in model_stage2.parameters() if id(p) not in gumbel_ids]

        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": stage2_backbone_lr},
                {"params": gumbel_params, "lr": lr},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

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
            print(
                f"Epoch [{epoch+1}/{epochs_stage2}]: "
                f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}" + mask_info
            )

            if wandb_run is not None:
                wandb_run.log({
                    "stage": 2,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": stage2_best_val_loss,
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    "lr_gumbel_bin": optimizer.param_groups[1]["lr"],
                    "mask_l1": model_stage2.mask_l1.item() if model_stage2.mask_l1 is not None else None,
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage2}] (patience={patience}).")
                break

    if not use_pretrained_stage2:
        model_stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
    stage2_test_loss, stage2_test_acc, stage2_test_f1 = _evaluate_classifier(model_stage2, test_loader, criterion, device)
    hard_bin_mask = _get_hard_bin_mask_from_model(model_stage2).detach().cpu()
    bin_keep_ratio = hard_bin_mask.mean().item()

    print("-" * 50)
    print("Stage 2 Summary:")
    if stage2_best_val_loss is not None:
        print(f"Best Val Loss: {stage2_best_val_loss:.4f} at Epoch {stage2_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage2 checkpoint)")
    print(f"Test Loss: {stage2_test_loss:.4f} | Test Acc: {stage2_test_acc:.2f}% | Test F1 Macro: {stage2_test_f1:.4f}")
    print(f"Hard input bins kept: {(hard_bin_mask > 0.5).sum().item()}/{hard_bin_mask.numel()} ({bin_keep_ratio:.1%})")


    if wandb_run is not None:
        wandb_run.log({
            "stage1_test_loss": stage1_test_loss,
            "stage1_test_acc": stage1_test_acc,
            "stage1_test_f1": stage1_test_f1,
            "stage2_test_loss": stage2_test_loss,
            "stage2_test_acc": stage2_test_acc,
            "stage2_test_f1": stage2_test_f1,
            "bin_keep_ratio": bin_keep_ratio,
        })

    print("\n" + "=" * 60)
    print("TWO-STAGE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Stage 1 (SeparableConvCNN): Test Acc: {stage1_test_acc:.2f}% | F1: {stage1_test_f1:.4f}")
    print(f"Stage 2 (Input Bin Pruning): Test Acc: {stage2_test_acc:.2f}% | F1: {stage2_test_f1:.4f}")

    return {
        "stage1": {
            "model": "SeparableConvCNN",
            "best_val_loss": stage1_best_val_loss,
            "best_epoch": stage1_best_epoch,
            "test_loss": stage1_test_loss,
            "test_acc": stage1_test_acc,
            "test_f1_macro": stage1_test_f1,
            "model_path": str(stage1_model_path),
            "loaded_from_checkpoint": use_pretrained_stage1,
        },
        "stage2": {
            "model": "GumbelMaskSeparableConvCNN",
            "best_val_loss": stage2_best_val_loss,
            "best_epoch": stage2_best_epoch,
            "test_loss": stage2_test_loss,
            "test_acc": stage2_test_acc,
            "test_f1_macro": stage2_test_f1,
            "model_path": str(stage2_model_path),
            "hard_bin_mask": hard_bin_mask.numpy(),
            "bin_keep_ratio": bin_keep_ratio,
            "loaded_from_checkpoint": use_pretrained_stage2,
        },
    }

def train_loso_wear_three_stage(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    Three-stage LOSO training for WEAR dataset:
    Stage 1: Train SeparableConvCNN on full input.
    Stage 2: Load stage-1 weights into GumbelMaskSeparableConvCNN and learn input-bin pruning.
    Stage 3: Apply the learned hard bin mask to data and retrain SeparableConvCNN.
    """
    from lib.model import SeparableConvCNN, GumbelMaskSeparableConvCNN

    epochs_stage1 = train_kwargs.get('epochs_stage1', 60)
    epochs_stage2 = train_kwargs.get('epochs_stage2', 60)
    epochs_stage3 = train_kwargs.get('epochs_stage3', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    preprocessing = train_kwargs.get('preprocessing', 'fft')
    dropout = train_kwargs.get('dropout', 0.4)
    tau_start = train_kwargs.get('tau_start', 10.0)
    tau_end = train_kwargs.get('tau_end', 1.0)
    sparsity_weight_bin = train_kwargs.get('sparsity_weight_bin', train_kwargs.get('sparsity_weight', 0.01))
    stage2_backbone_lr_factor = train_kwargs.get('stage2_backbone_lr_factor', 0.1)

    model_path = Path(train_kwargs.get('model_path', './models/best_wear_model_three_stage.pth'))
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

    use_pretrained_stage1 = stage1_model_path_arg is not None
    use_pretrained_stage2 = stage2_model_path_arg is not None
    use_pretrained_stage3 = stage3_model_path_arg is not None

    if use_pretrained_stage1 and not stage1_model_path.exists():
        raise FileNotFoundError(f"Provided stage1 model path does not exist: {stage1_model_path}")
    if use_pretrained_stage2 and not stage2_model_path.exists():
        raise FileNotFoundError(f"Provided stage2 model path does not exist: {stage2_model_path}")
    if use_pretrained_stage3 and not stage3_model_path.exists():
        raise FileNotFoundError(f"Provided stage3 model path does not exist: {stage3_model_path}")

    train_dataset = WEAR_Dataset(root_path, split='train', subject_ids=train_subjects, preprocessing=preprocessing)
    val_dataset = WEAR_Dataset(root_path, split='train', subject_ids=val_subjects, preprocessing=preprocessing)
    test_dataset = WEAR_Dataset(root_path, split='test', subject_ids=None, preprocessing=preprocessing)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    freq_bins = train_dataset[0][0].shape[-1]
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # STAGE 1
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage1:
        print("STAGE 1: Loading pretrained SeparableConvCNN")
        print(f"Checkpoint: {stage1_model_path}")
    else:
        print("STAGE 1: Training SeparableConvCNN")
    print("=" * 60)

    model_stage1 = SeparableConvCNN(num_classes=8, num_channels=6, freq_bins=freq_bins, dropout=dropout).to(device)
    stage1_best_val_loss = None
    stage1_best_epoch = None

    if not use_pretrained_stage1:
        optimizer = torch.optim.Adam(model_stage1.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
        stage1_best_val_loss = float('inf')
        stage1_best_epoch = 0
        no_improve = 0

        for epoch in range(epochs_stage1):
            model_stage1.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model_stage1(x)
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

            val_loss, val_acc, _ = _evaluate_classifier(model_stage1, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < stage1_best_val_loss - min_delta:
                stage1_best_val_loss = val_loss
                stage1_best_epoch = epoch + 1
                torch.save(model_stage1.state_dict(), stage1_model_path)
                no_improve = 0
            else:
                no_improve += 1

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
                    "lr": optimizer.param_groups[0]["lr"],
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage1}] (patience={patience}).")
                break

    model_stage1.load_state_dict(torch.load(stage1_model_path, map_location=device))
    stage1_test_loss, stage1_test_acc, stage1_test_f1 = _evaluate_classifier(model_stage1, test_loader, criterion, device)
    print("-" * 50)
    print("Stage 1 Summary:")
    if stage1_best_val_loss is not None:
        print(f"Best Val Loss: {stage1_best_val_loss:.4f} at Epoch {stage1_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage1 checkpoint)")
    print(f"Test Loss: {stage1_test_loss:.4f} | Test Acc: {stage1_test_acc:.2f}% | Test F1 Macro: {stage1_test_f1:.4f}")

    # ============================================================
    # STAGE 2
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage2:
        print("STAGE 2: Loading pretrained GumbelMaskSeparableConvCNN (input-bin pruning)")
        print(f"Checkpoint: {stage2_model_path}")
    else:
        print("STAGE 2: Training GumbelMaskSeparableConvCNN (input-bin pruning)")
    print("=" * 60)

    model_stage2 = GumbelMaskSeparableConvCNN(
        num_classes=8,
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
        _load_stage1_weights_to_gumbel_model(model_stage2, torch.load(stage1_model_path, map_location=device))

        stage2_backbone_lr = lr * stage2_backbone_lr_factor
        gumbel_params = [model_stage2.bin_logits]
        gumbel_ids = {id(p) for p in gumbel_params}
        backbone_params = [p for p in model_stage2.parameters() if id(p) not in gumbel_ids]

        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": stage2_backbone_lr},
                {"params": gumbel_params, "lr": lr},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

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
            print(
                f"Epoch [{epoch+1}/{epochs_stage2}]: "
                f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}" + mask_info
            )

            if wandb_run is not None:
                wandb_run.log({
                    "stage": 2,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": stage2_best_val_loss,
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    "lr_gumbel_bin": optimizer.param_groups[1]["lr"],
                    "mask_l1": model_stage2.mask_l1.item() if model_stage2.mask_l1 is not None else None,
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage2}] (patience={patience}).")
                break

    if not use_pretrained_stage2:
        model_stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
    stage2_test_loss, stage2_test_acc, stage2_test_f1 = _evaluate_classifier(model_stage2, test_loader, criterion, device)
    hard_bin_mask = _get_hard_bin_mask_from_model(model_stage2).detach().cpu()
    bin_keep_ratio = hard_bin_mask.mean().item()

    print("-" * 50)
    print("Stage 2 Summary:")
    if stage2_best_val_loss is not None:
        print(f"Best Val Loss: {stage2_best_val_loss:.4f} at Epoch {stage2_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage2 checkpoint)")
    print(f"Test Loss: {stage2_test_loss:.4f} | Test Acc: {stage2_test_acc:.2f}% | Test F1 Macro: {stage2_test_f1:.4f}")
    print(f"Hard input bins kept: {(hard_bin_mask > 0.5).sum().item()}/{hard_bin_mask.numel()} ({bin_keep_ratio:.1%})")

    # ============================================================
    # STAGE 3
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage3:
        print("STAGE 3: Loading pretrained SeparableConvCNN on pruned input")
        print(f"Checkpoint: {stage3_model_path}")
    else:
        print("STAGE 3: Retraining SeparableConvCNN on pruned input")
    print("=" * 60)

    keep_indices = torch.where(hard_bin_mask > 0.5)[0]
    if keep_indices.numel() == 0:
        raise ValueError("Stage 2 pruned all bins; cannot run Stage 3 with empty input.")

    pruned_train_dataset = SlicedWEARDataset(train_dataset, keep_indices)
    pruned_val_dataset = SlicedWEARDataset(val_dataset, keep_indices)
    pruned_test_dataset = SlicedWEARDataset(test_dataset, keep_indices)

    pruned_train_loader = DataLoader(pruned_train_dataset, batch_size=batch_size, shuffle=True)
    pruned_val_loader = DataLoader(pruned_val_dataset, batch_size=batch_size, shuffle=False)
    pruned_test_loader = DataLoader(pruned_test_dataset, batch_size=batch_size, shuffle=False)

    sample_x, _ = next(iter(train_loader))
    sample_x_pruned, _ = next(iter(pruned_train_loader))
    kept_bins = int(keep_indices.numel())
    total_bins = int(hard_bin_mask.numel())
    print(
        f"Stage 3 input check: original batch shape={tuple(sample_x.shape)}, "
        f"pruned batch shape={tuple(sample_x_pruned.shape)}, "
        f"kept bins={kept_bins}/{total_bins}"
    )

    pruned_freq_bins = kept_bins
    model_stage3 = SeparableConvCNN(num_classes=8, num_channels=6, freq_bins=pruned_freq_bins, dropout=dropout).to(device)
    stage3_best_val_loss = None
    stage3_best_epoch = None

    if use_pretrained_stage3:
        model_stage3.load_state_dict(torch.load(stage3_model_path, map_location=device))
    else:
        stage2_state = torch.load(stage2_model_path, map_location=device) # Load weight from previous stage
        _load_matching_weights(model_stage3, stage2_state)

        optimizer = torch.optim.Adam(model_stage3.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
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

            print(
                f"Epoch [{epoch+1}/{epochs_stage3}]: "
                f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "stage": 3,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": stage3_best_val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage3}] (patience={patience}).")
                break

        model_stage3.load_state_dict(torch.load(stage3_model_path, map_location=device))
    stage3_test_loss, stage3_test_acc, stage3_test_f1 = _evaluate_classifier(model_stage3, pruned_test_loader, criterion, device)

    print("-" * 50)
    print("Stage 3 Summary:")
    if stage3_best_val_loss is not None:
        print(f"Best Val Loss: {stage3_best_val_loss:.4f} at Epoch {stage3_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage3 checkpoint)")
    print(f"Test Loss: {stage3_test_loss:.4f} | Test Acc: {stage3_test_acc:.2f}% | Test F1 Macro: {stage3_test_f1:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "stage1_test_loss": stage1_test_loss,
            "stage1_test_acc": stage1_test_acc,
            "stage1_test_f1": stage1_test_f1,
            "stage2_test_loss": stage2_test_loss,
            "stage2_test_acc": stage2_test_acc,
            "stage2_test_f1": stage2_test_f1,
            "stage3_test_loss": stage3_test_loss,
            "stage3_test_acc": stage3_test_acc,
            "stage3_test_f1": stage3_test_f1,
            "bin_keep_ratio": bin_keep_ratio,
        })

    print("\n" + "=" * 60)
    print("THREE-STAGE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Stage 1 (SeparableConvCNN): Test Acc: {stage1_test_acc:.2f}% | F1: {stage1_test_f1:.4f}")
    print(f"Stage 2 (Input Bin Pruning): Test Acc: {stage2_test_acc:.2f}% | F1: {stage2_test_f1:.4f}")
    print(f"Stage 3 (Pruned Input Retrain): Test Acc: {stage3_test_acc:.2f}% | F1: {stage3_test_f1:.4f}")

    return {
        "stage1": {
            "model": "SeparableConvCNN",
            "best_val_loss": stage1_best_val_loss,
            "best_epoch": stage1_best_epoch,
            "test_loss": stage1_test_loss,
            "test_acc": stage1_test_acc,
            "test_f1_macro": stage1_test_f1,
            "model_path": str(stage1_model_path),
            "loaded_from_checkpoint": use_pretrained_stage1,
        },
        "stage2": {
            "model": "GumbelMaskSeparableConvCNN",
            "best_val_loss": stage2_best_val_loss,
            "best_epoch": stage2_best_epoch,
            "test_loss": stage2_test_loss,
            "test_acc": stage2_test_acc,
            "test_f1_macro": stage2_test_f1,
            "model_path": str(stage2_model_path),
            "hard_bin_mask": hard_bin_mask.numpy(),
            "bin_keep_ratio": bin_keep_ratio,
            "loaded_from_checkpoint": use_pretrained_stage2,
        },
        "stage3": {
            "model": "SeparableConvCNN",
            "best_val_loss": stage3_best_val_loss,
            "best_epoch": stage3_best_epoch,
            "test_loss": stage3_test_loss,
            "test_acc": stage3_test_acc,
            "test_f1_macro": stage3_test_f1,
            "model_path": str(stage3_model_path),
            "loaded_from_checkpoint": use_pretrained_stage3,
        },
    }


def train_loso_wear_multi_stage(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    Four-stage LOSO training for WEAR dataset:
    Stage 1: Train SeparableConvCNN on full input.
    Stage 2: Load stage-1 weights into GumbelMaskSeparableConvCNN and learn input-bin pruning.
    Stage 3: Apply the learned hard bin mask to data and retrain SeparableConvCNN.
    Stage 4: Load stage-3 weights into GumbelChannelPruningCNN and train channel pruning on pruned input.
    """
    from lib.model import SeparableConvCNN, GumbelMaskSeparableConvCNN, GumbelChannelPruningCNN

    epochs_stage1 = train_kwargs.get('epochs_stage1', 60)
    epochs_stage2 = train_kwargs.get('epochs_stage2', 60)
    epochs_stage3 = train_kwargs.get('epochs_stage3', 60)
    epochs_stage4 = train_kwargs.get('epochs_stage4', 60)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    preprocessing = train_kwargs.get('preprocessing', 'fft')
    dropout = train_kwargs.get('dropout', 0.4)
    tau_start = train_kwargs.get('tau_start', 10.0)
    tau_end = train_kwargs.get('tau_end', 1.0)

    sparsity_weight_bin = train_kwargs.get('sparsity_weight_bin', train_kwargs.get('sparsity_weight', 0.01))
    sparsity_weight_channel = train_kwargs.get('sparsity_weight_channel', train_kwargs.get('sparsity_weight', 0.01))

    stage2_backbone_lr_factor = train_kwargs.get('stage2_backbone_lr_factor', 0.1)
    stage4_backbone_lr_factor = train_kwargs.get('stage4_backbone_lr_factor', 0.1)

    model_path = Path(train_kwargs.get('model_path', './models/best_wear_model_multi_stage.pth'))
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

    use_pretrained_stage1 = stage1_model_path_arg is not None
    use_pretrained_stage2 = stage2_model_path_arg is not None
    use_pretrained_stage3 = stage3_model_path_arg is not None
    use_pretrained_stage4 = stage4_model_path_arg is not None

    if use_pretrained_stage1 and not stage1_model_path.exists():
        raise FileNotFoundError(f"Provided stage1 model path does not exist: {stage1_model_path}")
    if use_pretrained_stage2 and not stage2_model_path.exists():
        raise FileNotFoundError(f"Provided stage2 model path does not exist: {stage2_model_path}")
    if use_pretrained_stage3 and not stage3_model_path.exists():
        raise FileNotFoundError(f"Provided stage3 model path does not exist: {stage3_model_path}")
    if use_pretrained_stage4 and not stage4_model_path.exists():
        raise FileNotFoundError(f"Provided stage4 model path does not exist: {stage4_model_path}")

    train_dataset = WEAR_Dataset(root_path, split='train', subject_ids=train_subjects, preprocessing=preprocessing)
    val_dataset = WEAR_Dataset(root_path, split='train', subject_ids=val_subjects, preprocessing=preprocessing)
    test_dataset = WEAR_Dataset(root_path, split='test', subject_ids=None, preprocessing=preprocessing)

    if len(train_dataset) == 0:
        raise ValueError('Empty training dataset for selected subjects.')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    freq_bins = train_dataset[0][0].shape[-1]
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # STAGE 1
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage1:
        print("STAGE 1: Loading pretrained SeparableConvCNN")
        print(f"Checkpoint: {stage1_model_path}")
    else:
        print("STAGE 1: Training SeparableConvCNN")
    print("=" * 60)

    model_stage1 = SeparableConvCNN(num_classes=8, num_channels=6, freq_bins=freq_bins, dropout=dropout).to(device)
    stage1_best_val_loss = None
    stage1_best_epoch = None

    if not use_pretrained_stage1:
        optimizer = torch.optim.Adam(model_stage1.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
        stage1_best_val_loss = float('inf')
        stage1_best_epoch = 0
        no_improve = 0

        for epoch in range(epochs_stage1):
            model_stage1.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model_stage1(x)
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

            val_loss, val_acc, _ = _evaluate_classifier(model_stage1, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < stage1_best_val_loss - min_delta:
                stage1_best_val_loss = val_loss
                stage1_best_epoch = epoch + 1
                torch.save(model_stage1.state_dict(), stage1_model_path)
                no_improve = 0
            else:
                no_improve += 1

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
                    "lr": optimizer.param_groups[0]["lr"],
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage1}] (patience={patience}).")
                break

    model_stage1.load_state_dict(torch.load(stage1_model_path, map_location=device))
    stage1_test_loss, stage1_test_acc, stage1_test_f1 = _evaluate_classifier(model_stage1, test_loader, criterion, device)
    print("-" * 50)
    print("Stage 1 Summary:")
    if stage1_best_val_loss is not None:
        print(f"Best Val Loss: {stage1_best_val_loss:.4f} at Epoch {stage1_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage1 checkpoint)")
    print(f"Test Loss: {stage1_test_loss:.4f} | Test Acc: {stage1_test_acc:.2f}% | Test F1 Macro: {stage1_test_f1:.4f}")

    # ============================================================
    # STAGE 2
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage2:
        print("STAGE 2: Loading pretrained GumbelMaskSeparableConvCNN (input-bin pruning)")
        print(f"Checkpoint: {stage2_model_path}")
    else:
        print("STAGE 2: Training GumbelMaskSeparableConvCNN (input-bin pruning)")
    print("=" * 60)

    model_stage2 = GumbelMaskSeparableConvCNN(
        num_classes=8,
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
        _load_stage1_weights_to_gumbel_model(model_stage2, torch.load(stage1_model_path, map_location=device))

        stage2_backbone_lr = lr * stage2_backbone_lr_factor
        gumbel_params = [model_stage2.bin_logits]
        gumbel_ids = {id(p) for p in gumbel_params}
        backbone_params = [p for p in model_stage2.parameters() if id(p) not in gumbel_ids]

        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": stage2_backbone_lr},
                {"params": gumbel_params, "lr": lr},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

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
            print(
                f"Epoch [{epoch+1}/{epochs_stage2}]: "
                f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}" + mask_info
            )

            if wandb_run is not None:
                wandb_run.log({
                    "stage": 2,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": stage2_best_val_loss,
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    "lr_gumbel_bin": optimizer.param_groups[1]["lr"],
                    "mask_l1": model_stage2.mask_l1.item() if model_stage2.mask_l1 is not None else None,
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage2}] (patience={patience}).")
                break

    if not use_pretrained_stage2:
        model_stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
    stage2_test_loss, stage2_test_acc, stage2_test_f1 = _evaluate_classifier(model_stage2, test_loader, criterion, device)
    hard_bin_mask = _get_hard_bin_mask_from_model(model_stage2).detach().cpu()
    bin_keep_ratio = hard_bin_mask.mean().item()

    print("-" * 50)
    print("Stage 2 Summary:")
    if stage2_best_val_loss is not None:
        print(f"Best Val Loss: {stage2_best_val_loss:.4f} at Epoch {stage2_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage2 checkpoint)")
    print(f"Test Loss: {stage2_test_loss:.4f} | Test Acc: {stage2_test_acc:.2f}% | Test F1 Macro: {stage2_test_f1:.4f}")
    print(f"Hard input bins kept: {(hard_bin_mask > 0.5).sum().item()}/{hard_bin_mask.numel()} ({bin_keep_ratio:.1%})")

    # ============================================================
    # STAGE 3
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage3:
        print("STAGE 3: Loading pretrained SeparableConvCNN on pruned input")
        print(f"Checkpoint: {stage3_model_path}")
    else:
        print("STAGE 3: Retraining SeparableConvCNN on pruned input")
    print("=" * 60)

    keep_indices = torch.where(hard_bin_mask > 0.5)[0]
    if keep_indices.numel() == 0:
        raise ValueError("Stage 2 pruned all bins; cannot run Stage 3 with empty input.")

    pruned_train_dataset = SlicedWEARDataset(train_dataset, keep_indices)
    pruned_val_dataset = SlicedWEARDataset(val_dataset, keep_indices)
    pruned_test_dataset = SlicedWEARDataset(test_dataset, keep_indices)

    pruned_train_loader = DataLoader(pruned_train_dataset, batch_size=batch_size, shuffle=True)
    pruned_val_loader = DataLoader(pruned_val_dataset, batch_size=batch_size, shuffle=False)
    pruned_test_loader = DataLoader(pruned_test_dataset, batch_size=batch_size, shuffle=False)

    pruned_freq_bins = int(keep_indices.numel())
    model_stage3 = SeparableConvCNN(num_classes=8, num_channels=6, freq_bins=pruned_freq_bins, dropout=dropout).to(device)
    stage3_best_val_loss = None
    stage3_best_epoch = None

    if use_pretrained_stage3:
        model_stage3.load_state_dict(torch.load(stage3_model_path, map_location=device))
    else:
        stage2_state = torch.load(stage2_model_path, map_location=device)
        _load_matching_weights(model_stage3, stage2_state)

        optimizer = torch.optim.Adam(model_stage3.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
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

            print(
                f"Epoch [{epoch+1}/{epochs_stage3}]: "
                f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "stage": 3,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": stage3_best_val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage3}] (patience={patience}).")
                break

        model_stage3.load_state_dict(torch.load(stage3_model_path, map_location=device))
    stage3_test_loss, stage3_test_acc, stage3_test_f1 = _evaluate_classifier(model_stage3, pruned_test_loader, criterion, device)

    print("-" * 50)
    print("Stage 3 Summary:")
    if stage3_best_val_loss is not None:
        print(f"Best Val Loss: {stage3_best_val_loss:.4f} at Epoch {stage3_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage3 checkpoint)")
    print(f"Test Loss: {stage3_test_loss:.4f} | Test Acc: {stage3_test_acc:.2f}% | Test F1 Macro: {stage3_test_f1:.4f}")

    # ============================================================
    # STAGE 4
    # ============================================================
    print("\n" + "=" * 60)
    if use_pretrained_stage4:
        print("STAGE 4: Loading pretrained GumbelChannelPruningCNN on pruned input")
        print(f"Checkpoint: {stage4_model_path}")
    else:
        print("STAGE 4: Training GumbelChannelPruningCNN on pruned input")
    print("=" * 60)

    model_stage4 = GumbelChannelPruningCNN(
        num_classes=8,
        num_channels=6,
        freq_bins=pruned_freq_bins,
        dropout=dropout,
        tau_start=tau_start,
        tau_end=tau_end,
    ).to(device)

    stage4_best_val_loss = None
    stage4_best_epoch = None

    if use_pretrained_stage4:
        model_stage4.load_state_dict(torch.load(stage4_model_path, map_location=device))
    else:
        print("\nLoading Stage 3 weights into Stage 4 model:")
        _load_stage1_weights_to_channel_gumbel_model(model_stage4, torch.load(stage3_model_path, map_location=device))

        stage4_backbone_lr = lr * stage4_backbone_lr_factor
        named_params = list(model_stage4.named_parameters())
        gumbel_params = [p for n, p in named_params if n.startswith('chan_logits_')]
        gumbel_ids = {id(p) for p in gumbel_params}
        backbone_params = [p for _, p in named_params if id(p) not in gumbel_ids]

        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": stage4_backbone_lr},
                {"params": gumbel_params, "lr": lr},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

        stage4_best_val_loss = float('inf')
        stage4_best_epoch = 0
        no_improve = 0

        for epoch in range(epochs_stage4):
            model_stage4.train()
            if hasattr(model_stage4, 'set_tau'):
                model_stage4.set_tau(epoch, epochs_stage4)

            train_loss_sum, train_correct, train_total = 0.0, 0, 0
            for x, y in pruned_train_loader:
                x, y = x.to(device), y.to(device)
                out = model_stage4(x)
                loss = criterion(out, y)
                loss = loss + sparsity_weight_channel * model_stage4.get_sparsity_loss()

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
            val_loss, val_acc, _ = _evaluate_classifier(model_stage4, pruned_val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < stage4_best_val_loss - min_delta:
                stage4_best_val_loss = val_loss
                stage4_best_epoch = epoch + 1
                torch.save(model_stage4.state_dict(), stage4_model_path)
                no_improve = 0
            else:
                no_improve += 1

            mask_info = f"; Soft On-Prob: {model_stage4.get_sparsity_loss().item():.2%}"
            print(
                f"Epoch [{epoch+1}/{epochs_stage4}]: "
                f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}" + mask_info
            )

            if wandb_run is not None:
                log_payload = {
                    "stage": 4,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": stage4_best_val_loss,
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    "lr_gumbel_channel": optimizer.param_groups[1]["lr"],
                }
                if hasattr(model_stage4, 'get_pruning_stats'):
                    log_payload.update(model_stage4.get_pruning_stats())
                wandb_run.log(log_payload)

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage4}] (patience={patience}).")
                break

        model_stage4.load_state_dict(torch.load(stage4_model_path, map_location=device))
    stage4_test_loss, stage4_test_acc, stage4_test_f1 = _evaluate_classifier(model_stage4, pruned_test_loader, criterion, device)
    hard_channel_masks = model_stage4.get_hard_masks()
    final_channel_mask = {
        "block2": hard_channel_masks["block2"].detach().cpu().numpy(),
        "block3": hard_channel_masks["block3"].detach().cpu().numpy(),
        "block4": hard_channel_masks["block4"].detach().cpu().numpy(),
    }
    pruning_stats = model_stage4.get_pruning_stats()

    print("-" * 50)
    print("Stage 4 Summary:")
    if stage4_best_val_loss is not None:
        print(f"Best Val Loss: {stage4_best_val_loss:.4f} at Epoch {stage4_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage4 checkpoint)")
    print(f"Test Loss: {stage4_test_loss:.4f} | Test Acc: {stage4_test_acc:.2f}% | Test F1 Macro: {stage4_test_f1:.4f}")
    print("Final Channel Pruning Statistics:")
    for k, v in pruning_stats.items():
        print(f"  {k}: {v:.2f}")

    if wandb_run is not None:
        wandb_run.log({
            "stage1_test_loss": stage1_test_loss,
            "stage1_test_acc": stage1_test_acc,
            "stage1_test_f1": stage1_test_f1,
            "stage2_test_loss": stage2_test_loss,
            "stage2_test_acc": stage2_test_acc,
            "stage2_test_f1": stage2_test_f1,
            "stage3_test_loss": stage3_test_loss,
            "stage3_test_acc": stage3_test_acc,
            "stage3_test_f1": stage3_test_f1,
            "stage4_test_loss": stage4_test_loss,
            "stage4_test_acc": stage4_test_acc,
            "stage4_test_f1": stage4_test_f1,
            "bin_keep_ratio": bin_keep_ratio,
            **pruning_stats,
        })

    print("\n" + "=" * 60)
    print("FOUR-STAGE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Stage 1 (SeparableConvCNN): Test Acc: {stage1_test_acc:.2f}% | F1: {stage1_test_f1:.4f}")
    print(f"Stage 2 (Input Bin Pruning): Test Acc: {stage2_test_acc:.2f}% | F1: {stage2_test_f1:.4f}")
    print(f"Stage 3 (Pruned Input Retrain): Test Acc: {stage3_test_acc:.2f}% | F1: {stage3_test_f1:.4f}")
    print(f"Stage 4 (Channel Pruning on Pruned Input): Test Acc: {stage4_test_acc:.2f}% | F1: {stage4_test_f1:.4f}")

    return {
        "stage1": {
            "model": "SeparableConvCNN",
            "best_val_loss": stage1_best_val_loss,
            "best_epoch": stage1_best_epoch,
            "test_loss": stage1_test_loss,
            "test_acc": stage1_test_acc,
            "test_f1_macro": stage1_test_f1,
            "model_path": str(stage1_model_path),
            "loaded_from_checkpoint": use_pretrained_stage1,
        },
        "stage2": {
            "model": "GumbelMaskSeparableConvCNN",
            "best_val_loss": stage2_best_val_loss,
            "best_epoch": stage2_best_epoch,
            "test_loss": stage2_test_loss,
            "test_acc": stage2_test_acc,
            "test_f1_macro": stage2_test_f1,
            "model_path": str(stage2_model_path),
            "hard_bin_mask": hard_bin_mask.numpy(),
            "bin_keep_ratio": bin_keep_ratio,
            "loaded_from_checkpoint": use_pretrained_stage2,
        },
        "stage3": {
            "model": "SeparableConvCNN",
            "best_val_loss": stage3_best_val_loss,
            "best_epoch": stage3_best_epoch,
            "test_loss": stage3_test_loss,
            "test_acc": stage3_test_acc,
            "test_f1_macro": stage3_test_f1,
            "model_path": str(stage3_model_path),
            "loaded_from_checkpoint": use_pretrained_stage3,
        },
        "stage4": {
            "model": "GumbelChannelPruningCNN",
            "best_val_loss": stage4_best_val_loss,
            "best_epoch": stage4_best_epoch,
            "test_loss": stage4_test_loss,
            "test_acc": stage4_test_acc,
            "test_f1_macro": stage4_test_f1,
            "model_path": str(stage4_model_path),
            "final_mask": final_channel_mask,
            "pruning_stats": pruning_stats,
            "loaded_from_checkpoint": use_pretrained_stage4,
        },
    }
