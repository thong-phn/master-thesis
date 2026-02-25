import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# Dataset 
class MyDataset(Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        subject_ids=None,
        use_gyro=False,
        log_fft=True,
        fft_norm_stats=None,
    ):
        """
        Load UCI-HAR data, compute FFT
        Args:
            root_path: Path to dataset
            split: 'train' or 'test'
            subject_ids: List of subject IDs to filter (optional)
            use_gyro: Whether to include gyro (angular velocity) data (default: False)
            log_fft: Apply log1p to FFT magnitudes
            fft_norm_stats: Tuple (mu, sigma) for FFT normalization, both shape (num_channels, num_freq_bins)
        """
        self.root_path = Path(root_path)
        self.split_path = self.root_path / split
        self.inertial_path = self.split_path/"Inertial Signals"
        self.use_gyro = use_gyro
        self.log_fft = log_fft
        self.fft_norm_stats = fft_norm_stats

        # Load Y (label) and subjects
        path_to_y_file = self.split_path/f"y_{split}.txt"
        path_to_subject_file = self.split_path/f"subject_{split}.txt"

        all_labels = np.loadtxt(path_to_y_file, dtype=int) - 1 # 0-indexed [0, 1, 2, 3, 4, 5]
        all_subjects = np.loadtxt(path_to_subject_file, dtype=int)
        
        # Load accelerometer data (body acceleration)
        signal_files = {       
            "X": f"body_acc_x_{split}.txt",
            "Y": f"body_acc_y_{split}.txt",
            "Z": f"body_acc_z_{split}.txt",
        }
        signals = []
        for axis in ["X", "Y", "Z"]:
            data = np.loadtxt(self.inertial_path/signal_files[axis])
            signals.append(data)

        # Load gyro data if requested
        if use_gyro:
            gyro_files = {
                "X": f"body_gyro_x_{split}.txt",
                "Y": f"body_gyro_y_{split}.txt",
                "Z": f"body_gyro_z_{split}.txt",
            }
            for axis in ["X", "Y", "Z"]:
                data = np.loadtxt(self.inertial_path/gyro_files[axis])
                signals.append(data)

        all_signals = np.stack(signals, axis=1) # Stack to shape (samples, num_channels, 128) where num_channels is 3 or 6

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

    def fit_fft_norm_stats(self):
        """Fit FFT normalization stats from this dataset only (typically train split)."""
        fft_mag = self._compute_fft_magnitude(self.signals)
        if self.log_fft:
            fft_mag = np.log1p(fft_mag)

        # Channel-wise normalization stats (shape: num_channels)
        # This is less aggressive than per-bin normalization and tends to be
        # more stable for cross-subject HAR.
        mu = fft_mag.mean(axis=(0, 2))  # Average over samples and freq bins
        sigma = fft_mag.std(axis=(0, 2))  # Std over samples and freq bins
        sigma = np.maximum(sigma, 1e-3)
        return mu, sigma

    def set_fft_norm_stats(self, mu, sigma):
        self.fft_norm_stats = (mu, sigma)

    def __getitem__(self, idx):
        # Get time-domain signal (3, 128) for accel only, or (6, 128) with gyro
        signal = self.signals[idx]

        # Apply FFT to each axis -> shape: (num_channels, num_freq_bins)
        fft_mag = self._compute_fft_magnitude(signal)

        if self.log_fft:
            fft_mag = np.log1p(fft_mag)

        if self.fft_norm_stats is not None:
            mu, sigma = self.fft_norm_stats
            # Broadcast: mu/sigma are (num_channels,), fft_mag is (num_channels, freq_bins)
            fft_mag = (fft_mag - mu[:, None]) / (sigma[:, None] + 1e-8)
        
        return torch.FloatTensor(fft_mag), torch.LongTensor([self.labels[idx]])[0]
        
# Training function
def train_loso(root_path, model_class, train_subjects, val_subjects, wandb_run=None, use_gyro=False, **train_kwargs):
    """
    Args:
        root_path: path to UCI-HAR
        model_class
        train_subjects
        val_subjects
        wandb_run:
        use_gyro: Whether to include gyro data (default: False)
        **train_kwargs
    """
    # Hyperparameters 
    epochs = train_kwargs.get('epochs', 30)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    model_path = Path(train_kwargs.get('model_path', './models/best_model.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dataset and dataloader
    # Exp-5: log1p only, no z-score normalization
    train_dataset = MyDataset(
        root_path,
        split='train',
        subject_ids=train_subjects,
        use_gyro=use_gyro,
        log_fft=True,
        fft_norm_stats=None,
    )

    val_dataset = MyDataset(
        root_path,
        split='train',
        subject_ids=val_subjects,
        use_gyro=use_gyro,
        log_fft=True,
        fft_norm_stats=None,
    )
    test_dataset = MyDataset(
        root_path,
        split='test',
        subject_ids=None,
        use_gyro=use_gyro,
        log_fft=True,
        fft_norm_stats=None,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Determine number of channels based on use_gyro
    num_channels = 6 if use_gyro else 3
    print(f"Using {num_channels} channels ({'accel + gyro' if use_gyro else 'accel only'})")

    # Training loop configuration
    model = model_class(num_channels=num_channels).to(device)
    criterion = nn.CrossEntropyLoss()
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
        # Train one epoch
        train_loss_sum = 0.0 # sum of training loss  
        train_correct = 0 # no. of training samples predicted correctly
        train_total = 0 # no. of training samples used
        
        model.train()

        for fft_mag, labels in train_dataloader:
            fft_mag, labels = fft_mag.to(device), labels.to(device)

            outputs = model(fft_mag) # 1. forward 
            loss = criterion(outputs, labels) # 2. loss
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
        
        print(f'Epoch [{epoch+1}/{epochs}]: '
              f'Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; '
              f'Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}')
        
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

    test_loss = test_loss_sum / max(test_total, 1)
    test_acc = 100.0 * test_correct / max(test_total, 1)
    
    print("-"*50)
    print(f"Summary:")
    print(f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    if wandb_run is not None: # tracking
        wandb_run.log({
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

    return {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "model_path": str(model_path),
    }




    
