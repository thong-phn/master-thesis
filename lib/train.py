import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.fftpack import dct

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

        # Load Y (label) and subjects
        path_to_y_file = self.split_path/f"y_{split}.txt"
        path_to_subject_file = self.split_path/f"subject_{split}.txt"

        all_labels = np.loadtxt(path_to_y_file, dtype=int) - 1 # 0-indexed [0, 1, 2, 3, 4, 5]
        all_subjects = np.loadtxt(path_to_subject_file, dtype=int)
        
        # Load accelerometer data (body acceleration)
        # body has better result than total
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
    def _compute_dct(signal):
        dct_vals = dct(signal, type=2, axis=-1, norm='ortho')
        return np.abs(dct_vals)

    def __getitem__(self, idx):
        # Get time-domain signal (6, 128) for accel + gyro
        signal = self.signals[idx]

        if self.preprocessing == 'dct':
            mag = self._compute_dct(signal)
        elif self.preprocessing == 'no':
            mag = self.signals[idx]
        elif self.preprocessing == 'fft':
            mag = self._compute_fft_magnitude(signal)
        else:
            mag = None

        return torch.FloatTensor(mag), torch.LongTensor([self.labels[idx]])[0]
        
# Training function
def train_loso(root_path, model_class, train_subjects, val_subjects, wandb_run=None, dataset_class=None, num_classes=6, num_channels=6, **train_kwargs):
    """
    Args:
        root_path: path to dataset
        model_class: neural network model class
        train_subjects: list of training subject IDs
        val_subjects: list of validation subject IDs
        wandb_run: wandb run object for logging
        dataset_class: dataset class to use (default: UCIHAR_Dataset)
        num_classes: number of activity classes (default: 6 for UCI-HAR)
        num_channels: number of input channels (default: 6 for UCI-HAR)
        **train_kwargs: additional training hyperparameters
    """
    # Use UCIHAR_Dataset by default if not specified
    if dataset_class is None:
        dataset_class = UCIHAR_Dataset
    
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
    train_dataset = dataset_class(
        root_path,
        split='train',
        subject_ids=train_subjects,
        preprocessing=preprocessing,
    )

    val_dataset = dataset_class(
        root_path,
        split='train',
        subject_ids=val_subjects,
        preprocessing=preprocessing,
    )
    test_dataset = dataset_class(
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
    model = model_class(num_classes=num_classes, num_channels=num_channels, freq_bins=freq_bins).to(device)
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

    # Mask will be automatically calculated as hard mask (0 or 1) from bin_logits
    
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




    
