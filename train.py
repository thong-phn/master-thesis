import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.ao.quantization.qconfig_mapping as qconfig_mapping
import copy


# Dataset 
class MyDataset(Dataset):
    def __init__(self, root_path, split='train', subject_ids = None, use_gyro=False):
        """
        Load UCI-HAR data, compute FFT
        Args:
            root_path: Path to dataset
            split: 'train' or 'test'
            subject_ids: List of subject IDs to filter (optional)
            use_gyro: Whether to include gyro (angular velocity) data (default: False)
        """
        self.root_path = Path(root_path)
        self.split_path = self.root_path / split
        self.inertial_path = self.split_path/"Inertial Signals"
        self.use_gyro = use_gyro

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

    def __getitem__(self, idx):
        # Get time-domain signal (3, 128) for accel only, or (6, 128) with gyro
        signal = self.signals[idx]
        
        # Apply FFT to each axis
        # Output: (3, 31) without gyro, or (6, 31) with gyro
        fft_mag = []
        for axis_signal in signal:
            fft_vals = np.fft.rfft(axis_signal)
            mag = np.abs(fft_vals) / len(axis_signal)
            
            # One-sided amplitude scaling
            if len(axis_signal) % 2 == 0:
                mag[1:-1] *= 2
            else:
                mag[1:] *= 2
            
            fft_mag.append(mag)
        
        fft_mag = np.stack(fft_mag, axis=0)  # shape: (num_channels, 31)
        
        return torch.FloatTensor(fft_mag), torch.LongTensor([self.labels[idx]])[0]
        
# Training function
def train_loso(root_path, model_class, train_subjects, val_subjects, wandb_run=None, use_gyro=False, use_qat=False, qat_backend='qnnpack', pretrained_float_path=None, **train_kwargs):
    """
    Args:
        root_path: path to UCI-HAR
        model_class
        train_subjects
        val_subjects
    use_gyro: Whether to include gyro data (default: False)
        use_qat: Whether to apply Quantization Aware Training
        qat_backend: QAT backend (e.g., 'qnnpack', 'x86', 'fbgemm')
        pretrained_float_path: Path to pre-trained floating point model for QAT fine-tuning
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
    train_dataset = MyDataset(root_path, split='train', subject_ids=train_subjects, use_gyro=use_gyro)
    val_dataset = MyDataset(root_path, split='train', subject_ids=val_subjects, use_gyro=use_gyro)
    test_dataset = MyDataset(root_path, split='test', subject_ids=None, use_gyro=use_gyro)

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
    
    # Load pre-trained weights if provided (Crucial for QAT fine-tuning)
    if pretrained_float_path is not None:
        pretrained_path = Path(pretrained_float_path)
        if pretrained_path.exists():
            print(f"Loading pre-trained float weights from {pretrained_float_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
        else:
            print(f"WARNING: Pre-trained file not found at {pretrained_float_path}. Model will be initialized randomly.")

    # Output prefix for saving models
    model_prefix = "qat_" if use_qat else "float_"

    # QAT Preparation
    if use_qat:
        print(f"Preparing model for QAT targeting backend: {qat_backend}")
        torch.backends.quantized.engine = qat_backend
        qconfig_map = qconfig_mapping.get_default_qat_qconfig_mapping(qat_backend)
        
        # --- Partial Quantization ---
        # Keep sensitive layers (first conv and final fc) in float32 for highest accuracy.
        print("Applying partial quantization (keeping 'sep_conv1' and 'fc2' in float32)")
        qconfig_map.set_module_name("sep_conv1", None)
        qconfig_map.set_module_name("fc2", None)

        # We need a sample input for FX tracing
        sample_input = torch.randn(1, num_channels, 31).to(device)
        model.eval()
        model = quantize_fx.prepare_qat_fx(model, qconfig_map, sample_input)
        print("Model prepared for QAT.")
        
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if use_qat:
        print("Using ReduceLROnPlateau for QAT Partial Fine-Tuning.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

    best_val_loss = float('inf')
    best_epoch = 0 # 
    epochs_no_improve = 0 # early stopping

    print("-"*50)
    # Training loop
    for epoch in range(epochs):
        # Freeze BN Stats and Observers halfway through QAT to stabilize scales and running averages
        if use_qat and epoch >= epochs // 2:
            model.apply(torch.ao.quantization.disable_observer)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

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
        
        if use_qat:
            scheduler.step(val_loss) # Restored back to Step LR scheduler on val loss
        else:
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
              f'Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}; '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if wandb_run is not None: # tracking
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_val_loss,
                "lr": optimizer.param_groups[0]["lr"],  # actual LR
            })

        if epochs_no_improve >= patience: # Early stop on both float and QAT
            print(f"Early Stopping: Epoch [{epoch+1}/{epochs}] (patience={patience}, min_delta={min_delta}).")
            break


    # Test with best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    if use_qat:
        print("Converting QAT model to fully quantized INT8 model for evaluation...")
        model.eval()
        # Convert to quantized model (creates FakeQuant modules into actual quant/dequant)
        # Note: The converted model typically runs on CPU backend (qnnpack). 
        # Moving to CPU for exact validation of the integer math.
        model.to('cpu')
        quantized_model = quantize_fx.convert_fx(copy.deepcopy(model))
        
        # Save the fully quantized int8 model as well
        int8_model_path = model_path.parent / f"int8_{model_path.name}"
        torch.save(quantized_model.state_dict(), int8_model_path)
        print(f"Saved fully quantized INT8 model to {int8_model_path}")
        
        eval_model = quantized_model
        eval_device = torch.device('cpu')
    else:
        model.eval()
        eval_model = model
        eval_device = device

    test_loss_sum, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for fft_mag, labels in test_dataloader:
            fft_mag, labels = fft_mag.to(eval_device), labels.to(eval_device)
            outputs = eval_model(fft_mag)
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




    
