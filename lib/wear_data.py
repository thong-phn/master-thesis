import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from scipy.fftpack import dct

# Mapping of fine-grained text labels to broader integer labels [0, 7]
# 1. jogging (jogging and jogging sidestep, skipping, butt-kicks, rotating arms)                      
# 2. stretching (stretching lunging, hamstrings, triceps, shoulders)                        
# 3. lunges                              
# 4. sit-ups (sit-ups and sit-ups complex)                            
# 5. push-ups (push-ups and push-ups complex)                           
# 6. burpees                             
# 7. bench-dips                          
# 8. idle (null) 

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


import csv

def load_and_window_subject_data(file_path, window_size=100, step_size=50):
    """
    Loads a single subject's CSV, extracts 'left_arm_acc_x,y,z' and 'label',
    maps the labels, and applies a sliding window.
    
    Args:
        file_path: pathlib.Path to the subject's csv file
        window_size: number of samples per window (default: 100 for 2s at 50Hz)
        step_size: number of samples to slide the window (default: 50 for 50% overlap)
        
    Returns:
        signals: numpy array of shape (num_windows, 3, window_size)
        labels: numpy array of shape (num_windows,)
    """
    acc_data = []
    mapped_labels = []

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        # WEAR CSV is: sbj_id, right_arm_acc_x, right_arm_acc_y, right_arm_acc_z, 
        # right_leg_acc_x, right_leg_acc_y, right_leg_acc_z, 
        # left_leg_acc_x, left_leg_acc_y, left_leg_acc_z, 
        # left_arm_acc_x, left_arm_acc_y, left_arm_acc_z, label
        
        try:
            lx_idx = headers.index('left_arm_acc_x')
            ly_idx = headers.index('left_arm_acc_y')
            lz_idx = headers.index('left_arm_acc_z')
            lbl_idx = headers.index('label')
        except ValueError as e:
            print(f"Error finding columns in {file_path}: {e}")
            return np.array([]), np.array([])

        for row in reader:
            lbl_str = row[lbl_idx].strip()
            
            # Skip rows where either x, y, or z is empty
            if not row[lx_idx].strip() or not row[ly_idx].strip() or not row[lz_idx].strip():
                continue
                
            acc_data.append([float(row[lx_idx]), float(row[ly_idx]), float(row[lz_idx])])
            
            if lbl_str in LABEL_MAP:
                mapped_labels.append(LABEL_MAP[lbl_str])
            else:
                mapped_labels.append(-1)

    acc_data = np.array(acc_data, dtype=np.float32)
    mapped_labels = np.array(mapped_labels, dtype=np.int64)
    
    num_samples = len(acc_data)
    
    windows_signals = []
    windows_labels = []
    
    # Sliding window
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        
        # Extracted window
        window_signal = acc_data[start:end]
        window_label_seq = mapped_labels[start:end]
        
        # Most frequent label in the window using offset to support -1
        counts = np.bincount(window_label_seq + 1)
        mode_idx = counts.argmax()
        mode_label = mode_idx - 1
        
        # Discard window if mode is invalid (-1)
        if mode_label == -1:
            continue
        
        # Append signal transposed to shape (3, window_size)
        windows_signals.append(window_signal.T)
        windows_labels.append(mode_label)
        
    return np.array(windows_signals, dtype=np.float32), np.array(windows_labels, dtype=np.int64)


class WearDataset(Dataset):
    def __init__(self, root_path, split=None, subject_ids=None, preprocessing='fft'):
        """
        Load WEAR data for specific subjects with optional preprocessing.
        
        Args:
            root_path: Path object to the WEAR dataset root directory
            split: 'train' or 'test' subdirectory (optional)
            subject_ids: List of integers specifying which subjects to load
            preprocessing: 'fft', 'dct', or 'no' (default: 'fft')
        """
        all_signals = []
        all_labels = []
        all_subjects = []
        
        root_path = Path(root_path)
        
        # If split is specified, use the split subdirectory
        if split in ['train', 'test']:
            data_path = root_path / split
        else:
            data_path = root_path
            
        self.preprocessing = preprocessing
        
        if subject_ids is None:
            # Load all available subjects if none specified
            subject_ids = []
            for csv_file in sorted(data_path.glob("sbj_*.csv")):
                sbj_id = int(csv_file.stem.split('_')[1])
                subject_ids.append(sbj_id)
        
        for sbj_id in subject_ids:
            file_path = data_path / f"sbj_{sbj_id}.csv"
            if not file_path.exists():
                print(f"Warning: {file_path} not found. Skipping.")
                continue
                
            signals, labels = load_and_window_subject_data(file_path)
            
            all_signals.append(signals)
            all_labels.append(labels)
            all_subjects.extend([sbj_id] * len(labels))
            
        if len(all_signals) > 0:
            self.signals = np.concatenate(all_signals, axis=0)
            self.labels = np.concatenate(all_labels, axis=0)
            self.subjects = np.array(all_subjects)
        else:
            self.signals = np.array([])
            self.labels = np.array([])
            self.subjects = np.array([])

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _compute_fft_magnitude(signal):
        """Compute FFT magnitude spectrum (one-sided)."""
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
        """Compute DCT (Discrete Cosine Transform)."""
        dct_vals = dct(signal, type=2, axis=-1, norm='ortho')
        return np.abs(dct_vals)

    def __getitem__(self, idx):
        # Get signal and apply preprocessing
        signal = self.signals[idx]

        if self.preprocessing == 'dct':
            mag = self._compute_dct(signal)
        elif self.preprocessing == 'no':
            mag = signal
        elif self.preprocessing == 'fft':
            mag = self._compute_fft_magnitude(signal)
        else:
            mag = signal

        # Return (input, target) as tensors
        x = torch.tensor(mag, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
