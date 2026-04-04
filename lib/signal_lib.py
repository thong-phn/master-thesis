from scipy.fftpack import dct
from scipy.signal import butter, filtfilt
import numpy as np
import csv

def _remove_gravity(signal, cutoff=0.3, fs=50, order=3):
    """High-pass filter to remove gravity (same as UCI-HAR preprocessing)."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, signal, axis=0)

def _load_and_window_subject_csv(label_map, file_path, window_size=100, step_size=50):
    """
    Load a single subject's CSV, extract left_arm_acc_x/y/z and label,
    map labels via label_map, and apply a sliding window.

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
            if lbl_str in label_map:
                mapped_labels.append(label_map[lbl_str])
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