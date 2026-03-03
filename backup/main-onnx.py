import torch
import litert_torch
import numpy as np
import tensorflow as tf
import os

from torch.utils.data import DataLoader

from train import MyDataset
from model import SeparableConvCNN

num_channels = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_path = './models/best_model_subject1_val.pth'
dummy_input = torch.randn(1, 6, 65)
test_dataset = MyDataset(
    root_path='./uci-har',
    split='test',
    use_gyro=True,
)

# Load the best model
model = SeparableConvCNN(num_channels=num_channels).to(device)
state = torch.load(pretrain_path, map_location=device)
model.load_state_dict(state, strict=False)
print(f"[Base model] Loaded pretrain weights from {[pretrain_path]}")
model.eval()
# Calculate the accuracy of the best model
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

q_device = next(model.parameters()).device
test_correct = 0
test_total = 0

with torch.no_grad():
    for fft_mag, labels in test_loader:
        fft_mag = fft_mag.to(q_device)
        labels = labels.to(q_device)

        outputs = model(fft_mag)
        preds = torch.argmax(outputs, dim=1)

        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()

test_acc = 100.0 * test_correct / max(test_total, 1)
print(f"[Base model] Test accuracy: {test_acc:.2f}%")

# To ONNX
# 3. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "my_model.onnx",
    export_params=True,
    input_names=['input'],
    output_names=['output']
)
print("ONNX export complete!")


