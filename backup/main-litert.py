# pytorch --> tflite
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

# Quantize


# Convert to tftlite
sample_inputs = (torch.randn(1, 6, 65,),)
torch_output = model(*sample_inputs)

edge_model = litert_torch.convert(model, sample_inputs)
edge_output = edge_model(*sample_inputs)

if np.allclose(torch_output.detach().numpy(), edge_output, atol=1e-5):
    print("[Tflite model] Inference result with Pytorch and TfLite was within tolerance")
else:
    print("[Tflite model] Something wrong with Pytorch --> TfLite")

# Calculate the accuracy of tflite model
# Evaluate TFLite model on the test set
tflite_correct = 0
tflite_total = 0

with torch.no_grad():
    for fft_mag, labels in test_loader:
        # Convert input to numpy on CPU for the TFLite wrapper
        if isinstance(fft_mag, torch.Tensor):
            inp = fft_mag.cpu().numpy()
        else:
            inp = np.array(fft_mag)
        # Run TFLite model per-sample (some delegates like XNNPack
        # may not support reshaping for large batches)
        if isinstance(labels, torch.Tensor):
            lbls = labels.cpu().numpy()
        else:
            lbls = np.array(labels)

        for i in range(lbls.shape[0]):
            sample = inp[i : i + 1]
            edge_out = edge_model(sample)
            edge_out = np.asarray(edge_out)
            pred = int(np.argmax(edge_out, axis=1)[0])
            tflite_total += 1
            tflite_correct += int(pred == int(lbls[i]))

tflite_acc = 100.0 * tflite_correct / max(tflite_total, 1)
print(f"[Tflite model] Test accuracy: {tflite_acc:.2f}%")

# Save the TFLite flatbuffer to disk if possible, then evaluate using
# TensorFlow Lite Interpreter to independently compute accuracy.
tflite_path = './models/edge_model.tflite'
edge_model.export(tflite_path)

# If saved successfully, attempt to run TensorFlow Lite Interpreter

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

itflite_correct = 0
itflite_total = 0

# Run per-sample to avoid delegate reshape issues
for fft_mag, labels in test_loader:
    if isinstance(fft_mag, torch.Tensor):
        inp = fft_mag.cpu().numpy()
    else:
        inp = np.array(fft_mag)

    if isinstance(labels, torch.Tensor):
        lbls = labels.cpu().numpy()
    else:
        lbls = np.array(labels)

    for i in range(lbls.shape[0]):
        sample = inp[i : i + 1]
        # Cast to interpreter expected dtype
        in_dtype = input_details[0]['dtype']
        sample = sample.astype(in_dtype)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        pred = int(np.argmax(out, axis=1)[0])
        itflite_total += 1
        itflite_correct += int(pred == int(lbls[i]))

itflite_acc = 100.0 * itflite_correct / max(itflite_total, 1)
print(f"[TFLite Interpreter] Test accuracy: {itflite_acc:.2f}%")

