import torch
import torch.nn as nn
from model import SeparableConvCNN
import onnx
import onnxruntime as ort
import os
import tensorflow as tf
import numpy as np

# Load the dataset logic from train to build the representative dataset generator
from train import MyDataset
from torch.utils.data import DataLoader

def get_representative_dataset_generator():
    # Load training data for representative PTQ calibration
    train_dataset = MyDataset(root_path='uci-har/data', split='train', subject_ids=[3, 5, 6, 7], use_gyro=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    def representative_dataset():
        for i, (data, _) in enumerate(train_loader):
            if i > 500: # Use only a subset to save time
                break
            # TF expects the batch dim, but we should yield single items
            # PyTorch shape is (B, C, L), TF takes exactly the same if we export correctly
            # Actually, tf.lite representative datasets require the exact input shape of the TFLite model
            # which we will export as (1, 6, 31)
            yield [data.numpy().astype(np.float32)]
            
    return representative_dataset

def main():
    model_path = 'models/qat_best_model_subject1_val.pth'
    onnx_path = 'models/model.onnx'
    tflite_path = 'models/model_ptq_int8.tflite'
    
    print("1. Loading PyTorch Partial QAT Model...")
    model = SeparableConvCNN(num_channels=6)
    
    # Needs to match the modified partial QAT graph (quantization applied to the middle)
    # The saved model has FakeQuantize nodes embedded.
    import torch.ao.quantization.quantize_fx as quantize_fx
    import torch.ao.quantization.qconfig_mapping as qconfig_mapping
    
    qat_backend = 'qnnpack'
    qconfig_map = qconfig_mapping.get_default_qat_qconfig_mapping(qat_backend)
    qconfig_map.set_module_name("sep_conv1", None)
    qconfig_map.set_module_name("fc2", None)
    
    # We must prepare the model exactly as it was during training before loading
    sample_input = torch.randn(1, 6, 31)
    model.eval()
    model = quantize_fx.prepare_qat_fx(model, qconfig_map, sample_input)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Convert to a fully quantize/dequantize integer model (removing fakequantize tracking)
    # The edges will remain float, the middle will be quantized math.
    print("2. Converting to PyTorch INT8 math representation...")
    model = quantize_fx.convert_fx(model)
    
    print("3. Exporting to ONNX...")
    # ONNX export
    torch.onnx.export(
        model, 
        sample_input, 
        onnx_path, 
        export_params=True, 
        opset_version=13, # Important for quantization ops
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
    )
    print(f"ONNX Model saved to {onnx_path}")
    
    
    # After generating ONNX, we must convert it to TFLite
    print("4. Converting ONNX to TensorFlow and TFLite...")
    import subprocess
    
    # 4a. Run onnx2tf to convert ONNX to TF SavedModel
    # Note: onnx2tf outputs to a directory called `saved_model`
    subprocess.run(["onnx2tf", "-i", onnx_path, "-o", "models/tf_saved_model"], check=True)
    
    # 4b. Load TF SavedModel and convert to TFLite with PTQ
    print("5. Applying Post-Training Quantization (PTQ) to float edges...")
    converter = tf.lite.TFLiteConverter.from_saved_model("models/tf_saved_model")
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Assign representative dataset
    converter.representative_dataset = get_representative_dataset_generator()
    
    # Ensure optimal INT8 operations everywhere possible
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Force the input and output to be exactly INT8 for the Cortex M4 requirement
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    print(f"Successfully saved TFLite model with full INT8 constraints to {tflite_path}")

if __name__ == '__main__':
    main()
