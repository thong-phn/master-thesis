# pytorch --> onnx --> tflite
import os
import tensorflow as tf
import torch

from train import MyDataset
import numpy as np
import sys

def evaluate_saved_model(concrete_func, dataloader):
    # Determine input name and expected shape
    input_specs = concrete_func.structured_input_signature[1]
    input_name = list(input_specs.keys())[0]

    total, correct = 0, 0
    for batch, labels in dataloader:
        # prepare numpy inputs, transpose to channel-last if necessary
        if isinstance(batch, torch.Tensor):
            arr = batch.cpu().numpy().astype('float32')
        else:
            arr = np.array(batch, dtype='float32')
        if arr.ndim == 3:
            arr = arr.transpose(0, 2, 1)

        for i in range(arr.shape[0]):
            inp = tf.convert_to_tensor(arr[i:i+1])
            out = concrete_func(**{input_name: inp})
            # get first tensor output
            out_val = list(out.values())[0].numpy()
            pred = np.argmax(out_val, axis=-1)[0]
            gt = int(labels[i].item())
            total += 1
            correct += int(pred == gt)
    acc = correct / max(total, 1)
    print(f"SavedModel accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc

def evaluate_tflite_model(tflite_buf, dataloader):
    # tflite_buf can be bytes or a filepath
    import io
    if isinstance(tflite_buf, (bytes, bytearray)):
        interpreter = tf.lite.Interpreter(model_content=bytes(tflite_buf))
    else:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_buf))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    total, correct = 0, 0

    in_scale, in_zero = input_details.get('quantization', (0.0, 0))
    out_scale, out_zero = output_details.get('quantization', (0.0, 0))

    for batch, labels in dataloader:
        if isinstance(batch, torch.Tensor):
            arr = batch.cpu().numpy().astype('float32')
        else:
            arr = np.array(batch, dtype='float32')
        if arr.ndim == 3:
            arr = arr.transpose(0, 2, 1)

        for i in range(arr.shape[0]):
            sample = arr[i:i+1]
            # Resize if interpreter expects different shape
            try:
                interpreter.resize_tensor_input(input_details['index'], sample.shape)
                interpreter.allocate_tensors()
            except Exception:
                pass

            # quantize input if needed
            if input_details['dtype'] == np.int8 or input_details['dtype'] == np.uint8:
                if in_scale and in_scale != 0:
                    q = (sample / in_scale).round().astype(input_details['dtype']) + in_zero
                else:
                    q = sample.astype(input_details['dtype'])
                interpreter.set_tensor(input_details['index'], q)
            else:
                interpreter.set_tensor(input_details['index'], sample.astype(input_details['dtype']))

            interpreter.invoke()
            out = interpreter.get_tensor(output_details['index'])
            # dequantize if needed
            if output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8:
                if out_scale and out_scale != 0:
                    out = (out.astype('float32') - out_zero) * out_scale
            pred = np.argmax(out, axis=-1)[0]
            gt = int(labels[i].item())
            total += 1
            correct += int(pred == gt)

    acc = correct / max(total, 1)
    print(f"TFLite model accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc


def run_converter(converter):
    # wrapper to run conversion and surface helpful errors
    try:
        return converter.convert()
    except Exception as e:
        raise RuntimeError('TFLite conversion failed: ' + str(e))

def convert_int8_io_int8(concrete_func, keras_model, dataset_generator,
                         dir_path, filename, disable_per_channel=False):
    """
    Convert to INT8 TFLite using a concrete function + Keras model (or None).
    - `concrete_func`: tf.function.get_concrete_function(...) or a concrete function
    - `keras_model`: a Keras model instance or None (some converters require it)
    - `dataset_generator`: a callable that yields lists of numpy float32 inputs for calibration
    - `dir_path`, `filename`: where to write the tflite bytes
    """
    os.makedirs(dir_path, exist_ok=True)
    print('Converting TensorFlow Lite int8 quantized model...', flush=True)

    converter_quantize = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], keras_model)
    if disable_per_channel:
        try:
            converter_quantize._experimental_disable_per_channel = True
            print('Per-channel quantization disabled (override).')
        except Exception:
            pass

    converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quantize.representative_dataset = dataset_generator
    # Force int8 ops / I/O
    converter_quantize.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quantize.target_spec.supported_types = [tf.int8]
    converter_quantize.inference_input_type = tf.int8
    converter_quantize.inference_output_type = tf.int8

    tflite_quant_model = run_converter(converter_quantize)
    out_file = os.path.join(dir_path, filename)
    with open(out_file, 'wb') as f:
        f.write(tflite_quant_model)
    print('Wrote INT8 TFLite to', out_file)
    return tflite_quant_model

def rep_gen_from_dataloader(dataloader, max_samples=1024):
    produced = 0
    for batch, _ in dataloader:
        # batch: torch.Tensor (N, C, L) -> convert to float32 numpy
        if isinstance(batch, torch.Tensor):
            arr = batch.cpu().numpy().astype('float32')
        else:
            arr = np.array(batch, dtype='float32')
        # Convert to channel-last `(N, L, C)` which many TF models expect
        if arr.ndim == 3:
            arr = arr.transpose(0, 2, 1)
        for i in range(arr.shape[0]):
            yield [arr[i:i+1]]    # yield list of inputs for multi-input models
            produced += 1
            if produced >= max_samples:
                return
            
# Create a test dataloader for representative dataset (uses UCI-HAR folder)
test_dataset = MyDataset(root_path='./uci-har', split='test', use_gyro=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# load SavedModel then get concrete function (example input shape [1,6,65])
saved_model_dir = './tf_saved_model_dir'
loaded = tf.saved_model.load(saved_model_dir)
# if the SavedModel has a 'serving_default' signature:
concrete = loaded.signatures['serving_default']
# Evaluate SavedModel (before quantization)
print('Evaluating SavedModel (float) on test set...')
evaluate_saved_model(concrete, test_loader)

# Convert to INT8 TFLite and evaluate (after quantization)
print('Converting to INT8 TFLite and evaluating...')
tflite_bytes = convert_int8_io_int8(concrete, None, lambda: rep_gen_from_dataloader(test_loader), './models', 'edge_model_int8.tflite')
evaluate_tflite_model(tflite_bytes, test_loader)