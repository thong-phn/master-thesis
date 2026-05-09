# Efficient Multi-Stage Pruning and Quantization for Human Activity Recognition (HAR)

## 📌 Project Overview
This repository provides a complete PyTorch-based experimental framework aimed at building lightweight, highly efficient neural networks for wearable sensors and low-power edge devices. Focusing on Human Activity Recognition (HAR), the project applies advanced continuous pruning techniques—specifically using Gumbel-Softmax differentiable masking—to identify optimal features and internal parameters.

It evaluates these concepts primarily through two datasets (**UCI-HAR** and **WEAR**) using a rigorous **Leave-One-Subject-Out (LOSO)** cross-validation methodology, culminating in post-training optimized conversions to **TensorFlow Lite (TFLite)** suitable for edge deployment.

## ✨ Key Features & Methodologies
- **Model Architecture**: Centered around **Separable Convolutional Neural Networks (SeparableConvCNN)**, which natively strip down the parameter count while preserving spatial and temporal learning capacity.
- **Data Preprocessing**: Compares various data representation techniques including **Raw Time-Domain** signals, **Fast Fourier Transform (FFT)**, and **Discrete Cosine Transform (DCT)**. 
- **Leave-One-Subject-Out (LOSO)**: Ensures realistic out-of-sample generalization for wearable users.
- **Gumbel-Softmax Pruning Methods**:
  - **Input Pruning**: Identifies and drops uninformative time-steps or frequency bins (in FFT/DCT) via a learnable mask.
  - **Channel Pruning**: Dynamically removes unused convolutional channels during training.
- **TFLite Post-Training Quantization (PTQ)**: Provides a deployment-ready pipeline (`PyTorch .pth` → `ONNX` → `TF SavedModel` via `onnx2tf` → `.tflite`). Target quantization formats include `W8A16_FLOAT_IO`, `W8A16_INT_IO`, and `W8A8_INT_IO`.

## ⚙️ Multi-Stage Training Pipeline
The project adopts robust multi-stage pipelines (found across 3-stage or 5-stage variations) to progressively simplify the model without sacrificing accuracy. For instance, the **Five-Stage Multi-Pruning Pipeline** works as follows:
- **Stage 1 (Pre-training)**: Train the standard `SeparableConvCNN` without masking to establish a solid weight initialization and baseline.
- **Stage 2 (Input Pruning Logic)**: Introduce Gumbel masking on inputs and train briefly to establish sparse binary masks indicating which frequency bins/steps to keep.
- **Stage 3 (Input Application)**: Extract the pruned input and fully retrain the `SeparableConvCNN`.
- **Stage 4 (Channel Pruning Logic)**: Introduce channel-based Gumbel masking to identify which network filters can be safely dropped.
- **Stage 5 (Physical Pruning & Fine-Tuning)**: Physically slice away unused sub-tensor parameters to yield a true compact model and fine-tune for the final evaluation. 

## 📂 Repository Structure
- **`lib/`**: Contains foundational utilities.
  - `model.py`: PyTorch Module bindings (`SeparableConvCNN`, `GumbelMaskSeparableConvCNN`, etc.).
  - `ml_lib.py` / `signal_lib.py`: Core training loops, signal-processing techniques, and Gumbel-pruning integration routines.
  - `uci_train.py` / `wear_train*.py`: Definitions for dataset classes, specific validation procedures, and stage pipelines.
- **`[dataset]_main_loso_*.py`**: Main entrypoint scripts for running multi-stage Gumbel Pruning over different paradigms (baseline, input_pruning, channel_pruning, five_stage). Prefixed for **UCI** (`uci_`) or **WEAR** (`wear_`).
- **`[dataset]_quantize_loso_tflite_ptq.py`**: Scripts facilitating the automated pipeline to port pruned PyTorch models into quantized `TFLite` executables.
- **`log/`**: Experimental result outputs, validation scores, and tracking files (logging hyperparameter setups for FFT/DCT metrics).
- **`models/`**, **`fig/`**, **`scripts/`**, **`wandb/`**: Corresponding spaces for saved weights, visualizations, utility scripts, and tracking.

## 🚀 Getting Started Focus (Usage Example)
To run a channel pruning pipeline on UCI-HAR leveraging LOSO validation:
\`\`\`bash
python uci_main_loso_channel_pruning.py 
\`\`\`
To proceed with packaging a Stage 5 pruned model to a quantized TFLite package for Subjects 1, 2, and 3:
\`\`\`bash
python uci_quantize_loso_tflite_ptq.py --stage 5 --subjects '1,2,3'
\`\`\`

