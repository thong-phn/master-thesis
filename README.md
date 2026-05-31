# Efficient Multi-Stage Pruning for Human Activity Recognition (HAR)

## Overview
This repository provides a complete experimental framework aimed at building lightweight yet highly efficient neural networks for wearable sensors. Focusing on HAR tasks, the project applies an advanced pruning technique (using Gumbel-Softmax as a masking function) to identify the optimal input features and convolution channels needed to retain baseline performance.

It evaluates these concepts on open-sourced datasets ([UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and [WEAR](https://mariusbock.github.io/wear/)) using a LOSO cross-validation methodology, culminating in post-training optimized conversions to TensorFlow Lite for deployment on an STM32F4 MCU.

## Experiment Configuration
The network is trained using the Adam optimizer with a learning rate of 1e-3, a weight decay of 1e-6, a batch size of 64, and a weighted cross-entropy loss function. Additionally, Gumbel τ is initially set to 10 and gradually decays to 0 as the network reaches convergence.
| Dataset  | Activity      | Classes | Channels               | Window (s) | Overlap |
|----------|--------------:|--------:|------------------------|-----------:|--------:|
| WEAR     | Sport         |       8 | 6 (Left+Right Acc)     | 2.00       | 50%     |
| UCI-HAR  | Daily living  |       6 | 6 (Acc+Gyro)           | 2.56       | 50%     |

## Results
Annotation
- `TD`: Time domain data; `FFT`: Fast Fourier Transform preprocessing data; `DCT`: Discrete Cosine Transform preprocessing data; `IHW`: Integer Haar Wavelet preprocessing data
- `F32`: 32-bit floating-point weights and activations; `W8A16`: 8-bit integer weights and 16-bit integer activations
- `Input ratio`: Ratio of the pruned input size to the default input size. For example, an `Input ratio` of 0.424 means that if the default model has 100 input frequency bins, the pruned model uses an average of only 42 frequency bins to achieve comparable performance. This helps reduce the model's inference time and RAM usage on edge devices.
- `Model param. ratio`: Ratio of the pruned model parameters to the default model parameters. For example, a `Model param. ratio` of 0.5 means that if the default model has 100,000 parameters, the pruned model requires only 50,000 parameters to achieve comparable performance. This significantly reduces the model's flash usage and computational load.

**Overall**: With this pruning method, we achieved inference speeds 3.5 times faster than the baseline, using only half of the RAM and Flash memory, while maintaining a slightly higher performance

| Pre-processing | Input<br>ratio   | Model<br>param.<br>ratio | Accuracy<br>(F32)            | F1<br>(F32)                 | Accuracy<br>(W8A16)         | F1<br>(W8A16)               |Inference<br>(ms)            | RAM (KB)                    | ROM (KB)                     | 
|:---------------|:-----------------|:-------------------------|:-----------------------------|:----------------------------|:----------------------------|:----------------------------|:----------------------------|:----------------------------|:-----------------------------|
| Baseline-TD    | 1.000            | 1.000                    | 80.94<br><sub>±2.13</sub>    | 71.80<br><sub>±3.13</sub>   | 80.92<br><sub>±2.31</sub>   | 72.07<br><sub>±2.33</sub>   | 80.91<br><sub>±0.01</sub>   | 22.87<br><sub>±0.00</sub>   | 162.27<br><sub>±0.00</sub>   |
| DCT            | 1.000            | 1.000                    | 77.78<br><sub>±3.81</sub>    | 68.11<br><sub>±1.91</sub>   | 77.41<br><sub>±4.12</sub>   | 67.92<br><sub>±2.14</sub>   | 80.90<br><sub>±0.01</sub>   | 22.87<br><sub>±0.00</sub>   | 162.27<br><sub>±0.00</sub>   |
| Pruned-DCT     | 0.424            | 0.498                    | 79.03<br><sub>±0.98</sub>    | 68.36<br><sub>±1.07</sub>   | 79.00<br><sub>±1.00</sub>   | 68.35<br><sub>±1.07</sub>   | 24.72<br><sub>±6.57</sub>   |**11.22<br><sub>±1.79</sub>**| 86.02<br><sub>±12.22</sub>   |
| IHW            | 1.000            | 1.000                    | 81.09<br><sub>±2.24</sub>    | 72.07<br><sub>±1.26</sub>   | 81.08<br><sub>±2.26</sub>   | 72.07<br><sub>±1.28</sub>   | 80.91<br><sub>±0.01</sub>   | 22.87<br><sub>±0.00</sub>   | 162.27<br><sub>±0.00</sub>   |
| Pruned-IHW     | 0.490            | 0.555                    | **81.62<br><sub>±0.63</sub>**|**72.30<br><sub>±0.57</sub>**|**81.56<br><sub>±0.64</sub>**|**72.27<br><sub>±0.56</sub>**|**22.15<br><sub>±5.77</sub>**| 11.63<br><sub>±2.20</sub>   |**74.19<br><sub>±13.34</sub>**|

## Methodologies & Training Pipeline
### Methodologies
- **Model Architecture**: Centered around **Separable Convolutional Neural Networks (SeparableConvCNN)**, which natively strip down the parameter count while preserving spatial and temporal learning capacity.
- **Data Preprocessing**: Compares various data representation techniques including **Raw Time-Domain** signals, **Fast Fourier Transform (FFT)**, and **Discrete Cosine Transform (DCT)**. 
- **Leave-One-Subject-Out (LOSO)**: Ensures realistic out-of-sample generalization for wearable users.
- **Gumbel-Softmax Pruning Methods**:
  - **Input Pruning**: Identifies and drops uninformative time-steps or frequency bins (in FFT/DCT) via a learnable mask.
  - **Channel Pruning**: Dynamically removes unused convolutional channels during training.
- **TFLite Post-Training Quantization (PTQ)**: Provides a deployment-ready pipeline (`PyTorch .pth` → `ONNX` → `TF SavedModel` via `onnx2tf` → `.tflite`). Target quantization formats include `W8A16_FLOAT_IO`, `W8A16_INT_IO`, and `W8A8_INT_IO`.

### Training Pipeline
The project adopts robust multi-stage pipelines (found across 3-stage or 5-stage variations) to progressively simplify the model without sacrificing accuracy. For instance, the **Five-Stage Multi-Pruning Pipeline** works as follows:
- **Stage 1 (Pre-training)**: Train the standard `SeparableConvCNN` without masking to establish a solid weight initialization and baseline.
- **Stage 2 (Input Pruning Logic)**: Introduce Gumbel masking on inputs and train briefly to establish sparse binary masks indicating which frequency bins/steps to keep.
- **Stage 3 (Input Application)**: Extract the pruned input and fully retrain the `SeparableConvCNN`.
- **Stage 4 (Channel Pruning Logic)**: Introduce channel-based Gumbel masking to identify which network filters can be safely dropped.
- **Stage 5 (Physical Pruning & Fine-Tuning)**: Physically slice away unused sub-tensor parameters to yield a true compact model and fine-tune for the final evaluation. 