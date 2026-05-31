# Efficient Multi-Stage Pruning for Human Activity Recognition (HAR)

## Overview
This repository provides a complete experimental framework aimed at building lightweight yet highly efficient neural networks for wearable sensors. Focusing on HAR tasks, the project applies an advanced pruning technique (using Gumbel-Softmax as a masking function) to identify the optimal input features and convolution channels needed to retain baseline performance.

It evaluates these concepts on open-sourced datasets ([UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and [WEAR](https://mariusbock.github.io/wear/)) using a LOSO cross-validation methodology, culminating in post-training optimized conversions to TensorFlow Lite for deployment on an STM32F4 MCU.

## Experiment Configuration
The input size per axis is 100 samples for the WEAR dataset (50 Hz, 2 s) and 128 samples for the UCI-HAR dataset. When utilizing all 6 axes, the input tensor shapes are (1, 100, 6) and (1, 128, 6), respectively.

| Dataset  | Activity      | Classes | Channels               | Window (s) | Overlap | Number of subjects |
|----------|--------------:|--------:|------------------------|-----------:|--------:|-------------------:|
| WEAR     | Sport         |       8 | 6 (Left+Right Acc)     | 2.00       | 50%     | 30                 |
| UCI-HAR  | Daily living  |       6 | 6 (Acc+Gyro)           | 2.56       | 50%     | 24                 |

The model leverages Depthwise Separable Convolutions to maintain a low parameter count. The baseline model has approximately 37k parameters, which will be further pruned to be around 18k parameters. The detailed model architecture is discussed in [model_arch](docs/model_arch.md).

The network is trained using the Adam optimizer with a learning rate of 1e-3, a weight decay of 1e-6, a batch size of 64, and a weighted cross-entropy loss function. Additionally, Gumbel τ is initially set to 10 and gradually decays to 0 as the network reaches convergence. 

## Results
With this pruning method, we achieved inference speeds 3.5 times faster than the baseline, using only half of the RAM and Flash memory, while maintaining a slightly higher performance

| Pre-processing | Input<br>ratio   | Model<br>param.<br>ratio | Accuracy<br>(F32)            | F1<br>(F32)                 | Accuracy<br>(W8A16)         | F1<br>(W8A16)               |Inference<br>(ms)            | RAM (KB)                    | ROM (KB)                     | 
|:---------------|:-----------------|:-------------------------|:-----------------------------|:----------------------------|:----------------------------|:----------------------------|:----------------------------|:----------------------------|:-----------------------------|
| Baseline-TD    | 1.000            | 1.000                    | 80.94<br><sub>±2.13</sub>    | 71.80<br><sub>±3.13</sub>   | 80.92<br><sub>±2.31</sub>   | 72.07<br><sub>±2.33</sub>   | 80.91<br><sub>±0.01</sub>   | 22.87<br><sub>±0.00</sub>   | 162.27<br><sub>±0.00</sub>   |
| DCT            | 1.000            | 1.000                    | 77.78<br><sub>±3.81</sub>    | 68.11<br><sub>±1.91</sub>   | 77.41<br><sub>±4.12</sub>   | 67.92<br><sub>±2.14</sub>   | 80.90<br><sub>±0.01</sub>   | 22.87<br><sub>±0.00</sub>   | 162.27<br><sub>±0.00</sub>   |
| Pruned-DCT     | 0.424            | 0.498                    | 79.03<br><sub>±0.98</sub>    | 68.36<br><sub>±1.07</sub>   | 79.00<br><sub>±1.00</sub>   | 68.35<br><sub>±1.07</sub>   | 24.72<br><sub>±6.57</sub>   |**11.22<br><sub>±1.79</sub>**| 86.02<br><sub>±12.22</sub>   |
| IHW            | 1.000            | 1.000                    | 81.09<br><sub>±2.24</sub>    | 72.07<br><sub>±1.26</sub>   | 81.08<br><sub>±2.26</sub>   | 72.07<br><sub>±1.28</sub>   | 80.91<br><sub>±0.01</sub>   | 22.87<br><sub>±0.00</sub>   | 162.27<br><sub>±0.00</sub>   |
| Pruned-IHW     | 0.490            | 0.555                    | **81.62<br><sub>±0.63</sub>**|**72.30<br><sub>±0.57</sub>**|**81.56<br><sub>±0.64</sub>**|**72.27<br><sub>±0.56</sub>**|**22.15<br><sub>±5.77</sub>**| 11.63<br><sub>±2.20</sub>   |**74.19<br><sub>±13.34</sub>**|

Annotation
- `TD`: Time domain data; `FFT`: Fast Fourier Transform; `DCT`: Discrete Cosine Transform; `IHW`: Integer Haar Wavelet
- `F32`: 32-bit floating-point weights and activations; `W8A16`: 8-bit integer weights and 16-bit integer activations
- `Input ratio`: Ratio of the pruned input size to the default input size. For example, an `Input ratio` of 0.424 means that if the default model has 100 input frequency bins, the pruned model uses an average of only 42 frequency bins to achieve comparable performance. This helps reduce the model's inference time and RAM usage on edge devices.
- `Model param. ratio`: Ratio of the pruned model parameters to the default model parameters. For example, a `Model param. ratio` of 0.5 means that if the default model has 100,000 parameters, the pruned model requires only 50,000 parameters to achieve comparable performance. This significantly reduces the model's flash usage and computational load.
<!-- 
## MCU demonstration
TODO -->

## Methodologies
![overview](docs/overview.png)
**Methodologies**:
- **Data Preprocessing**: Preprocessing techniques including raw time-domain signals, Fast Fourier Transform (FFT), Discrete Cosine Transform (DCT), and Integer Haar Wavelet (IHW). 
- **Validation Scheme:**: Leave-One-Subject-Out (LOSO) ensures eneralization.
- **Gumbel-Softmax Pruning Sequences**:
  - **Input Pruning**: Identifies and drops uninformative frequency bins (FFT/DCT/IHW) via a learnable mask.
  - **Channel Pruning**: Dynamically removes unused convolutional channels during training.
- **TFLite Post-Training Quantization (PTQ)**: Provides a deployment-ready pipeline (`PyTorch .pth` → `ONNX` → `TF SavedModel` via `onnx2tf` → `.tflite`). Target quantization format is `W8A16_INT_IO`
- **Profiling**: Measure predictive and efficiency metrics on STM32 MCUs.

**Training Pipeline**
![training_pipeline](docs/training_pipeline.png)