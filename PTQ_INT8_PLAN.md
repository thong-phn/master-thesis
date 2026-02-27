# PTQ Full-Integer (INT8) Integration Plan

## Goal
Add a reproducible post-training quantization pipeline for LOSO training that:
1. Trains and saves the best FP32 checkpoint.
2. Builds representative calibration data from training split.
3. Converts PyTorch checkpoint to LiteRT/TFLite using LiteRT-Torch.
4. Applies post-training quantization flow for INT8 deployment.

## Scope
- Update `main-int8.py` with stage-based pipeline and runtime tracking.
- Add `train_loso_int8(...)` to `train.py` for INT8-oriented runs.
- Reuse existing `representative_data.py`.

## Implementation Steps
- [x] Create plan document and define tracking format.
- [x] Add `train_loso_int8(...)` wrapper in `train.py`.
- [x] Add tracking helper in `main-int8.py` (timestamped stage events).
- [x] Integrate representative data export stage in `main-int8.py`.
- [x] Integrate LiteRT-Torch conversion stage in `main-int8.py`.
- [x] Integrate LiteRT-Torch-based quantization stage for INT8 output.

## Tracking Schema
Each stage appends one event object:
- `time`: ISO timestamp
- `stage`: stage identifier (e.g. `train`, `representative_data`, `litert_convert`, `ptq_tflite`)
- `status`: `start` | `ok` | `error`
- `message`: human-readable progress
- `details`: optional metadata dictionary

## Expected Outputs
- FP32 checkpoint: `models/best_model_subject1_val.pth`
- Representative dataset: `models/representative_data.npz`
- INT8 TFLite: `models/best_model_int8.tflite`

## Notes
- Full-integer PTQ requires representative data calibration.
- LiteRT-Torch must be installed in the active environment.
- If conversion fails, tracking log captures the failing stage and message.
