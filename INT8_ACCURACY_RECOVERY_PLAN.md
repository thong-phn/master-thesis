# INT8 Accuracy Recovery Plan (PyTorch 90.36% -> INT8 69.70%)

## Why the drop is large
Likely causes in the current pipeline:
1. **Calibration mismatch**: PTQ calibration uses only 200 random samples; distribution may not represent full train set.
2. **Quantization configuration mismatch**: LiteRT conversion path may not be applying strict/optimal full-integer settings across all sensitive ops.
3. **Input quantization sensitivity**: Current model consumes FFT features; small quantization errors in low-amplitude bins can strongly affect logits.
4. **No layer sensitivity handling**: Pure PTQ quantizes all eligible layers equally; some layers may need higher precision.
5. **No post-conversion validation gate**: Conversion succeeds without checking intermediate tensor scale quality.

## Target
- Recover INT8 accuracy to within **<= 3-5% absolute** from FP32 baseline.
- Raise macro-F1 close to FP32 trend.

## Execution Plan

### Phase 1 — Verify baseline and reproducibility
- [x] Freeze one FP32 checkpoint and one fixed test set evaluation script.
- [x] Confirm TFLite evaluation path matches PyTorch preprocessing exactly (FFT, channel order, dtype, normalization).
- [x] Add deterministic seed for representative sampling and report class distribution.

### Phase 2 — Improve calibration quality (highest priority)
- [x] Increase representative set size from **200 -> 1000+** (or full train subset by class stratification).
- [x] Use **stratified representative sampling** by class (and optionally by subject) instead of pure random.
- [x] Run calibration ablation: 200 / 500 / 1000 / 2000 samples and record INT8 acc/F1.

### Phase 3 — Quantization setting sweep
- [ ] Keep **full-int8 only** (no dynamic-range, no float fallback) and tune within this constraint.
- [ ] Check converter flags and confirm resulting model truly uses full-int8 kernels.
- [ ] Capture and inspect input/output scale/zero-point and per-tensor quantization stats.
- [ ] Add a strict validation gate: fail run if input/output dtype is not `INT8`.

### Phase 4 — Model-side robustness improvements
- [ ] Add quantization-aware-friendly training tweaks in FP32 training:
  - [ ] slightly lower LR near convergence
  - [ ] mild regularization tuning (dropout/weight decay)
  - [ ] ensure stable activation ranges
- [ ] Optional: evaluate **QAT** path if PTQ remains >5% below FP32.

### Phase 5 — Validation and acceptance gates
- [ ] Add a report table per run:
  - FP32 acc/F1
  - INT8 acc/F1
  - absolute drop
  - calibration size
  - quant mode
- [ ] Define pass criteria: INT8 drop <= 5% absolute and no class collapse.

## Immediate next experiment (recommended)
1. Keep current checkpoint fixed.
2. Export representative data with **1000 stratified samples**.
3. Reconvert INT8 with same pipeline.
4. Re-evaluate and compare to current 69.70% / 0.682.

## Implementation Progress (Current)
- [x] Added INT8 evaluation metrics in `main-int8.py` (`int8_test_acc`, `int8_test_f1_macro`).
- [x] Added stratified representative sampling support in `representative_data.py`.
- [x] Added calibration sweep logic in `main-int8.py` for `[200, 500, 1000, 2000]`.
- [x] Execute sweep successfully and capture full results table.

## Experiment Results Log

| run_id | checkpoint | calibration_count | sampling_method | quant_mode | fp32_acc | int8_acc | int8_f1_macro | abs_drop | status | notes |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| baseline_ptq_200_random | `models/best_model_subject1_val.pth` | 200 | random | full_int8 | 90.36 | 69.70 | 0.6820 | 20.66 | completed | Initial PTQ result before Phase 2 updates |
| phase2_sweep_calib200 | `models/best_model_subject1_val.pth` | 200 | stratified | full_int8 | 90.36 | 72.9216 | 0.7234 | 17.4415 | completed | Improved over random-200 baseline |
| phase2_sweep_calib500 | `models/best_model_subject1_val.pth` | 500 | stratified | full_int8 | 90.36 | 72.7859 | 0.7234 | 17.5772 | completed | Similar to calib200 |
| phase2_sweep_calib1000 | `models/best_model_subject1_val.pth` | 1000 | stratified | full_int8 | 90.36 | 73.8378 | 0.7334 | 16.5253 | completed | **Best in sweep** (`best_model_int8_calib1000.tflite`) |
| phase2_sweep_calib2000 | `models/best_model_subject1_val.pth` | 2000 | stratified | full_int8 | 90.36 | 69.4605 | 0.6785 | 20.9026 | completed | Performance drop at larger calibration set |

## Sweep Summary
- Best configuration so far: **stratified calibration = 1000**
- Best INT8 metrics: **Accuracy 73.84%**, **Macro-F1 0.7334**
- Relative to previous random-200 baseline (69.70%), this is a **+4.14% absolute** gain in INT8 accuracy.

## Tracking template
Use this per experiment:
- `run_id`
- `checkpoint`
- `calibration_count`
- `sampling_method`
- `quant_mode`
- `int8_test_acc`
- `int8_test_f1_macro`
- `notes`
