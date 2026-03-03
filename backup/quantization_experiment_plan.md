# PT2E INT8 Accuracy Recovery Plan (UCI-HAR)

## Goal
Reduce post-training quantization accuracy drop from **12.42%** to **<= 1.0%** vs FP32 (stretch: **<= 0.5%**).

## Current Baseline
- FP32 accuracy: **90.36%**
- INT8 accuracy: **77.94%**
- Drop: **12.42%**
- Script: `post-training-quantization.py`

## Success Criteria
- Primary: `accuracy_drop <= 1.0`
- Stretch: `accuracy_drop <= 0.5`
- Must remain reproducible with fixed seed and saved config.

## Experiment Strategy

### Phase 1: Calibration quality (lowest cost, highest expected gain)
1. Sweep calibration batch count (`max_calibration_batches`) aggressively.
2. Keep checkpoint/model fixed; only change calibration amount.
3. Use same evaluation path for fair comparison.

### Phase 2: Quantization config sensitivity
1. Compare per-channel vs per-tensor weights.
2. Compare activation quant ranges (symmetric and narrower ranges).
3. Keep calibration size at best value from Phase 1.

### Phase 3: Data representativeness
1. Calibrate with full train subjects vs selected subset.
2. Try shuffled vs deterministic calibration ordering.
3. Validate robustness across 2-3 random seeds.

### Phase 4: Model-level fallbacks (if still >1%)
1. Layerwise sensitivity scan (identify most fragile blocks).
2. Keep sensitive layer(s) in FP32 if runtime allows.
3. Re-evaluate mixed-precision compromise.

## Tracking Table
| Run ID | Date | Command/Config | FP32 Acc (%) | INT8 Acc (%) | Drop (%) | Status | Notes |
|---|---|---|---:|---:|---:|---|---|
| R0 | 2026-02-27 | `--max-calibration-batches 1 --batch-size 128` | 90.36 | 77.94 | 12.42 | Done | Baseline (from current script output) |
| R1 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128` | 90.36 | 83.41 | 6.96 | Done | Better than baseline |
| R2 | 2026-02-27 | `--max-calibration-batches 5 --batch-size 128` | 90.36 | 77.03 | 13.34 | Done | Worse |
| R3 | 2026-02-27 | `--max-calibration-batches 10 --batch-size 128` | 90.36 | 78.69 | 11.67 | Done | Worse |
| R4 | 2026-02-27 | `--max-calibration-batches 20 --batch-size 128` | 90.36 | 74.08 | 16.29 | Done | Worse |
| R5 | 2026-02-27 | `--max-calibration-batches 50 --batch-size 128` | 90.36 | 73.94 | 16.42 | Done | Worst in this sweep |
| R6 | 2026-02-27 | `--max-calibration-batches 100 --batch-size 128` | 90.36 | 82.22 | 8.14 | Done | Better than baseline, not best |
| R7 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --per-channel` | 90.36 | 83.41 | 6.96 | Done | Config baseline |
| R8 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --no-per-channel` | 90.36 | 82.73 | 7.63 | Done | Per-tensor slightly worse |
| R9 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --seed 7 --per-channel` | 90.36 | 74.79 | 15.58 | Done | With shuffled calibration, seed-sensitive |
| R10 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --act-qmin -127 --act-qmax 127` | 90.36 | 83.03 | 7.33 | Done | Slightly worse than R7 |
| R11 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --no-calibration-shuffle` | 90.36 | 85.61 | 4.75 | Done | **Best so far** |
| R12 | 2026-02-27 | `--max-calibration-batches 1 --batch-size 128 --no-calibration-shuffle` | 90.36 | 83.95 | 6.41 | Done | Better than shuffled |
| R13 | 2026-02-27 | `--max-calibration-batches 5 --batch-size 128 --no-calibration-shuffle` | 90.36 | 77.74 | 12.62 | Done | Worse |
| R14 | 2026-02-27 | `--max-calibration-batches 10 --batch-size 128 --no-calibration-shuffle` | 90.36 | 80.69 | 9.67 | Done | Worse than R11 |
| R15 | 2026-02-27 | `--max-calibration-batches 20 --batch-size 128 --no-calibration-shuffle` | 90.36 | 81.98 | 8.38 | Done | Worse than R11 |
| R16 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --no-calibration-shuffle --seed 1` | 90.36 | 85.61 | 4.75 | Done | Stable |
| R17 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --no-calibration-shuffle --seed 7` | 90.36 | 85.61 | 4.75 | Done | Stable |
| R18 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --no-calibration-shuffle --seed 13` | 90.36 | 85.61 | 4.75 | Done | Stable |
| R19 | 2026-02-27 | `--max-calibration-batches 1000 --batch-size 128 --no-calibration-shuffle --calibration-samples 64 --balanced-calibration` | 90.36 | 79.10 | 11.27 | Done | P1 balanced subset |
| R20 | 2026-02-27 | `--max-calibration-batches 1000 --batch-size 128 --no-calibration-shuffle --calibration-samples 128 --balanced-calibration` | 90.36 | 78.22 | 12.15 | Done | P1 balanced subset |
| R21 | 2026-02-27 | `--max-calibration-batches 1000 --batch-size 128 --no-calibration-shuffle --calibration-samples 256 --balanced-calibration` | 90.36 | 77.98 | 12.39 | Done | P1 balanced subset |
| R22 | 2026-02-27 | `--max-calibration-batches 1000 --batch-size 128 --no-calibration-shuffle --calibration-samples 512 --balanced-calibration` | 90.36 | 82.66 | 7.70 | Done | P1 balanced subset (best in balanced sweep) |
| R23 | 2026-02-27 | `--max-calibration-batches 1000 --batch-size 128 --no-calibration-shuffle --calibration-samples 64 --no-balanced-calibration` | 90.36 | 80.12 | 10.25 | Done | P1 random subset |
| R24 | 2026-02-27 | `--max-calibration-batches 1000 --batch-size 128 --no-calibration-shuffle --calibration-samples 128 --no-balanced-calibration` | 90.36 | 80.45 | 9.91 | Done | P1 random subset |
| R25 | 2026-02-27 | `--max-calibration-batches 1000 --batch-size 128 --no-calibration-shuffle --calibration-samples 256 --no-balanced-calibration` | 90.36 | 80.22 | 10.15 | Done | P1 random subset |
| R26 | 2026-02-27 | `--max-calibration-batches 1000 --batch-size 128 --no-calibration-shuffle --calibration-samples 512 --no-balanced-calibration` | 90.36 | 79.27 | 11.10 | Done | P1 random subset |
| R27 | 2026-02-27 | `--max-calibration-batches 2 --batch-size 128 --no-calibration-shuffle --per-channel --minmax-normalize` | 59.59 | 67.19 | -7.60 | Done | P2 min-max normalization changed input distribution; FP32 collapsed |

## Notes / Learnings
- Biggest gain came from **deterministic calibration order** (`--no-calibration-shuffle`), improving drop from 6.96% (R7) to **4.75%** (R11).
- Calibration batch count is non-monotonic; more calibration data did not consistently improve PTQ for this model.
- Per-channel quantization remains better than per-tensor in current tests.
- With shuffled calibration, results can be seed-sensitive; deterministic calibration removes this instability.
- P1 class-balanced calibration subsets (64/128/256/512) did not beat R11; best balanced result was 7.70% drop (R22).
- P1 random subsets also underperformed R11; best random result was 9.91% drop (R24).
- For this checkpoint/model, using full train calibration pool with low batch count and deterministic order still works best.
- P2 min-max normalization (R27) is **not comparable** to prior runs because it changes model input distribution; FP32 accuracy dropped sharply (90.36% -> 59.59%).
- Conclusion: do not enable min-max normalization for this pretrained checkpoint unless model is retrained or normalization is built into training pipeline.
- `.pte` export may fail in this environment due backend/op support mismatch; this does not block PT2E accuracy experiments.

## Current Best Config
- `--max-calibration-batches 2 --batch-size 128 --per-channel --no-calibration-shuffle`
- FP32: **90.36%**
- INT8: **85.61%**
- Drop: **4.75%**

## Gap to Goal
- Target is <= 1.0% drop, current best is 4.75% drop.
- Remaining gap: **3.75%** (or **4.25%** to stretch target 0.5%).

## Next Runs (queued)
- P2: **Static normalization before quantization**: clamp/log-scale FFT magnitudes before observer collection and compare drop.
- P3: **Module sensitivity check**: keep first block or classifier head in FP32 (mixed precision) to estimate accuracy ceiling quickly.
- P4: **QAT fallback** (likely required for <=1%): short QAT fine-tune (3-10 epochs) initialized from FP32 checkpoint with PT2E/QAT flow.
- P5: **Per-subject calibration**: calibrate on subject distribution closer to test-set subjects and compare generalization.
- P6: **TorchAO/ExecuTorch quantizer migration**: retry with newer quantizer path to rule out PT2E deprecation-path regression.
