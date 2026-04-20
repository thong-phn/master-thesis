"""
TFLite Post-Training Quantization for WEAR LOSO Models (Stage 1, 3, 5).

Pipeline:  PyTorch .pth  →  ONNX  →  TF SavedModel (via onnx2tf)  →  TFLite (PTQ)
Configs:   W8A16_FLOAT_IO  |  W8A16_INT_IO  |  W8A8_INT_IO

Usage:
    python wear_quantize_loso_tflite_ptq.py --stage 1 --subjects '0'
    python wear_quantize_loso_tflite_ptq.py --stage 3 --subjects '0,1,2'
    python wear_quantize_loso_tflite_ptq.py --stage 5
    python wear_quantize_loso_tflite_ptq.py --stage 3 --mask-log-file log/custom_results.txt
"""

import os
import io
import argparse
import re
import ast
import subprocess
import shutil
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import wandb

# ── project imports ──────────────────────────────────────────────────────────
from lib.model import SeparableConvCNN, PrunedSeparableConvCNN
from lib.wear_train import WEAR_Dataset, SlicedWEARDataset

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _str2bool(value):
    """Parse common CLI boolean strings so '--wandb False' works as expected."""
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


def _load_subject_ids(path):
    return sorted(np.unique(np.atleast_1d(np.loadtxt(path, dtype=int)).astype(int).tolist()))


def _parse_subject_selection(subjects_arg: str | None):
    if subjects_arg is None:
        return None
    tokens = [t for t in re.split(r"[\s,\.]+", subjects_arg.strip()) if t]
    if not tokens:
        return None
    return sorted({int(t) for t in tokens})


# ── Stage-3 hard-bin mask extraction ─────────────────────────────────────────
def _get_hard_bin_mask(log_file: Path, val_subject: int):
    """Parse the results log to get the binary hard-bin mask for a given fold."""
    if not log_file.exists():
        print(f"[warn] log file {log_file} not found – cannot extract mask for subject {val_subject}")
        return None
    content = log_file.read_text()
    pattern = rf"Fold Val Subject {val_subject}:\n(.*?)(?=\n=|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    fold_text = match.group(1)
    mask_match = re.search(r"Hard Bin Mask:\s*(\[.*?\])", fold_text)
    if not mask_match:
        return None
    return np.array(ast.literal_eval(mask_match.group(1)), dtype=np.float32)


# ── Model loading ────────────────────────────────────────────────────────────
def _load_pytorch_model(stage: int, model_path: Path, device: torch.device):
    """Instantiate and load the correct architecture from a .pth checkpoint."""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    num_channels = state_dict["sep_conv1.depthwise.weight"].shape[0]
    num_classes  = state_dict["fc2.weight"].shape[0]

    if stage in (1, 3):
        model = SeparableConvCNN(num_classes=num_classes, num_channels=num_channels)
    elif stage == 5:
        b2 = state_dict["sep_conv2.pointwise.weight"].shape[0]
        b3 = state_dict["sep_conv3.pointwise.weight"].shape[0]
        b4 = state_dict["sep_conv4.pointwise.weight"].shape[0]
        model = PrunedSeparableConvCNN(
            num_classes=num_classes, num_channels=num_channels,
            block2_channels=b2, block3_channels=b3, block4_channels=b4,
        )
    else:
        raise ValueError("Only stages 1, 3, 5 are supported.")

    model.load_state_dict(state_dict)
    model.eval()
    return model, num_channels


def _count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


# ── ONNX → TF SavedModel ────────────────────────────────────────────────────
def _export_onnx_and_convert(pt_model, onnx_path: Path, saved_model_dir: Path,
                              input_shape: tuple):
    """Export PyTorch → ONNX → TF SavedModel (via onnx2tf CLI)."""
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        pt_model, dummy, str(onnx_path),
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
    )
    print(f"  ONNX exported → {onnx_path}")

    cmd = ["onnx2tf", "-i", str(onnx_path), "-o", str(saved_model_dir), "-osd"]
    print(f"  onnx2tf: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  TF SavedModel → {saved_model_dir}")


# ── Representative dataset generator ────────────────────────────────────────
def _make_representative_gen(dataset, n_samples: int = 256):
    """
    Build a stratified calibration generator that yields float32 numpy arrays
    shaped (1, channels, freq_bins) – one sample at a time.
    ``dataset`` can be a WEAR_Dataset *or* a SlicedWEARDataset.
    """
    total = len(dataset)
    if total == 0:
        raise RuntimeError("Empty dataset – cannot build representative set")

    # Collect all labels for stratification
    labels = []
    for i in range(total):
        _, y = dataset[i]
        labels.append(int(y.item()) if isinstance(y, torch.Tensor) else int(y))
    labels = np.array(labels)

    n_samples = min(n_samples, total)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
    indices, _ = next(sss.split(np.zeros(total), labels))

    def gen():
        for idx in indices:
            x, _ = dataset[idx]
            # onnx2tf converts to NHWC: (1, freq_bins, channels)
            arr = x.numpy().astype(np.float32)
            arr = np.transpose(arr)  # (C, F) → (F, C)
            yield [arr[np.newaxis, ...]]

    return gen


# ── TFLite conversion ────────────────────────────────────────────────────────
PTQ_CONFIGS = ["W8A16_INT_IO"]


def _parse_macs_from_log(log_text: str):
    """Extract estimated MACs from TFLite converter log output."""
    # Pattern: "Estimated count of arithmetic ops: 0.574 M  ops, equivalently 0.287 M  MACs"
    match = re.search(
        r"Estimated count of arithmetic ops:\s+([\d.]+)\s+([KMG]?)\s*ops.*?equivalently\s+([\d.]+)\s+([KMG]?)\s*MACs",
        log_text,
    )
    if not match:
        return None, None
    multipliers = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}
    ops  = float(match.group(1)) * multipliers.get(match.group(2), 1)
    macs = float(match.group(3)) * multipliers.get(match.group(4), 1)
    return ops, macs


def _convert_to_tflite(saved_model_dir: str, ptq_config: str, rep_gen):
    """Convert SavedModel to TFLite with given PTQ config.
    Returns (tflite_bytes, ops, macs) or (None, None, None) on failure."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen

    if ptq_config == "W8A16_FLOAT_IO":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]

    elif ptq_config == "W8A16_INT_IO":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16

    elif ptq_config == "W8A8_INT_IO":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # Capture C++ stderr to parse MACs/OPs (TFLite logs these via native code)
    import sys
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, stderr_fd)

    try:
        tflite_model = converter.convert()
    except Exception as exc:
        os.dup2(saved_stderr, stderr_fd)
        os.close(pipe_w)
        os.close(pipe_r)
        os.close(saved_stderr)
        print(f"  [ERROR] TFLite conversion ({ptq_config}): {exc}")
        return None, None, None

    # Restore stderr and read captured output
    os.dup2(saved_stderr, stderr_fd)
    os.close(pipe_w)
    os.close(saved_stderr)

    captured = b""
    while True:
        chunk = os.read(pipe_r, 4096)
        if not chunk:
            break
        captured += chunk
    os.close(pipe_r)

    log_text = captured.decode("utf-8", errors="replace")
    ops, macs = _parse_macs_from_log(log_text)
    return tflite_model, ops, macs


# ── TFLite evaluation ────────────────────────────────────────────────────────
def _evaluate_tflite(tflite_model: bytes, dataloader: DataLoader):
    """Run inference on every sample; return (accuracy%, f1_macro)."""
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()

    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    inp_idx = inp_det["index"]
    out_idx = out_det["index"]

    is_int_in  = inp_det["dtype"] in (np.int8, np.int16)
    is_int_out = out_det["dtype"] in (np.int8, np.int16)

    in_scale, in_zp   = inp_det["quantization"]
    out_scale, out_zp = out_det["quantization"]
    if in_scale == 0.0:
        in_scale = 1.0
    if out_scale == 0.0:
        out_scale = 1.0

    all_preds, all_targets = [], []

    for x_batch, y_batch in dataloader:
        x_np = x_batch.numpy().astype(np.float32)
        y_np = y_batch.numpy()

        for i in range(len(x_np)):
            sample = x_np[i]                           # (C, F)
            # onnx2tf converts to NHWC: (F, C)
            sample = np.transpose(sample)              # (C, F) → (F, C)
            sample = sample[np.newaxis, ...]            # (1, F, C)

            if is_int_in:
                sample = np.round(sample / in_scale + in_zp).astype(inp_det["dtype"])

            interp.set_tensor(inp_idx, sample)
            interp.invoke()
            out = interp.get_tensor(out_idx)[0]

            if is_int_out:
                out = (out.astype(np.float32) - out_zp) * out_scale

            all_preds.append(int(np.argmax(out)))
            all_targets.append(int(y_np[i]))

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    acc = float(np.mean(preds == targets)) * 100.0
    f1  = float(f1_score(targets, preds, average="macro"))
    return acc, f1


# Main

def main():
    parser = argparse.ArgumentParser(description="TFLite PTQ evaluation for WEAR models")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 3, 5])
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated validation subjects, e.g. '0' or '0,1,2'. "
                             "If omitted, runs all subjects.")
    parser.add_argument("--mask-log-file", type=Path, default=None,
                        help="Path to the log txt file used to extract the stage-3/5 hard-bin mask. "
                             "Defaults to the matching file under log/.")
    parser.add_argument("--preprocessing", type=str, default="fft",
                        choices=["fft", "dct", "ihw", "no"])
    parser.add_argument(
        "--wandb",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable or disable W&B logging (e.g., --wandb False to disable).",
    )
    parser.add_argument("--log_name", type=str, default=None)
    parser.add_argument("--tflite-output-path", type=Path, default=None)
    args = parser.parse_args()

    set_seed(42)

    project_root = Path(__file__).resolve().parent
    root_path    = project_root / "wear"

    all_subjects = _load_subject_ids(root_path / "train" / "subject_train.txt")
    requested    = _parse_subject_selection(args.subjects)
    fold_subjects = requested if requested is not None else all_subjects

    mask_log_file = args.mask_log_file or (project_root / "log" / f"wear_loso_five_stage_results_{args.preprocessing}.txt")

    results_path = project_root / "log" / f"wear_ptq_stage{args.stage}_{args.log_name}.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.wandb:
        wandb.login()
        wandb.init(
            project="thesis",
            name=f"tflite-ptq-stage{args.stage}",
            job_type="upload",
            config=vars(args),
        )

    with open(results_path, "w") as f:
        f.write(f"TFLite PTQ Results – Stage {args.stage}\n{'=' * 50}\n\n")

    device = torch.device("cpu")          # export & conversion run on CPU
    metrics_history = {c: {"acc": [], "f1": []} for c in PTQ_CONFIGS}

    # ── fold loop ────────────────────────────────────────────────────────────
    for val_subject in fold_subjects:
        print(f"\n{'=' * 60}")
        print(f"  Fold – validation subject {val_subject}")
        print(f"{'=' * 60}")

        # checkpoint path
        if args.stage == 1:
            ckpt = project_root / "models" / "wear" / "stage1" / f"{args.preprocessing}"/f"wear_best_model_subject{val_subject}_val.pth"
        elif args.stage == 3:
            ckpt = project_root / "models" / f"wear_best_model_three_stage_subject{val_subject}_val_stage3_pruned_input.pth"
        elif args.stage == 5:
            ckpt = project_root / "models" / f"wear_best_model_three_stage_channel_subject{val_subject}_val_stage3_pruned_channel.pth"
        else:
            ckpt = project_root / "models" / f"wear_best_model_five_stage_subject{val_subject}_val_stage5_compact.pth"

        if not ckpt.exists():
            print(f"  [skip] checkpoint not found: {ckpt}")
            continue

        pt_model, in_channels = _load_pytorch_model(args.stage, ckpt, device)

        # ── datasets ─────────────────────────────────────────────────────────
        train_subjects = [s for s in all_subjects if s != val_subject]
        train_ds = WEAR_Dataset(root_path, split="train", subject_ids=train_subjects,
                                preprocessing=args.preprocessing)
        test_ds  = WEAR_Dataset(root_path, split="test", subject_ids=None,
                                preprocessing=args.preprocessing)

        # Stage 3 & 5: physically slice bins using the hard-bin mask
        if args.stage in (3, 5):
            mask = _get_hard_bin_mask(mask_log_file, val_subject)
            if mask is not None:
                keep_idx = torch.from_numpy(np.where(mask > 0.5)[0]).long()
                train_ds = SlicedWEARDataset(train_ds, keep_idx)
                test_ds  = SlicedWEARDataset(test_ds, keep_idx)
                print(f"  Stage {args.stage}: sliced to {len(keep_idx)} bins using hard-bin mask")
            else:
                print("  [warn] No hard-bin mask found; using unsliced input")

        # Determine input shape from the first sample
        sample_x, _ = train_ds[0]
        freq_bins = sample_x.shape[-1]
        input_shape = (1, in_channels, freq_bins)
        n_params = _count_parameters(pt_model)
        print(f"  Input shape for export: {input_shape}")
        print(f"  PyTorch model parameters: {n_params:,}")

        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        # ── export & convert once per fold ───────────────────────────────────
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            onnx_path      = tmpdir / "model.onnx"
            saved_model_dir = tmpdir / "saved_model"

            _export_onnx_and_convert(pt_model, onnx_path, saved_model_dir, input_shape)

            rep_gen = _make_representative_gen(train_ds, n_samples=256)

            # Log model info once per fold
            with open(results_path, "a") as f:
                f.write(f"\nFold {val_subject} | Params: {n_params:,}\n")

            # ── quantize & eval each config ──────────────────────────────────
            for cfg in PTQ_CONFIGS:
                print(f"\n  ── {cfg} ──")
                tflite_model, ops, macs = _convert_to_tflite(str(saved_model_dir), cfg, rep_gen)
                if tflite_model is None:
                    continue

                tflite_size_kb = len(tflite_model) / 1024

                # Optionally save .tflite for inspection
                tflite_out = tmpdir / f"model_{cfg}.tflite"
                tflite_out.write_bytes(tflite_model)
                if args.tflite_output_path is not None:
                    dest_path = args.tflite_output_path / f"subject{val_subject}_{cfg}.tflite"
                    shutil.copy(tflite_out, dest_path)
                    print(f"  TFLite model path: {dest_path}")

                ops_str  = f"{ops/1e6:.3f} M" if ops is not None else "N/A"
                macs_str = f"{macs/1e6:.3f} M" if macs is not None else "N/A"
                print(f"  TFLite size: {tflite_size_kb:.1f} KB  |  OPs: {ops_str}  |  MACs: {macs_str}")

                acc, f1 = _evaluate_tflite(tflite_model, test_loader)
                print(f"  Result: Acc {acc:.2f}%  F1 {f1:.4f}")

                metrics_history[cfg]["acc"].append(acc)
                metrics_history[cfg]["f1"].append(f1)

                with open(results_path, "a") as f:
                    f.write(
                        f"  {cfg} | Acc: {acc:.2f}% | F1: {f1:.4f}"
                        f" | Size: {tflite_size_kb:.1f} KB"
                        f" | OPs: {ops_str} | MACs: {macs_str}\n"
                    )

    # ── summary ──────────────────────────────────────────────────────────────
    with open(results_path, "a") as f:
        f.write(f"\n{'=' * 50}\nOVERALL AVERAGES\n{'=' * 50}\n")
        for cfg in PTQ_CONFIGS:
            accs = metrics_history[cfg]["acc"]
            f1s  = metrics_history[cfg]["f1"]
            if accs:
                f.write(f"{cfg}  |  Acc: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%  "
                        f"|  F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")

    # ── WandB artifact ───────────────────────────────────────────────────────
    if args.wandb:
        artifact = wandb.Artifact(
            name=f"wear-tflite-ptq-stage{args.stage}", type="results-log",
        )
        artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)
        print(f"\nResults saved → {results_path}  (uploaded to W&B)")
        wandb.finish()
    else:
        print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
