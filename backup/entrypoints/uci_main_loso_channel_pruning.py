"""
Three-Stage Channel-Pruning Training for UCI-HAR Dataset LOSO:
    Stage 1: Train SeparableConvCNN without Gumbel mask and save best weights
    Stage 2: Load stage 1 weights into GumbelChannelPruningCNN and retrain
    Stage 3: Physically prune the unused channels and fine tune the compact model
"""
from pathlib import Path
import argparse
import wandb
import random
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

from lib.uci_train import train_loso_uci_three_stage_pruning_channel

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['SEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_subject_ids(path):
    # subject_*.txt stores one subject ID per sample; LOSO needs unique subject IDs.
    values = np.atleast_1d(np.loadtxt(path, dtype=int)).astype(int).tolist()
    return sorted(set(values))


def main():
    parser = argparse.ArgumentParser(
        description='Three-stage channel-pruning LOSO training on UCI-HAR dataset'
    )
    parser.add_argument('--preprocessing', type=str, choices=['fft', 'dct', 'ihw', 'no'], default='fft',
                        help='Preprocessing applied to signals: fft, dct, ihw, or no')
    parser.add_argument('--epochs_stage1', type=int, default=60,
                        help='Number of epochs for stage 1 (SeparableConvCNN)')
    parser.add_argument('--epochs_stage2', type=int, default=60,
                        help='Number of epochs for stage 2 (channel pruning)')
    parser.add_argument('--epochs_stage3', type=int, default=60,
                        help='Number of epochs for stage 3 (pruned-model fine tuning)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for both stages')
    parser.add_argument('--stage2_backbone_lr_factor', type=float, default=0.1,
                        help='Stage 2 LR multiplier for non-Gumbel parameters (channel-pruning stage)')
    parser.add_argument('--stage3_loaded_lr_factor', type=float, default=0.1,
                        help='Stage 3 LR multiplier for weights loaded from the checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--performance', action='store_true',
                        help='Enable auto-tuned high-throughput DataLoader settings')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--tau_start', type=float, default=10.0,
                        help='Initial temperature for Gumbel-Softmax in stage 2')
    parser.add_argument('--tau_end', type=float, default=1.0,
                        help='Final temperature for Gumbel-Softmax in stage 2')
    parser.add_argument('--sparsity_weight', type=float, default=0.1,
                        help='Sparsity weight for stage 2 channel pruning')
    parser.add_argument('--stage1_model_path', type=str, default=None,
                        help='Optional pretrained Stage 1 checkpoint path; if set, Stage 1 training is skipped. Supports {subject} placeholder for per-fold checkpoints.')
    parser.add_argument('--stage2_model_path', type=str, default=None,
                        help='Optional Stage 2 output checkpoint path override.')
    parser.add_argument('--stage3_model_path', type=str, default=None,
                        help='Optional Stage 3 output checkpoint path override.')
    parser.add_argument('--subject', type=int,
                        help='Optional validation subject ID for a single LOSO fold (e.g., 1).')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--wandb', type=int, default=0, help='Disable wandb logging')
    args = parser.parse_args()

    set_seed(42)

    project_root = Path(__file__).resolve().parent
    root_path = project_root / "uci-har"
    
    subject_train_path = root_path / "train" / "subject_train.txt"
    all_subjects = _load_subject_ids(subject_train_path)

    subject_test_path = root_path / "test" / "subject_test.txt"
    test_subjects = _load_subject_ids(subject_test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_log_path = project_root / 'log' / f"uci_channel_pruning_swc:{args.sparsity_weight}_pre:{args.preprocessing}.txt"
    results_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_log_path, "w") as f:
        f.write("UCI-HAR LOSO Three-Stage Channel-Pruning Training Results\n")
        f.write(f"Preprocessing: {args.preprocessing}\n")
        f.write(f"Epochs Stage 1: {args.epochs_stage1}\n")
        f.write(f"Epochs Stage 2: {args.epochs_stage2}\n")
        f.write(f"Epochs Stage 3: {args.epochs_stage3}\n")
        f.write(f"Sparsity Weight: {args.sparsity_weight}\n")
        f.write(f"Performance Mode: {args.performance}\n")
        f.write(f"Stage 2 Backbone LR Factor: {args.stage2_backbone_lr_factor}\n")
        f.write(f"Stage 3 Loaded LR Factor: {args.stage3_loaded_lr_factor}\n")
        if args.stage1_model_path is not None:
            f.write(f"Stage 1 Checkpoint Override: {args.stage1_model_path}\n")
        if args.stage2_model_path is not None:
            f.write(f"Stage 2 Checkpoint Override: {args.stage2_model_path}\n")
        if args.stage3_model_path is not None:
            f.write(f"Stage 3 Checkpoint Override: {args.stage3_model_path}\n")
        f.write("\n")

    stage_names = ["stage1", "stage2", "stage3"]
    metrics_history = {s: {"acc": [], "f1": []} for s in stage_names}

    if args.subject is None:
        fold_subjects = all_subjects
    else:
        if args.subject not in all_subjects:
            raise ValueError(
                f"Requested subject ID {args.subject} not found. "
                f"Available subject IDs: {all_subjects}"
            )
        fold_subjects = [args.subject]

    for val_subject in fold_subjects:
        val_subjects = [val_subject]
        train_subjects = [subject for subject in all_subjects if subject not in val_subjects]

        resolved_stage1_model_path = args.stage1_model_path
        if resolved_stage1_model_path is not None and "{subject}" in resolved_stage1_model_path:
            resolved_stage1_model_path = resolved_stage1_model_path.format(subject=val_subject)

        print("=" * 50)
        print(f"Fold: Val Subject {val_subjects[0]}")
        print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
        if bool(args.wandb):
            wandb_run = wandb.init(
                project="thesis-uci",
                name=f"uci-loso-channelpruning-val:{val_subject}-pre:{args.preprocessing}-swc{args.sparsity_weight}-{args.run_name}",
                config={
                    "dataset": "UCI-HAR",
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "epochs_stage1": args.epochs_stage1,
                    "epochs_stage2": args.epochs_stage2,
                    "epochs_stage3": args.epochs_stage3,
                    "lr": args.lr,
                    "stage2_backbone_lr_factor": args.stage2_backbone_lr_factor,
                    "stage3_loaded_lr_factor": args.stage3_loaded_lr_factor,
                    "batch_size": args.batch_size,
                    "performance": args.performance,
                    "preprocessing": args.preprocessing,
                    "sparsity_weight": args.sparsity_weight,
                    "training_type": "three_stage_channel_pruning",
                    "stage1_model_path": resolved_stage1_model_path,
                    "stage2_model_path": args.stage2_model_path,
                    "stage3_model_path": args.stage3_model_path,
                },
                reinit=True
            )
        else:
            wandb_run = None

        fold_model_path = (
            Path(args.stage3_model_path).expanduser()
            if args.stage3_model_path is not None
            else project_root / "models" / f"uci_best_model_three_stage_channel_subject{val_subject}_val.pth"
        )

        metrics = train_loso_uci_three_stage_pruning_channel(
            root_path=root_path,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            wandb_run=wandb_run,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            epochs_stage3=args.epochs_stage3,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            model_path=fold_model_path,
            preprocessing=args.preprocessing,
            sparsity_weight=args.sparsity_weight,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            dropout=args.dropout,
            stage2_backbone_lr_factor=args.stage2_backbone_lr_factor,
            performance=args.performance,
            stage1_model_path=resolved_stage1_model_path,
            stage3_loaded_lr_factor=args.stage3_loaded_lr_factor,
            stage2_model_path=args.stage2_model_path,
            stage3_model_path=args.stage3_model_path,
        )

        for stage in stage_names:
            metrics_history[stage]["acc"].append(metrics[stage]["test_acc"])
            metrics_history[stage]["f1"].append(metrics[stage]["test_f1_macro"])

        with open(results_log_path, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Fold Val Subject {val_subjects[0]}:\n")
            if resolved_stage1_model_path is not None:
                f.write(f"  Stage 1 Checkpoint Used: {resolved_stage1_model_path}\n")

            for stage in stage_names:
                f.write(f"\n{stage.upper()} ({metrics[stage]['model']}):\n")
                f.write(f"  Test Accuracy: {metrics[stage]['test_acc']:.2f}%\n")
                f.write(f"  Test F1 Macro: {metrics[stage]['test_f1_macro']:.4f}\n")

            if "param_reduction_pct" in metrics["stage3"]:
                f.write("\n  Stage 3 Model Size Reduction:\n")
                f.write(f"    Dense Params: {metrics['stage3']['dense_param_count']:,}\n")
                f.write(f"    Pruned Params: {metrics['stage3']['pruned_param_count']:,}\n")
                f.write(f"    Reduction: {metrics['stage3']['param_reduction_pct']:.2f}%\n")

            f.write(
                f"\n  Improvement Stage2 - Stage1: "
                f"{metrics['stage2']['test_acc'] - metrics['stage1']['test_acc']:.2f}%\n"
            )
            f.write(
                f"  Improvement Stage3 - Stage2: "
                f"{metrics['stage3']['test_acc'] - metrics['stage2']['test_acc']:.2f}%\n"
            )
            f.write(
                f"  Improvement Stage3 - Stage1: "
                f"{metrics['stage3']['test_acc'] - metrics['stage1']['test_acc']:.2f}%\n"
            )

            final_mask = metrics["stage2"].get("final_mask")
            if isinstance(final_mask, dict):
                f.write("  Final Masks:\n")
                for block_name, block_mask in final_mask.items():
                    f.write(f"    {block_name}: {block_mask.tolist()}\n")

            pruning_stats = metrics["stage2"].get("pruning_stats")
            if pruning_stats is not None:
                f.write("  Pruning Stats:\n")
                for k, v in pruning_stats.items():
                    f.write(f"    {k}: {v:.2f}\n")

        if wandb_run is not None:
            wandb_run.finish()

    print("=" * 50)
    print("UCI-HAR LOSO Three-Stage Channel-Pruning Cross-Validation Results")
    print("=" * 50)

    with open(results_log_path, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Overall UCI-HAR LOSO Three-Stage Channel-Pruning Results\n")
        f.write("=" * 50 + "\n")

    for stage in stage_names:
        mean_acc = float(np.mean(metrics_history[stage]["acc"]))
        std_acc = float(np.std(metrics_history[stage]["acc"]))
        mean_f1 = float(np.mean(metrics_history[stage]["f1"]))
        std_f1 = float(np.std(metrics_history[stage]["f1"]))

        print(f"\n{stage.upper()}:")
        print(f"  Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"  Test F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}")

        with open(results_log_path, "a") as f:
            f.write(f"\n{stage.upper()}:\n")
            f.write(f"  Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
            f.write(f"  Test F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}\n")

    print("\n" + "=" * 50)
    print("Results log saved to:", results_log_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
