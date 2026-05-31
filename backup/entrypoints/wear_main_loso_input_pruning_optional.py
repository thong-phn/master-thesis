"""
Two-Stage Training for WEAR Dataset LOSO:
    Stage 1: Train SeparableConvCNN without Gumbel mask and save best weights
    Stage 2: Load stage 1 weights into GumbelMaskSeparableConvCNN and train with Gumbel mask
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

from lib.wear_train import train_loso_wear_two_stage_input_pruning


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['SEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description='Two-stage LOSO training on WEAR dataset (input pruning)'
    )
    parser.add_argument('--preprocessing', type=str, choices=['fft', 'dct', 'ihw', 'no'], default='fft',
                        help='Preprocessing applied to signals: fft, dct, ihw, or no')
    parser.add_argument('--sparsity_weight_bin', '--sparsity_weight', dest='sparsity_weight_bin', type=float, default=0.1,
                        help='Sparsity weight for stage 2 input-bin pruning')
    parser.add_argument('--epochs_stage1', type=int, default=60,
                        help='Number of epochs for stage 1 (SeparableConvCNN)')
    parser.add_argument('--epochs_stage2', type=int, default=60,
                        help='Number of epochs for stage 2 (input-bin Gumbel pruning)')
    parser.add_argument('--log_every_n_epochs', type=int, default=5,
                        help='Log/print stage training metrics every N epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Base learning rate')
    parser.add_argument('--stage2_backbone_lr_factor', type=float, default=0.1,
                        help='Stage 2 LR multiplier for non-Gumbel parameters (input-bin pruning stage)')
    parser.add_argument('--performance', action='store_true',
                        help='Enable auto-tuned high-throughput DataLoader settings')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--tau_start', type=float, default=10.0,
                        help='Initial temperature for Gumbel-Softmax in stage 2')
    parser.add_argument('--tau_end', type=float, default=1.0,
                        help='Final temperature for Gumbel-Softmax in stage 2')
    parser.add_argument('--stage1_model_path', type=str, default=None,
                        help='Optional pretrained Stage 1 checkpoint path; if set, Stage 1 training is skipped. Supports {subject} placeholder for per-fold checkpoints.')
    parser.add_argument('--stage2_model_path', type=str, default=None,
                        help='Optional path to a pretrained Stage 2 checkpoint. If provided, Stage 2 training is skipped.')
    parser.add_argument('--single_subject_only', action='store_true',
                        help='Run only one LOSO fold (subject 0 if available).')
    parser.add_argument('--wandb_run_name', type=str)
    
    args = parser.parse_args()

    set_seed(42)

    # Detect if running on Kaggle and set appropriate path
    if os.path.exists('/kaggle/input'):
        root_path = Path('/kaggle/input/datasets/thongp/wearthesis/wear')
        project_root = Path('/kaggle/working')
    else:
        project_root = Path(__file__).resolve().parent
        root_path = project_root / "wear"

    subject_train_path = root_path / "train" / "subject_train.txt"
    all_subjects = sorted(np.atleast_1d(np.loadtxt(subject_train_path, dtype=int)).astype(int).tolist())

    subject_test_path = root_path / "test" / "subject_test.txt"
    test_subjects = sorted(np.atleast_1d(np.loadtxt(subject_test_path, dtype=int)).astype(int).tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_log_path = project_root / 'log' / f"wear_loso_two_stage_input_pruning_results_{args.preprocessing}.txt"
    results_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_log_path, "w") as f:
        f.write("WEAR LOSO Two-Stage Training Results\n")
        f.write(f"WANDB run name: {args.wandb_run_name}\n")
        f.write(f"Preprocessing: {args.preprocessing}\n")
        f.write(f"Sparsity Weight Bin: {args.sparsity_weight_bin}\n")
        f.write(f"Epochs Stage 1: {args.epochs_stage1}\n")
        f.write(f"Epochs Stage 2: {args.epochs_stage2}\n")
        f.write(f"Log Every N Epochs: {args.log_every_n_epochs}\n")
        f.write(f"Stage 2 Backbone LR Factor: {args.stage2_backbone_lr_factor}\n")
        f.write(f"Performance Mode: {args.performance}\n")
        if args.stage1_model_path is not None:
            f.write(f"Stage 1 Checkpoint Override: {args.stage1_model_path}\n")
        if args.stage2_model_path is not None:
            f.write(f"Stage 2 Checkpoint Override: {args.stage2_model_path}\n")
        f.write("\n")

    test_accs_stage1 = []
    test_f1s_stage1 = []
    test_accs_stage2 = []
    test_f1s_stage2 = []
    stage2_hard_bin_masks = []
    per_fold_final_summary_logs = []

    stage1_label = 'SeparableConvCNN'
    stage2_label = 'GumbelMaskSeparableConvCNN'

    fold_subjects = [all_subjects[0]] if args.single_subject_only else all_subjects

    for val_subject in fold_subjects:
        val_subjects = [val_subject]
        train_subjects = [subject for subject in all_subjects if subject not in val_subjects]

        resolved_stage1_model_path = args.stage1_model_path
        if resolved_stage1_model_path is not None and "{subject}" in resolved_stage1_model_path:
            resolved_stage1_model_path = resolved_stage1_model_path.format(subject=val_subject)

        print("=" * 50)
        print(f"Fold: Val Subject {val_subjects[0]}")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

        # Tracking init
        wandb_run = wandb.init(
            project="thesis-analysis",
            name=f"wear-loso-{args.wandb_run_name}-{val_subject}-{args.preprocessing}",
            config={
                "dataset": "WEAR",
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "epochs_stage1": args.epochs_stage1,
                "epochs_stage2": args.epochs_stage2,
                "log_every_n_epochs": args.log_every_n_epochs,
                "lr": args.lr,
                "stage2_backbone_lr_factor": args.stage2_backbone_lr_factor,
                "performance": args.performance,
                "batch_size": args.batch_size,
                "preprocessing": args.preprocessing,
                "sparsity_weight_bin": args.sparsity_weight_bin,
                "training_type": "two_stage_input_pruning",
                "stage1_model_path": resolved_stage1_model_path,
                "stage2_model_path": args.stage2_model_path,
            },
            reinit=True
        )

        metrics = train_loso_wear_two_stage_input_pruning(
            root_path=root_path,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            wandb_run=wandb_run,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            log_every_n_epochs=args.log_every_n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            model_path=project_root / "models" / f"wear_best_model_two_stage_input_pruning_subject{val_subject}_val.pth",
            preprocessing=args.preprocessing,
            sparsity_weight_bin=args.sparsity_weight_bin,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            dropout=args.dropout,
            stage2_backbone_lr_factor=args.stage2_backbone_lr_factor,
            performance=args.performance,
            stage1_model_path=resolved_stage1_model_path,
            stage2_model_path=args.stage2_model_path,
        )

        # Extract metrics from both stages
        test_acc_stage1 = metrics["stage1"]["test_acc"]
        test_f1_stage1 = metrics["stage1"]["test_f1_macro"]
        
        test_acc_stage2 = metrics["stage2"]["test_acc"]
        test_f1_stage2 = metrics["stage2"]["test_f1_macro"]

        hard_bin_mask = metrics["stage2"].get("hard_bin_mask", None)
        final_summary_log_path = metrics.get("final_summary_log_path")

        if final_summary_log_path:
            per_fold_final_summary_logs.append(str(final_summary_log_path))

        if hard_bin_mask is not None:
            stage2_hard_bin_masks.append({
                "val_subject": int(val_subject),
                "hard_bin_mask": hard_bin_mask.tolist() if hasattr(hard_bin_mask, "tolist") else hard_bin_mask,
            })

        test_accs_stage1.append(test_acc_stage1)
        test_f1s_stage1.append(test_f1_stage1)
        test_accs_stage2.append(test_acc_stage2)
        test_f1s_stage2.append(test_f1_stage2)

        # Log results
        with open(results_log_path, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Fold Val Subject {val_subjects[0]}:\n")
            if resolved_stage1_model_path is not None:
                f.write(f"  Stage 1 Checkpoint Used: {resolved_stage1_model_path}\n")
            f.write(f"\nStage 1 ({stage1_label}):\n")
            f.write(f"  Test Accuracy: {test_acc_stage1:.2f}%\n")
            f.write(f"  Test F1 Macro: {test_f1_stage1:.4f}\n")
            f.write(f"\nStage 2 ({stage2_label}):\n")
            f.write(f"  Test Accuracy: {test_acc_stage2:.2f}%\n")
            f.write(f"  Test F1 Macro: {test_f1_stage2:.4f}\n")

            f.write(f"\n  Improvement Stage2 - Stage1: {test_acc_stage2 - test_acc_stage1:.2f}%\n")

            if hard_bin_mask is not None:
                f.write(f"  Hard Bin Mask: {hard_bin_mask.tolist()}\n")
            if final_summary_log_path is not None:
                f.write(f"  Final Summary Log: {final_summary_log_path}\n")

        if wandb_run is not None:
            wandb_run.finish()

    # Compute overall statistics
    mean_acc_stage1 = np.mean(test_accs_stage1)
    std_acc_stage1 = np.std(test_accs_stage1)
    mean_f1_stage1 = np.mean(test_f1s_stage1)
    std_f1_stage1 = np.std(test_f1s_stage1)
    
    mean_acc_stage2 = np.mean(test_accs_stage2)
    std_acc_stage2 = np.std(test_accs_stage2)
    mean_f1_stage2 = np.mean(test_f1s_stage2)
    std_f1_stage2 = np.std(test_f1s_stage2)

    print("=" * 50)
    print("WEAR LOSO Two-Stage Cross-Validation Results")
    print("=" * 50)
    print(f"\nStage 1 ({stage1_label}):")
    print(f"  Test Accuracy: {mean_acc_stage1:.2f}% ± {std_acc_stage1:.2f}%")
    print(f"  Test F1 Macro: {mean_f1_stage1:.4f} ± {std_f1_stage1:.4f}")
    print(f"\nStage 2 ({stage2_label}):")
    print(f"  Test Accuracy: {mean_acc_stage2:.2f}% ± {std_acc_stage2:.2f}%")
    print(f"  Test F1 Macro: {mean_f1_stage2:.4f} ± {std_f1_stage2:.4f}")
    print(f"\nImprovement (Stage 2 - Stage 1):")
    print(f"  Accuracy: {mean_acc_stage2 - mean_acc_stage1:.2f}%")
    print(f"  F1 Macro: {mean_f1_stage2 - mean_f1_stage1:.4f}")

    # Write overall results
    with open(results_log_path, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Overall WEAR LOSO Two-Stage Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nStage 1 ({stage1_label}):\n")
        f.write(f"  Test Accuracy: {mean_acc_stage1:.2f}% ± {std_acc_stage1:.2f}%\n")
        f.write(f"  Test F1 Macro: {mean_f1_stage1:.4f} ± {std_f1_stage1:.4f}\n")
        f.write(f"\nStage 2 ({stage2_label}):\n")
        f.write(f"  Test Accuracy: {mean_acc_stage2:.2f}% ± {std_acc_stage2:.2f}%\n")
        f.write(f"  Test F1 Macro: {mean_f1_stage2:.4f} ± {std_f1_stage2:.4f}\n")
        f.write(f"\nImprovement (Stage 2 - Stage 1):\n")
        f.write(f"  Accuracy: {mean_acc_stage2 - mean_acc_stage1:.2f}%\n")
        f.write(f"  F1 Macro: {mean_f1_stage2 - mean_f1_stage1:.4f}\n")

    final_summary_metrics = {
        "summary/stage1/test_acc_mean": float(mean_acc_stage1),
        "summary/stage1/test_acc_std": float(std_acc_stage1),
        "summary/stage1/test_f1_macro_mean": float(mean_f1_stage1),
        "summary/stage1/test_f1_macro_std": float(std_f1_stage1),
        "summary/stage2/test_acc_mean": float(mean_acc_stage2),
        "summary/stage2/test_acc_std": float(std_acc_stage2),
        "summary/stage2/test_f1_macro_mean": float(mean_f1_stage2),
        "summary/stage2/test_f1_macro_std": float(std_f1_stage2),
        "summary/improvement/test_acc": float(mean_acc_stage2 - mean_acc_stage1),
        "summary/improvement/test_f1_macro": float(mean_f1_stage2 - mean_f1_stage1),
    }

    final_stage2_hard_bin_mask_metrics = {
        "summary/stage2/hard_bin_masks": stage2_hard_bin_masks,
    }

    final_summary_log_metrics = {
        "summary/two_stage_input_pruning/final_summary_logs": per_fold_final_summary_logs,
    }

    # Upload the final consolidated log file as a W&B artifact.
    artifact_run = wandb.init(
        project="thesis-analysis",
        name=f"wear-loso-{args.wandb_run_name}-artifact-{args.preprocessing}",
        config={
            "dataset": "WEAR",
            "training_type": "two_stage_input_pruning",
            "preprocessing": args.preprocessing,
            "sparsity_weight_bin": args.sparsity_weight_bin,
            "fold_count": len(fold_subjects),
            "results_log_path": str(results_log_path),
        },
        job_type="log-artifact",
        reinit=True,
    )

    if artifact_run is not None:
        # Log final cross-fold summary metrics so they are visible in W&B charts/tables.
        artifact_run.log(final_summary_metrics)
        artifact_run.summary.update(final_summary_metrics)
        artifact_run.summary.update(final_stage2_hard_bin_mask_metrics)
        artifact_run.summary.update(final_summary_log_metrics)

        artifact = wandb.Artifact(
            name=f"wear-loso-{args.wandb_run_name}-results-{args.preprocessing}",
            type="results-log",
            metadata={
                "dataset": "WEAR",
                "preprocessing": args.preprocessing,
                "sparsity_weight_bin": args.sparsity_weight_bin,
                "epochs_stage1": args.epochs_stage1,
                "epochs_stage2": args.epochs_stage2,
                "log_every_n_epochs": args.log_every_n_epochs,
            },
        )
        artifact.add_file(str(results_log_path))
        artifact_run.log_artifact(artifact)
        artifact_run.finish()


if __name__ == "__main__":
    wandb.login()
    main()
