"""
Multi-Stage Training for WEAR Dataset LOSO:
    Stage 1: Train SeparableConvCNN without Gumbel mask and save best weights
    Stage 2: Load stage 1 weights into GumbelMaskSeparableConvCNN and prune input bins
    Stage 3: Apply pruned input and retrain SeparableConvCNN
    Stage 4: Apply channel pruning model on pruned input and retrain
    Stage 5: Physically prune the unused channels and fine tune the compact model
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

from lib.wear_train import train_loso_wear_multi_stage


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
    return sorted(np.atleast_1d(np.loadtxt(path, dtype=int)).astype(int).tolist())


def _upload_results_log_to_wandb(log_path: Path, preprocessing: str, single_subject_only: bool):
    if not log_path.exists():
        print(f"Skipping W&B upload: log file not found at {log_path}")
        return

    upload_run = None
    try:
        run_suffix = "single-subject" if single_subject_only else "all-subjects"
        upload_run = wandb.init(
            project="thesis",
            name=f"wear-loso-five-stage-log-{preprocessing}-{run_suffix}",
            job_type="results_log_upload",
            reinit=True,
            config={
                "dataset": "WEAR",
                "training_type": "five_stage",
                "preprocessing": preprocessing,
                "single_subject_only": single_subject_only,
                "results_log_file": str(log_path),
            },
        )

        artifact = wandb.Artifact(
            name=f"wear-loso-five-stage-results-{preprocessing}-{run_suffix}",
            type="results-log",
        )
        artifact.add_file(str(log_path))
        upload_run.log_artifact(artifact)
        print(f"Uploaded results log to W&B artifact: {log_path}")
    except Exception as e:
        print(f"Failed to upload results log to W&B: {e}")
    finally:
        if upload_run is not None:
            upload_run.finish()


def main():
    parser = argparse.ArgumentParser(description='Five-stage LOSO training on WEAR dataset')
    parser.add_argument('--preprocessing', type=str, choices=['fft', 'dct', 'ihw', 'no'], default='fft',
                        help='Preprocessing applied to signals: fft, dct, ihw, or no')
    parser.add_argument('--epochs_stage1', type=int, default=60,
                        help='Number of epochs for stage 1 (SeparableConvCNN)')
    parser.add_argument('--epochs_stage2', type=int, default=60,
                        help='Number of epochs for stage 2 (input-bin Gumbel pruning)')
    parser.add_argument('--epochs_stage3', type=int, default=60,
                        help='Number of epochs for stage 3 (retrain on pruned input)')
    parser.add_argument('--epochs_stage4', type=int, default=60,
                        help='Number of epochs for stage 4 (channel pruning on pruned input)')
    parser.add_argument('--epochs_stage5', type=int, default=60,
                        help='Number of epochs for stage 5 (pruned-model fine tuning)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Base learning rate')
    parser.add_argument('--stage2_backbone_lr_factor', type=float, default=0.1,
                        help='Stage 2 LR multiplier for non-Gumbel parameters (input-bin pruning stage)')
    parser.add_argument('--stage4_backbone_lr_factor', type=float, default=0.1,
                        help='Stage 4 LR multiplier for non-Gumbel parameters (channel pruning stage)')
    parser.add_argument('--stage5_loaded_lr_factor', type=float, default=0.1,
                        help='Stage 5 LR multiplier for weights loaded from the checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--tau_start', type=float, default=10.0,
                        help='Initial temperature for Gumbel-Softmax')
    parser.add_argument('--tau_end', type=float, default=1.0,
                        help='Final temperature for Gumbel-Softmax')
    parser.add_argument('--sparsity_weight_bin', type=float, default=0.1,
                        help='Sparsity weight for stage 2 input-bin pruning')
    parser.add_argument('--sparsity_weight_channel', type=float, default=0.1,
                        help='Sparsity weight for stage 4 channel pruning')
    parser.add_argument('--stage1_model_path', type=str, default=None,
                        help='Optional pretrained Stage 1 checkpoint path; if set, Stage 1 training is skipped.')
    parser.add_argument('--stage2_model_path', type=str, default=None,
                        help='Optional pretrained Stage 2 checkpoint path; if set, Stage 2 training is skipped.')
    parser.add_argument('--stage3_model_path', type=str, default=None,
                        help='Optional pretrained Stage 3 checkpoint path; if set, Stage 3 training is skipped.')
    parser.add_argument('--stage4_model_path', type=str, default=None,
                        help='Optional pretrained Stage 4 checkpoint path; if set, Stage 4 training is skipped.')
    parser.add_argument('--stage5_model_path', type=str, default=None,
                        help='Optional pretrained Stage 5 checkpoint path; if set, Stage 5 training is skipped.')
    parser.add_argument('--single_subject_only', action='store_true',
                        help='Run only the first LOSO fold (subject 0 if available).')

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
    all_subjects = _load_subject_ids(subject_train_path)

    subject_test_path = root_path / "test" / "subject_test.txt"
    test_subjects = _load_subject_ids(subject_test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_log_path = project_root / 'log' / f"wear_loso_five_stage_results_{args.preprocessing}.txt"
    results_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_log_path, "w") as f:
        f.write("WEAR LOSO Five-Stage Training Results\n")
        f.write(f"Preprocessing: {args.preprocessing}\n")
        f.write(f"Epochs Stage 1: {args.epochs_stage1}\n")
        f.write(f"Epochs Stage 2: {args.epochs_stage2}\n")
        f.write(f"Epochs Stage 3: {args.epochs_stage3}\n")
        f.write(f"Epochs Stage 4: {args.epochs_stage4}\n")
        f.write(f"Epochs Stage 5: {args.epochs_stage5}\n")
        f.write(f"Sparsity Weight Bin: {args.sparsity_weight_bin}\n")
        f.write(f"Sparsity Weight Channel: {args.sparsity_weight_channel}\n")
        f.write(f"Stage 2 Backbone LR Factor: {args.stage2_backbone_lr_factor}\n")
        f.write(f"Stage 4 Backbone LR Factor: {args.stage4_backbone_lr_factor}\n")
        f.write(f"Stage 5 Loaded LR Factor: {args.stage5_loaded_lr_factor}\n")
        if args.stage1_model_path is not None:
            f.write(f"Stage 1 Checkpoint Override: {args.stage1_model_path}\n")
        if args.stage2_model_path is not None:
            f.write(f"Stage 2 Checkpoint Override: {args.stage2_model_path}\n")
        if args.stage3_model_path is not None:
            f.write(f"Stage 3 Checkpoint Override: {args.stage3_model_path}\n")
        if args.stage4_model_path is not None:
            f.write(f"Stage 4 Checkpoint Override: {args.stage4_model_path}\n")
        if args.stage5_model_path is not None:
            f.write(f"Stage 5 Checkpoint Override: {args.stage5_model_path}\n")
        f.write("\n")

    stage_names = ["stage1", "stage2", "stage3", "stage4", "stage5"]
    metrics_history = {s: {"acc": [], "f1": []} for s in stage_names}

    fold_subjects = [all_subjects[0]] if args.single_subject_only else all_subjects

    for val_subject in fold_subjects:
    # for val_subject in [0]:

        val_subjects = [val_subject]
        train_subjects = [subject for subject in all_subjects if subject not in val_subjects]

        print("=" * 50)
        print(f"Fold: Val Subject {val_subjects[0]}")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

        wandb_run = wandb.init(
            project="thesis-analysis",
            name=f"wear-loso-five-stage-val-{val_subject}-{args.preprocessing}",
            config={
                "dataset": "WEAR",
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "epochs_stage1": args.epochs_stage1,
                "epochs_stage2": args.epochs_stage2,
                "epochs_stage3": args.epochs_stage3,
                "epochs_stage4": args.epochs_stage4,
                "epochs_stage5": args.epochs_stage5,
                "lr": args.lr,
                "stage2_backbone_lr_factor": args.stage2_backbone_lr_factor,
                "stage4_backbone_lr_factor": args.stage4_backbone_lr_factor,
                "stage5_loaded_lr_factor": args.stage5_loaded_lr_factor,
                "batch_size": args.batch_size,
                "preprocessing": args.preprocessing,
                "sparsity_weight_bin": args.sparsity_weight_bin,
                "sparsity_weight_channel": args.sparsity_weight_channel,
                "training_type": "five_stage",
                "stage1_model_path": args.stage1_model_path,
                "stage2_model_path": args.stage2_model_path,
                "stage3_model_path": args.stage3_model_path,
                "stage4_model_path": args.stage4_model_path,
                "stage5_model_path": args.stage5_model_path,
            },
            reinit=True,
        )

        metrics = train_loso_wear_multi_stage(
            root_path=root_path,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            wandb_run=wandb_run,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            epochs_stage3=args.epochs_stage3,
            epochs_stage4=args.epochs_stage4,
            epochs_stage5=args.epochs_stage5,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            model_path=project_root / "models" / f"wear_best_model_five_stage_subject{val_subject}_val.pth",
            preprocessing=args.preprocessing,
            sparsity_weight_bin=args.sparsity_weight_bin,
            sparsity_weight_channel=args.sparsity_weight_channel,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            dropout=args.dropout,
            stage2_backbone_lr_factor=args.stage2_backbone_lr_factor,
            stage4_backbone_lr_factor=args.stage4_backbone_lr_factor,
            stage5_loaded_lr_factor=args.stage5_loaded_lr_factor,
            stage1_model_path=args.stage1_model_path,
            stage2_model_path=args.stage2_model_path,
            stage3_model_path=args.stage3_model_path,
            stage4_model_path=args.stage4_model_path,
            stage5_model_path=args.stage5_model_path,
        )

        for stage in stage_names:
            metrics_history[stage]["acc"].append(metrics[stage]["test_acc"])
            metrics_history[stage]["f1"].append(metrics[stage]["test_f1_macro"])

        with open(results_log_path, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Fold Val Subject {val_subjects[0]}:\n")
            for stage in stage_names:
                f.write(f"\n{stage.upper()} ({metrics[stage]['model']}):\n")
                f.write(f"  Test Accuracy: {metrics[stage]['test_acc']:.2f}%\n")
                f.write(f"  Test F1 Macro: {metrics[stage]['test_f1_macro']:.4f}\n")
            f.write(f"\n  Improvement Stage5 - Stage1: {metrics['stage5']['test_acc'] - metrics['stage1']['test_acc']:.2f}%\n")
            
            if "param_reduction_pct" in metrics["stage5"]:
                f.write("\n  Stage 5 Model Size Reduction:\n")
                f.write(f"    Dense Params: {metrics['stage5']['dense_param_count']:,}\n")
                f.write(f"    Pruned Params: {metrics['stage5']['pruned_param_count']:,}\n")
                f.write(f"    Reduction: {metrics['stage5']['param_reduction_pct']:.2f}%\n")

            hard_bin_mask = metrics["stage2"].get("hard_bin_mask")
            if hard_bin_mask is not None:
                f.write(f"  Hard Bin Mask: {hard_bin_mask.tolist()}\n")

            final_mask = metrics["stage4"].get("final_mask")
            if isinstance(final_mask, dict):
                f.write("  Final Channel Masks:\n")
                for block_name, block_mask in final_mask.items():
                    f.write(f"    {block_name}: {block_mask.tolist()}\n")

            pruning_stats = metrics["stage4"].get("pruning_stats")
            if pruning_stats is not None:
                f.write("  Channel Pruning Stats:\n")
                for k, v in pruning_stats.items():
                    f.write(f"    {k}: {v:.2f}\n")

        if wandb_run is not None:
            wandb_run.finish()

    print("=" * 50)
    print("WEAR LOSO Five-Stage Cross-Validation Results")
    print("=" * 50)
    with open(results_log_path, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Overall WEAR LOSO Five-Stage Results\n")
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

    improve_acc = float(np.mean(metrics_history['stage5']['acc']) - np.mean(metrics_history['stage1']['acc']))
    improve_f1 = float(np.mean(metrics_history['stage5']['f1']) - np.mean(metrics_history['stage1']['f1']))

    print("\nImprovement (Stage5 - Stage1):")
    print(f"  Accuracy: {improve_acc:.2f}%")
    print(f"  F1 Macro: {improve_f1:.4f}")

    with open(results_log_path, "a") as f:
        f.write("\nImprovement (Stage5 - Stage1):\n")
        f.write(f"  Accuracy: {improve_acc:.2f}%\n")
        f.write(f"  F1 Macro: {improve_f1:.4f}\n")

    _upload_results_log_to_wandb(
        log_path=results_log_path,
        preprocessing=args.preprocessing,
        single_subject_only=args.single_subject_only,
    )


if __name__ == "__main__":
    wandb.login()
    main()
