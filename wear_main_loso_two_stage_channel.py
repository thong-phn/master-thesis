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

from lib.wear_train import train_loso_wear_two_stage_channel


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
        description='Two-stage LOSO training on WEAR dataset'
    )
    parser.add_argument('--preprocessing', type=str, choices=['fft', 'dct', 'no'], default='fft',
                        help='Preprocessing applied to signals: fft, dct, or no')
    parser.add_argument('--model', type=str, choices=['Separable'], default='Separable',
                        help="Model family for this script: 'Separable' (channel-pruning stage 2)")
    parser.add_argument('--sparsity_weight', type=float, default=0.01,
                        help='Weight for sparsity loss in stage 2 (Gumbel mask)')
    parser.add_argument('--epochs_stage1', type=int, default=60,
                        help='Number of epochs for stage 1 (SeparableConvCNN)')
    parser.add_argument('--epochs_stage2', type=int, default=60,
                        help='Number of epochs for stage 2 (GumbelMask)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for both stages')
    parser.add_argument('--stage2_backbone_lr_factor', type=float, default=0.1,
                        help='Stage 2 LR multiplier for non-Gumbel parameters (final backbone LR = lr * factor)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--tau_start', type=float, default=10.0,
                        help='Initial temperature for Gumbel-Softmax in stage 2')
    parser.add_argument('--tau_end', type=float, default=1.0,
                        help='Final temperature for Gumbel-Softmax in stage 2')
    parser.add_argument('--stage1_model_path', type=str, default=None,
                        help='Optional path to a pretrained Stage 1 checkpoint. If provided, Stage 1 training is skipped.')
    
    args = parser.parse_args()

    set_seed(42)

    project_root = Path(__file__).resolve().parent
    root_path = project_root / "wear"

    # Load train and test subject IDs
    subject_train_path = root_path / "train" / "subject_train.txt"
    all_subjects = sorted(np.loadtxt(subject_train_path, dtype=int).tolist())

    subject_test_path = root_path / "test" / "subject_test.txt"
    all_test_subjects = sorted(np.loadtxt(subject_test_path, dtype=int).tolist())
    test_subjects = [subject for subject in all_test_subjects]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_log_path = project_root / 'log' / f"wear_loso_two_stage_results_{args.model}_{args.preprocessing}.txt"
    results_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_log_path, "w") as f:
        f.write("WEAR LOSO Two-Stage Training Results\n")
        f.write(f"Preprocessing: {args.preprocessing}\n")
        f.write(f"Model Family: {args.model}\n")
        f.write(f"Sparsity Weight: {args.sparsity_weight}\n")
        f.write(f"Epochs Stage 1: {args.epochs_stage1}\n")
        f.write(f"Epochs Stage 2: {args.epochs_stage2}\n")
        f.write(f"Stage 2 Backbone LR Factor: {args.stage2_backbone_lr_factor}\n")
        if args.stage1_model_path is not None:
            f.write(f"Stage 1 Checkpoint Override: {args.stage1_model_path}\n")
        f.write("\n")

    test_accs_stage1 = []
    test_f1s_stage1 = []
    test_accs_stage2 = []
    test_f1s_stage2 = []

    stage1_label = 'SeparableConvCNN'
    stage2_label = 'GumbelChannelPruningCNN'

    # Run LOSO on first subject as example (can be extended to all subjects)
    # for val_subject in all_subjects:
    for val_subject in [0]:
        val_subjects = [val_subject]
        train_subjects = [subject for subject in all_subjects if subject not in val_subjects]

        print("=" * 50)
        print(f"Fold: Val Subject {val_subjects[0]}")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

        # Tracking init
        wandb_run = wandb.init(
            project="thesis",
            name=f"wear-loso-two-stage-val-{val_subject}-{args.preprocessing}",
            config={
                "dataset": "WEAR",
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "epochs_stage1": args.epochs_stage1,
                "epochs_stage2": args.epochs_stage2,
                "lr": args.lr,
                "stage2_backbone_lr_factor": args.stage2_backbone_lr_factor,
                "batch_size": args.batch_size,
                "model_family": args.model,
                "preprocessing": args.preprocessing,
                "sparsity_weight": args.sparsity_weight,
                "training_type": "two_stage",
                "stage1_model_path": args.stage1_model_path,
            },
            reinit=True
        )

        metrics = train_loso_wear_two_stage_channel(
            root_path=root_path,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            wandb_run=wandb_run,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            model_path=project_root / "models" / f"wear_best_model_two_stage_subject{val_subject}_val.pth",
            preprocessing=args.preprocessing,
            sparsity_weight=args.sparsity_weight,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            dropout=args.dropout,
            stage2_backbone_lr_factor=args.stage2_backbone_lr_factor,
            stage1_model_path=args.stage1_model_path,
        )

        # Extract metrics from both stages
        test_acc_stage1 = metrics["stage1"]["test_acc"]
        test_f1_stage1 = metrics["stage1"]["test_f1_macro"]
        
        test_acc_stage2 = metrics["stage2"]["test_acc"]
        test_f1_stage2 = metrics["stage2"]["test_f1_macro"]
        final_mask = metrics["stage2"].get("final_mask", None)
        pruning_stats = metrics["stage2"].get("pruning_stats", None)

        test_accs_stage1.append(test_acc_stage1)
        test_f1s_stage1.append(test_f1_stage1)
        test_accs_stage2.append(test_acc_stage2)
        test_f1s_stage2.append(test_f1_stage2)

        # Log results
        with open(results_log_path, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Fold Val Subject {val_subjects[0]}:\n")
            f.write(f"\nStage 1 ({stage1_label}):\n")
            f.write(f"  Test Accuracy: {test_acc_stage1:.2f}%\n")
            f.write(f"  Test F1 Macro: {test_f1_stage1:.4f}\n")
            f.write(f"\nStage 2 ({stage2_label}):\n")
            f.write(f"  Test Accuracy: {test_acc_stage2:.2f}%\n")
            f.write(f"  Test F1 Macro: {test_f1_stage2:.4f}\n")
            f.write(f"  Improvement: {test_acc_stage2 - test_acc_stage1:.2f}%\n")
            if isinstance(final_mask, dict):
                f.write("  Final Masks:\n")
                for block_name, block_mask in final_mask.items():
                    f.write(f"    {block_name}: {block_mask.tolist()}\n")
            elif final_mask is not None:
                f.write(f"  Final Mask: {final_mask.tolist()}\n")
            if pruning_stats is not None:
                f.write("  Pruning Stats:\n")
                for k, v in pruning_stats.items():
                    f.write(f"    {k}: {v:.2f}\n")

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


if __name__ == "__main__":
    wandb.login()
    main()
