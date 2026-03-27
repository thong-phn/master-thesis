"""
Tasks:
	Init model and dataset
	Run Leave-One-Subject-Out cross validation on WEAR dataset
	Log training to wandb
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

from lib.wear_train import train_loso_wear
from lib.model import GumbelMaskSeparableConvCNN, SeparableConvCNN


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
	parser = argparse.ArgumentParser()
	parser.add_argument('--preprocessing', type=str, choices=['fft', 'dct', 'no'], default='fft',
	                    help='Preprocessing applied to signals: fft or dct')
	parser.add_argument('--sparsity_weight', type=float, default=0.01)
	parser.add_argument('--model', type=str, choices=['GumbelMaskSeparableConvCNN', 'SeparableConvCNN'], 
	                    default='GumbelMaskSeparableConvCNN', help='Model architectur to use')
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

	results_log_path = project_root / 'log' / f"wear_loso_results_{args.model}_{args.preprocessing}_{args.sparsity_weight}.txt"
	results_log_path.parent.mkdir(parents=True, exist_ok=True)
	with open(results_log_path, "w") as f:
		f.write("WEAR LOSO Results\n")

	test_accs = []
	test_f1s = []

	model_class = globals()[args.model]

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
			name=f"wear-loso-val-{val_subject}-{args.preprocessing}",
			config={
				"dataset": "WEAR",
				"train_subjects": train_subjects,
				"val_subjects": val_subjects,
				"test_subjects": test_subjects,
				"epochs": 60,
				"lr": 1e-3,
				"batch_size": 64,
				"model": args.model,
				"preprocessing": args.preprocessing,
			},
			reinit=True
		)

		metrics = train_loso_wear(
			root_path=root_path,
			model_class=model_class,
			train_subjects=train_subjects,
			val_subjects=val_subjects,
			wandb_run=wandb_run,
			epochs=60,
			lr=1e-3,
			batch_size=64,
			device=device,
			model_path=project_root / "models" / f"wear_best_model_subject{val_subject}_val.pth",
			preprocessing=args.preprocessing,
			sparsity_weight=args.sparsity_weight,
			tau_start=20.0,
			tau_end=1.0,
		)

		test_acc = metrics["test_acc"]
		test_f1_macro = metrics["test_f1_macro"]
		final_mask = metrics.get("final_mask", None)

		test_accs.append(test_acc)
		test_f1s.append(test_f1_macro)

		with open(results_log_path, "a") as f:
			f.write(f"\nFold Val Subject {val_subjects[0]}:\n")
			f.write(f"  Test Accuracy: {test_acc:.2f}%\n")
			f.write(f"  Test F1 Macro: {test_f1_macro:.4f}\n")
			if final_mask is not None:
				f.write(f"  Final Mask: {final_mask.tolist()}\n")

		if wandb_run is not None:
			wandb_run.finish()

	mean_acc = np.mean(test_accs)
	std_acc = np.std(test_accs)
	mean_f1 = np.mean(test_f1s)
	std_f1 = np.std(test_f1s)

	print("=" * 50)
	print("WEAR LOSO Cross-Validation Results")
	print(f"Model: {args.model}")
	print(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
	print(f"Test F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}")

	with open(results_log_path, "a") as f:
		f.write("\n" + "=" * 50 + "\n")
		f.write("Overall WEAR LOSO Results\n")
		f.write(f"Model: {args.model}\n")
		f.write(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
		f.write(f"Test F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}\n")


if __name__ == "__main__":
	wandb.login()
	main()
