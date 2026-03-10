"""
Tasks:
	Init model and dataset
	Run Leave-One-Subject-Out cross validation
	Log training to wandb
"""
from pathlib import Path
import wandb
import random
import numpy as np
import torch

from train import train_loso
from model import GumbelMaskSeparableConvCNN


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def main():
	set_seed(42)

	project_root = Path(__file__).resolve().parent
	root_path = project_root / "uci-har"

	subject_train_path = root_path / "train" / "subject_train.txt"
	all_subjects = sorted(np.unique(np.loadtxt(subject_train_path, dtype=int)).tolist())

	subject_test_path = root_path / "test" / "subject_test.txt"
	all_test_subjects = sorted(np.unique(np.loadtxt(subject_test_path, dtype=int)).tolist())
	test_subjects = [subject for subject in all_test_subjects]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

	test_accs = []
	test_f1s = []

	for val_subject in all_subjects:
		val_subjects = [val_subject]
		train_subjects = [subject for subject in all_subjects if subject not in val_subjects]

		print("="*50)
		print(f"Starting Fold: Val Subject {val_subjects[0]}")
		print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
		
		# Tracking init
		wandb_run = wandb.init(
			project="thesis",
			name=f"loso-val-{val_subject}-gyro-exp7-lr1e3",
			config={
				"train_subjects": train_subjects,
				"val_subjects": val_subjects,
				"test_subjects": test_subjects,
				"epochs": 60,
				"lr": 1e-3,
				"batch_size": 64,
				"model": "GumbelMaskSeparableConvCNN",
			},
			reinit=True
		)

		metrics = train_loso(
			root_path=root_path,
			model_class=GumbelMaskSeparableConvCNN,
			train_subjects=train_subjects,
			val_subjects=val_subjects,
			wandb_run=wandb_run,
			epochs=60,
			lr=1e-3,
			batch_size=64,
			device=device,
			model_path=project_root / "models" / f"best_model_subject{val_subject}_val.pth",
		)

		test_accs.append(metrics["test_acc"])
		test_f1s.append(metrics["test_f1_macro"])

		if wandb_run is not None:
			wandb_run.finish()

	mean_acc = np.mean(test_accs)
	std_acc = np.std(test_accs)
	mean_f1 = np.mean(test_f1s)
	std_f1 = np.std(test_f1s)

	print("="*50)
	print("LOSO Cross-Validation Results")
	print(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
	print(f"Test F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}")

	# Log the summary results in an overall metrics dictionary (could also write to a file)
	with open(project_root / "loso_results.txt", "w") as f:
		f.write("LOSO Cross-Validation Results\n")
		f.write(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
		f.write(f"Test F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}\n")


if __name__ == "__main__":
	wandb.login()
	main()
