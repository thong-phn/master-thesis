"""
Tasks:
	Init model and dataset
	Log training to wandb
"""
from pathlib import Path
import wandb
import random
import numpy as np
import torch

from train import train_loso
from model import SeparableConvCNN


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

	val_subjects = [1]
	train_subjects = [subject for subject in all_subjects if subject not in val_subjects]

	subject_test_path = root_path / "test" / "subject_test.txt"
	all_test_subjects = sorted(np.unique(np.loadtxt(subject_test_path, dtype=int)).tolist())
	test_subjects = [subject for subject in all_test_subjects]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print(f"Using device: {device}")
	print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
	print(f"Val subjects: {val_subjects}")
	print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

	# Tracking init
	wandb_run = wandb.init(
		project="thesis",
		name=f"val-subject-{val_subjects}-gyro-exp7-log1p-lr5e4",
		config={
			"train_subjects": train_subjects,
			"val_subjects": val_subjects,
			"test_subjects": test_subjects,
			"epochs": 60,
			"lr": 5e-4,
			"batch_size": 64,
			"model": "SeparableConvCNN",
			"use_gyro": True,
		},
	)
	# Log code version
	wandb_run.log_code(
		root=str(project_root),
		include_fn=lambda p: p.endswith((".py", ".yaml", ".yml"))
	)
	# Run training loop
	metrics = train_loso(
		root_path=root_path,
		model_class=SeparableConvCNN,
		train_subjects=train_subjects,
		val_subjects=val_subjects,
		wandb_run=wandb_run,
		use_gyro=True,
		epochs=60,
		lr=5e-4,
		batch_size=64,
		device=device,
		model_path=project_root / "models" / "best_model_subject1_val.pth",
	)

	# Training loop output
	print("Final metrics:")
	for key, value in metrics.items():
		print(f"  {key}: {value}")

	# Tracking finish
	if wandb_run is not None:
		wandb_run.finish()


if __name__ == "__main__":
	wandb.login()
	main()