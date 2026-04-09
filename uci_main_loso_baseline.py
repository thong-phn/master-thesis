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
from lib.model import SeparableConvCNN


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
	parser.add_argument('--preprocessing', type=str, choices=['fft', 'dct', 'ihw', 'no'], default='fft',
	                    help='Preprocessing applied to signals: fft, dct, ihw, or no')
	parser.add_argument('--model', type=str, default='SeparableConvCNN')
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--performance', action='store_true',
	                    help='Enable auto-tuned high-throughput DataLoader settings')
	parser.add_argument('--wandb_run_name', type=str, default=None,
	                    help='Optional base W&B run name. If provided, fold and summary suffixes are appended.')
	parser.add_argument('--wandb', type=bool, default=False)
	args = parser.parse_args()

	set_seed(42)
	project_root = Path(__file__).resolve().parent
	# UCI-HAR dataset root in this repository.
	root_path = project_root / "uci-har"

	# Load train and test subject IDs
	subject_train_path = root_path / "train" / "subject_train.txt"
	all_subjects = sorted(np.unique(np.loadtxt(subject_train_path, dtype=int)).tolist())

	subject_test_path = root_path / "test" / "subject_test.txt"
	all_test_subjects = sorted(np.unique(np.loadtxt(subject_test_path, dtype=int)).tolist())
	test_subjects = [subject for subject in all_test_subjects]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	results_log_path = project_root / 'log' / f"uci_loso_baseline_{args.preprocessing}.txt"
	results_log_path.parent.mkdir(parents=True, exist_ok=True)
	with open(results_log_path, "w") as f:
		f.write("UCI-HAR LOSO Results\n")
		f.write(f"Batch size: {args.batch_size}\n")
		f.write(f"Performance mode: {args.performance}\n")

	test_accs = []
	test_f1s = []

	model_class = globals()[args.model]

	for val_subject in all_subjects:
		val_subjects = [val_subject]
		train_subjects = [subject for subject in all_subjects if subject not in val_subjects]

		print("=" * 50)
		print(f"Fold: Val Subject {val_subjects[0]}")
		print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
		print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

		# Tracking init
		if args.wandb is not False:
			wandb_run = wandb.init(
				project="thesis-uci",
				name=f"uci-loso-val-{val_subject}-{args.preprocessing}-{args.wandb_run_name}",
				config={
					"dataset": "UCI-HAR",
					"train_subjects": train_subjects,
					"val_subjects": val_subjects,
					"test_subjects": test_subjects,
					"epochs": 60,
					"lr": 1e-3,
					"batch_size": args.batch_size,
					"performance": args.performance,
					"model": args.model,
					"preprocessing": args.preprocessing,
				},
				reinit=True
			)
		else: 
			wandb_run = None

		metrics = train_loso_wear(
			root_path=root_path,
			model_class=model_class,
			train_subjects=train_subjects,
			val_subjects=val_subjects,
			wandb_run=wandb_run,
			epochs=60,
			lr=1e-3,
			batch_size=args.batch_size,
			performance=args.performance,
			device=device,
			model_path=project_root / "models" / f"uci_best_model_subject{val_subject}_val.pth",
			preprocessing=args.preprocessing,
			dataset_name="uci-har",
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

		if wandb_run is not None:
			wandb_run.finish()

	mean_acc = np.mean(test_accs)
	std_acc = np.std(test_accs)
	mean_f1 = np.mean(test_f1s)
	std_f1 = np.std(test_f1s)

	print("=" * 50)
	print("UCI-HAR LOSO Cross-Validation Results")
	print(f"Model: {args.model}")
	print(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
	print(f"Test F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}")

	with open(results_log_path, "a") as f:
		f.write("\n" + "=" * 50 + "\n")
		f.write("Overall UCI-HAR LOSO Results\n")
		f.write(f"Model: {args.model}\n")
		f.write(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
		f.write(f"Test F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}\n")

	# Upload final training log as a W&B artifact.
	if args.wandb is not False:
		artifact_run_name = (
			f"{args.wandb_run_name}-summary" if args.wandb_run_name is not None
			else f"uci-loso-summary-{args.model}-{args.preprocessing}"
		)
		artifact_name = (
			f"uci-loso-log-{args.model}-{args.preprocessing}"
			.replace(".", "_")
		)
		artifact_run = wandb.init(
			project="thesis-uci",
			name=artifact_run_name,
			job_type="log-upload",
			reinit=True,
			config={
				"dataset": "UCI-HAR",
				"model": args.model,
				"preprocessing": args.preprocessing,
				"log_path": str(results_log_path),
			}
		)
		artifact = wandb.Artifact(
			name=artifact_name,
			type="training-log",
			description="Final UCI-HAR LOSO training log file"
		)
		artifact.add_file(str(results_log_path))
		artifact_run.log_artifact(artifact)
		artifact_run.finish()
		print(f"Uploaded log artifact '{artifact_name}' from: {results_log_path}")


if __name__ == "__main__":
	wandb.login()
	main()
