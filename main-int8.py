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
from datetime import datetime, timezone

from train import train_loso_int8, MyDataset
from model import SeparableConvCNN
from representative_data import export_representative_data


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def track_event(tracker, stage, status, message, details=None):
	entry = {
		"time": datetime.now(timezone.utc).isoformat(),
		"stage": stage,
		"status": status,
		"message": message,
	}
	if details is not None:
		entry["details"] = details
	tracker.append(entry)
	print(f"[TRACK][{status}] {stage}: {message}")


def _macro_f1_score(y_true, y_pred, num_classes):
	f1_scores = []
	for class_id in range(num_classes):
		tp = int(np.sum((y_true == class_id) & (y_pred == class_id)))
		fp = int(np.sum((y_true != class_id) & (y_pred == class_id)))
		fn = int(np.sum((y_true == class_id) & (y_pred != class_id)))

		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		if (precision + recall) == 0.0:
			f1_scores.append(0.0)
		else:
			f1_scores.append(2.0 * precision * recall / (precision + recall))

	return float(np.mean(f1_scores))


def evaluate_tflite_int8(root_path, tflite_path, use_gyro=False):
	try:
		import tensorflow as tf
	except Exception as exc:
		raise RuntimeError("TensorFlow is required to run TFLite accuracy/F1 evaluation.") from exc

	interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
	interpreter.allocate_tensors()

	input_detail = interpreter.get_input_details()[0]
	output_detail = interpreter.get_output_details()[0]

	test_dataset = MyDataset(root_path=root_path, split="test", subject_ids=None, use_gyro=use_gyro)
	y_true = []
	y_pred = []

	for idx in range(len(test_dataset)):
		features, label = test_dataset[idx]
		x = features.numpy()[None, ...].astype(np.float32)

		input_dtype = input_detail["dtype"]
		if input_dtype in (np.int8, np.uint8):
			scale, zero_point = input_detail["quantization"]
			if scale <= 0:
				raise ValueError("Invalid input quantization scale in TFLite model.")
			x = np.round(x / scale + zero_point)
			qinfo = np.iinfo(input_dtype)
			x = np.clip(x, qinfo.min, qinfo.max).astype(input_dtype)
		else:
			x = x.astype(input_dtype)

		interpreter.set_tensor(input_detail["index"], x)
		interpreter.invoke()
		logits = interpreter.get_tensor(output_detail["index"])
		pred = int(np.argmax(logits, axis=-1)[0])

		y_true.append(int(label.item()))
		y_pred.append(pred)

	y_true = np.array(y_true, dtype=np.int64)
	y_pred = np.array(y_pred, dtype=np.int64)
	acc = float((y_true == y_pred).mean() * 100.0)
	num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
	f1_macro = _macro_f1_score(y_true, y_pred, num_classes=num_classes)

	return acc, f1_macro


def convert_checkpoint_to_tflite_with_litert_torch(
	model_class,
	checkpoint_path,
	tflite_output_path,
	num_channels,
	input_len,
	representative_npz_path=None,
	full_integer_int8=True,
	calibration_count=200,
):
	try:
		import litert_torch
	except Exception as exc:
		raise RuntimeError(
			"Failed to import litert_torch. This is often caused by a broken TensorFlow/tf-nightly "
			"binary in the active environment. Reinstall litert-torch with a compatible TensorFlow stack. "
			f"Original error: {type(exc).__name__}: {exc}"
		) from exc

	model = model_class(num_channels=num_channels, freq_bins=input_len)
	model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
	model.eval()

	representative_array = None
	if representative_npz_path is not None:
		representative = np.load(representative_npz_path)
		if "input" in representative:
			representative_array = representative["input"].astype(np.float32)
		elif "x" in representative:
			representative_array = representative["x"].astype(np.float32)
		else:
			raise ValueError("Representative NPZ must contain 'input' or 'x'.")
		sample = representative_array[0:1]
		sample_input = torch.from_numpy(sample.astype(np.float32))
	else:
		sample_input = torch.randn(1, num_channels, input_len, dtype=torch.float32)

	ai_edge_converter_flags = None
	if full_integer_int8:
		try:
			import tensorflow as tf
		except Exception as exc:
			raise RuntimeError(
				"TensorFlow import failed while preparing INT8 converter flags. "
				f"Original error: {type(exc).__name__}: {exc}"
			) from exc

		if representative_array is None:
			raise ValueError("Representative data is required for full-integer INT8 calibration.")

		calibration_count = int(min(calibration_count, representative_array.shape[0]))

		def representative_dataset():
			for i in range(calibration_count):
				yield [representative_array[i:i+1]]

		ai_edge_converter_flags = {
			"optimizations": [tf.lite.Optimize.DEFAULT],
			"representative_dataset": representative_dataset,
			"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
			"inference_input_type": tf.int8,
			"inference_output_type": tf.int8,
		}

	edge_model = litert_torch.convert(
		model,
		(sample_input,),
		_ai_edge_converter_flags=ai_edge_converter_flags,
	)
	tflite_output_path.parent.mkdir(parents=True, exist_ok=True)
	edge_model.export(str(tflite_output_path))




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
	use_gyro = True
	tracker = []

	print(f"Using device: {device}")
	print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
	print(f"Val subjects: {val_subjects}")
	print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

	# Tracking init
	wandb_run = wandb.init(
		project="thesis",
		name=f"val-subject-{val_subjects}-gyro-21firstbin",
		config={
			"train_subjects": train_subjects,
			"val_subjects": val_subjects,
			"test_subjects": test_subjects,
			"epochs": 60,
			"lr": 1e-3,
			"batch_size": 64,
			"model": "SeparableConvCNN",
			"use_gyro": use_gyro,
		},
	)
	# Log code version
	wandb_run.log_code(
		root=str(project_root),
		include_fn=lambda p: p.endswith((".py", ".yaml", ".yml", ".md"))
	)

	# TRAINING LOOP
	track_event(tracker, "train", "start", "Starting LOSO training for INT8 pipeline")
	metrics = train_loso_int8(
		root_path=root_path,
		model_class=SeparableConvCNN,
		train_subjects=train_subjects,
		val_subjects=val_subjects,
		wandb_run=wandb_run,
		use_gyro=use_gyro,
		epochs=60,
		lr=1e-3,
		batch_size=64,
		device=device,
		model_path=project_root / "models" / "best_model_subject1_val.pth",
	)
	track_event(tracker, "train", "ok", "Training completed", {
		"best_val_loss": metrics.get("best_val_loss"),
		"test_acc": metrics.get("test_acc"),
	})

	# Model I/O shape metadata
	num_channels = int(metrics["num_channels"])
	input_len = int(metrics["input_len"])

	# Phase 2: Calibration-size sweep with stratified representative sampling
	calibration_sweep = [200, 500, 1000, 2000]
	int8_sweep_results = []
	best_result = None

	for calibration_count in calibration_sweep:
		representative_npz = project_root / "models" / f"representative_data_stratified_{calibration_count}.npz"
		tflite_path = project_root / "models" / f"best_model_int8_calib{calibration_count}.tflite"

		track_event(
			tracker,
			"representative_data",
			"start",
			"Exporting representative calibration data",
			{"npz": str(representative_npz), "calibration_count": calibration_count, "sampling_method": "stratified"},
		)
		export_representative_data(
			root_path=root_path,
			output_path=representative_npz,
			num_samples=calibration_count,
			split="train",
			use_gyro=use_gyro,
			onnx_path=None,
			sampling_method="stratified",
			seed=42,
		)
		track_event(tracker, "representative_data", "ok", "Representative data exported", {"npz": str(representative_npz)})

		track_event(tracker, "litert_convert", "start", "Converting checkpoint to TFLite via LiteRT-Torch", {"tflite_path": str(tflite_path), "calibration_count": calibration_count})
		convert_checkpoint_to_tflite_with_litert_torch(
			model_class=SeparableConvCNN,
			checkpoint_path=metrics["model_path"],
			tflite_output_path=tflite_path,
			num_channels=num_channels,
			input_len=input_len,
			representative_npz_path=representative_npz,
			calibration_count=calibration_count,
		)
		track_event(tracker, "litert_convert", "ok", "LiteRT-Torch conversion completed", {"tflite_path": str(tflite_path)})

		track_event(tracker, "int8_eval", "start", "Evaluating INT8 TFLite model on test split", {"calibration_count": calibration_count})
		int8_test_acc, int8_test_f1_macro = evaluate_tflite_int8(
			root_path=root_path,
			tflite_path=tflite_path,
			use_gyro=use_gyro,
		)
		track_event(tracker, "int8_eval", "ok", "INT8 evaluation completed", {
			"calibration_count": calibration_count,
			"int8_test_acc": int8_test_acc,
			"int8_test_f1_macro": int8_test_f1_macro,
		})

		result = {
			"calibration_count": calibration_count,
			"sampling_method": "stratified",
			"tflite_path": str(tflite_path),
			"int8_test_acc": int8_test_acc,
			"int8_test_f1_macro": int8_test_f1_macro,
		}
		int8_sweep_results.append(result)

		if best_result is None or result["int8_test_acc"] > best_result["int8_test_acc"]:
			best_result = result

	metrics["int8_sweep_results"] = int8_sweep_results
	if best_result is not None:
		metrics["tflite_int8_path"] = best_result["tflite_path"]
		metrics["int8_test_acc"] = best_result["int8_test_acc"]
		metrics["int8_test_f1_macro"] = best_result["int8_test_f1_macro"]
		metrics["best_calibration_count"] = best_result["calibration_count"]

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