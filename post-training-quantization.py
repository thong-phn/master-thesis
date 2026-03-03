from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.utils.data import DataLoader, Subset
from torch.export import export

from model import SeparableConvCNN
from train import MyDataset


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def compute_minmax_stats(calibration_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
	min_vals = None
	max_vals = None

	for features, _ in calibration_loader:
		batch_min = features.amin(dim=(0, 2), keepdim=True)
		batch_max = features.amax(dim=(0, 2), keepdim=True)
		if min_vals is None:
			min_vals = batch_min
			max_vals = batch_max
		else:
			min_vals = torch.minimum(min_vals, batch_min)
			max_vals = torch.maximum(max_vals, batch_max)

	if min_vals is None or max_vals is None:
		raise RuntimeError("Calibration loader is empty. Cannot compute min-max normalization stats.")

	return min_vals, max_vals


def apply_minmax_normalization(
	features: torch.Tensor,
	min_vals: torch.Tensor,
	max_vals: torch.Tensor,
	eps: float = 1e-8,
) -> torch.Tensor:
	denominator = (max_vals - min_vals).clamp_min(eps)
	return (features - min_vals) / denominator


def build_dataloaders(
	root_path: Path,
	train_subjects: list[int],
	test_subjects: list[int],
	use_gyro: bool,
	batch_size: int,
	num_workers: int,
	calibration_shuffle: bool,
	calibration_samples: int,
	balanced_calibration: bool,
	seed: int,
) -> Tuple[DataLoader, DataLoader]:
	calibration_dataset = MyDataset(
		root_path=root_path,
		split="train",
		subject_ids=train_subjects,
		use_gyro=use_gyro,
	)
	test_dataset = MyDataset(
		root_path=root_path,
		split="test",
		subject_ids=test_subjects,
		use_gyro=use_gyro,
	)

	if calibration_samples > 0:
		total_len = len(calibration_dataset)
		requested_samples = min(calibration_samples, total_len)
		labels = np.asarray(calibration_dataset.labels)
		rng = np.random.default_rng(seed)

		if balanced_calibration:
			selected_indices = []
			class_ids = sorted(np.unique(labels).tolist())
			num_classes = max(len(class_ids), 1)
			base_per_class = requested_samples // num_classes
			remainder = requested_samples % num_classes

			for class_index, class_id in enumerate(class_ids):
				class_indices = np.where(labels == class_id)[0]
				rng.shuffle(class_indices)
				target = base_per_class + (1 if class_index < remainder else 0)
				target = min(target, len(class_indices))
				selected_indices.extend(class_indices[:target].tolist())

			if len(selected_indices) < requested_samples:
				remaining = requested_samples - len(selected_indices)
				selected_set = set(selected_indices)
				pool = np.array([idx for idx in range(total_len) if idx not in selected_set])
				rng.shuffle(pool)
				selected_indices.extend(pool[:remaining].tolist())
		else:
			all_indices = np.arange(total_len)
			rng.shuffle(all_indices)
			selected_indices = all_indices[:requested_samples].tolist()

		calibration_dataset = Subset(calibration_dataset, selected_indices)

	calibration_loader = DataLoader(
		calibration_dataset,
		batch_size=batch_size,
		shuffle=calibration_shuffle,
		num_workers=num_workers,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
	)
	return calibration_loader, test_loader


def evaluate_model(
	model: nn.Module,
	dataloader: DataLoader,
	device: torch.device,
	force_singleton: bool = False,
	norm_stats: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Dict[str, float]:
	try:
		model.eval()
	except NotImplementedError:
		tq.move_exported_model_to_eval(model)
	criterion = nn.CrossEntropyLoss()
	loss_sum = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for features, labels in dataloader:
			if norm_stats is not None:
				min_vals, max_vals = norm_stats
				features = apply_minmax_normalization(features, min_vals, max_vals)
			features = features.to(device)
			labels = labels.to(device)
			if force_singleton and features.size(0) > 1:
				for index in range(features.size(0)):
					output = model(features[index : index + 1])
					target = labels[index : index + 1]
					loss = criterion(output, target)

					loss_sum += loss.item()
					total += 1
					correct += output.argmax(dim=1).eq(target).sum().item()
			else:
				outputs = model(features)
				loss = criterion(outputs, labels)
				batch_size = labels.size(0)

				loss_sum += loss.item() * batch_size
				total += batch_size
				correct += outputs.argmax(dim=1).eq(labels).sum().item()

	return {
		"loss": loss_sum / max(total, 1),
		"acc": 100.0 * correct / max(total, 1),
	}


def calibrate_model(
	prepared_model: nn.Module,
	calibration_loader: DataLoader,
	max_calibration_batches: int,
	norm_stats: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
	tq.move_exported_model_to_eval(prepared_model)
	with torch.no_grad():
		for batch_index, (features, _) in enumerate(calibration_loader):
			if norm_stats is not None:
				min_vals, max_vals = norm_stats
				features = apply_minmax_normalization(features, min_vals, max_vals)
			for index in range(features.size(0)):
				prepared_model(features[index : index + 1])
			if 0 < max_calibration_batches <= (batch_index + 1):
				break


def quantize_pt2e_full_int8(
	fp32_model: nn.Module,
	example_input: torch.Tensor,
	calibration_loader: DataLoader,
	max_calibration_batches: int,
	is_per_channel: bool,
	act_qmin: int,
	act_qmax: int,
	weight_qmin: int,
	weight_qmax: int,
	norm_stats: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> nn.Module:
	from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
	from torch.ao.quantization.quantizer.xnnpack_quantizer import (
		XNNPACKQuantizer,
		get_symmetric_quantization_config,
	)

	exported_program = export(fp32_model, (example_input,))
	graph_module = exported_program.module()

	quantizer = XNNPACKQuantizer().set_global(
		get_symmetric_quantization_config(
			is_per_channel=is_per_channel,
			is_qat=False,
			is_dynamic=False,
			act_qmin=act_qmin,
			act_qmax=act_qmax,
			weight_qmin=weight_qmin,
			weight_qmax=weight_qmax,
		)
	)

	prepared_model = prepare_pt2e(graph_module, quantizer)
	calibrate_model(
		prepared_model=prepared_model,
		calibration_loader=calibration_loader,
		max_calibration_batches=max_calibration_batches,
		norm_stats=norm_stats,
	)
	quantized_model = convert_pt2e(prepared_model)
	tq.move_exported_model_to_eval(quantized_model)
	return quantized_model


def maybe_export_pte(
	model: nn.Module,
	example_input: torch.Tensor,
	output_path: Path,
) -> None:
	try:
		from executorch.exir import to_edge

		exported_program = export(model, (example_input,))
		edge_program = to_edge(exported_program)
		executorch_program = edge_program.to_executorch()
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_bytes(executorch_program.buffer)
		print(f"Saved ExecuTorch .pte: {output_path}")
	except FileNotFoundError as error:
		print(
			"Skipping .pte export because required tool is missing "
			f"({error}). Install `flatc` to enable `.pte` export."
		)
	except Exception as error:
		print(f"Skipping .pte export due to error: {error}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="PT2E post-training quantization to INT8")
	parser.add_argument(
		"--root-path",
		type=Path,
		default=Path(__file__).resolve().parent / "uci-har",
		help="Path to UCI-HAR dataset root",
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=Path(__file__).resolve().parent / "models" / "best_model_subject1_val.pth",
		help="Path to FP32 trained checkpoint",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "models",
		help="Directory for quantized artifacts",
	)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--max-calibration-batches", type=int, default=100)
	parser.add_argument(
		"--calibration-shuffle",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Shuffle calibration dataloader (default: true)",
	)
	parser.add_argument(
		"--calibration-samples",
		type=int,
		default=0,
		help="If >0, use only this many calibration samples",
	)
	parser.add_argument(
		"--balanced-calibration",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="When using --calibration-samples, sample class-balanced calibration data",
	)
	parser.add_argument(
		"--per-channel",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Use per-channel quantization for weights (default: true)",
	)
	parser.add_argument("--act-qmin", type=int, default=-128)
	parser.add_argument("--act-qmax", type=int, default=127)
	parser.add_argument("--weight-qmin", type=int, default=-127)
	parser.add_argument("--weight-qmax", type=int, default=127)
	parser.add_argument(
		"--minmax-normalize",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Apply per-channel min-max normalization using calibration data stats",
	)
	parser.add_argument("--use-gyro", action="store_true", help="Use accel + gyro channels (6ch)")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--export-pte",
		action="store_true",
		help="Also export ExecuTorch .pte (requires flatc)",
	)
	parser.add_argument(
		"--metrics-file",
		type=Path,
		default=Path("quantization_metrics.json"),
		help="Metrics output file (relative to output-dir if not absolute)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	root_path = args.root_path
	subject_train_path = root_path / "train" / "subject_train.txt"
	all_train_subjects = sorted(np.unique(np.loadtxt(subject_train_path, dtype=int)).tolist())

	subject_test_path = root_path / "test" / "subject_test.txt"
	all_test_subjects = sorted(np.unique(np.loadtxt(subject_test_path, dtype=int)).tolist())

	state_dict = torch.load(args.checkpoint, map_location="cpu")
	detected_channels = int(state_dict["bn0.weight"].shape[0])
	if args.use_gyro and detected_channels != 6:
		raise ValueError(
			f"--use-gyro was set but checkpoint appears to use {detected_channels} channels."
		)

	use_gyro = args.use_gyro or detected_channels == 6
	if detected_channels not in (3, 6):
		raise ValueError(f"Unsupported input channels inferred from checkpoint: {detected_channels}")

	calibration_loader, test_loader = build_dataloaders(
		root_path=root_path,
		train_subjects=all_train_subjects,
		test_subjects=all_test_subjects,
		use_gyro=use_gyro,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		calibration_shuffle=args.calibration_shuffle,
		calibration_samples=args.calibration_samples,
		balanced_calibration=args.balanced_calibration,
		seed=args.seed,
	)

	model = SeparableConvCNN(num_channels=detected_channels)
	model.load_state_dict(state_dict)
	model.eval()

	norm_stats = None
	if args.minmax_normalize:
		norm_stats = compute_minmax_stats(calibration_loader)

	example_input = next(iter(calibration_loader))[0][:1].contiguous()
	if norm_stats is not None:
		min_vals, max_vals = norm_stats
		example_input = apply_minmax_normalization(example_input, min_vals, max_vals)

	fp32_metrics = evaluate_model(
		model=model,
		dataloader=test_loader,
		device=torch.device("cpu"),
		norm_stats=norm_stats,
	)
	quantized_model = quantize_pt2e_full_int8(
		fp32_model=model,
		example_input=example_input,
		calibration_loader=calibration_loader,
		max_calibration_batches=args.max_calibration_batches,
		is_per_channel=args.per_channel,
		act_qmin=args.act_qmin,
		act_qmax=args.act_qmax,
		weight_qmin=args.weight_qmin,
		weight_qmax=args.weight_qmax,
		norm_stats=norm_stats,
	)
	int8_metrics = evaluate_model(
		model=quantized_model,
		dataloader=test_loader,
		device=torch.device("cpu"),
		force_singleton=True,
		norm_stats=norm_stats,
	)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	quantized_state_dict_path = args.output_dir / "best_model_subject1_val_pt2e_int8_state_dict.pth"
	quantized_exported_program_path = args.output_dir / "best_model_subject1_val_pt2e_int8_exported.pt2"
	metrics_file_path = (
		args.metrics_file
		if args.metrics_file.is_absolute()
		else (args.output_dir / args.metrics_file)
	)

	torch.save(quantized_model.state_dict(), quantized_state_dict_path)
	quantized_exported_program = export(quantized_model, (example_input,))
	torch.export.save(quantized_exported_program, str(quantized_exported_program_path))

	accuracy_drop = fp32_metrics["acc"] - int8_metrics["acc"]
	metrics_payload = {
		"fp32_test_loss": fp32_metrics["loss"],
		"fp32_test_acc": fp32_metrics["acc"],
		"int8_test_loss": int8_metrics["loss"],
		"int8_test_acc": int8_metrics["acc"],
		"accuracy_drop": accuracy_drop,
		"checkpoint": str(args.checkpoint),
		"use_gyro": use_gyro,
		"calibration_batches": args.max_calibration_batches,
		"calibration_batch_size": args.batch_size,
		"calibration_shuffle": args.calibration_shuffle,
		"calibration_samples": args.calibration_samples,
		"balanced_calibration": args.balanced_calibration,
		"seed": args.seed,
		"per_channel": args.per_channel,
		"act_qmin": args.act_qmin,
		"act_qmax": args.act_qmax,
		"weight_qmin": args.weight_qmin,
		"weight_qmax": args.weight_qmax,
		"minmax_normalize": args.minmax_normalize,
	}
	metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
	metrics_file_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

	print("=" * 60)
	print("PT2E Post-Training Quantization Summary")
	print(f"Checkpoint: {args.checkpoint}")
	print(f"Use gyro: {use_gyro}")
	print(f"Calibration samples: up to {args.max_calibration_batches * args.batch_size}")
	if args.calibration_samples > 0:
		print(
			"Calibration subset: "
			f"samples={args.calibration_samples}, balanced={args.balanced_calibration}"
		)
	print(
		"Quant config: "
		f"per_channel={args.per_channel}, "
		f"act=[{args.act_qmin},{args.act_qmax}], "
		f"weight=[{args.weight_qmin},{args.weight_qmax}]"
	)
	print(f"Min-max normalization: {args.minmax_normalize}")
	print(f"FP32 Test Loss: {fp32_metrics['loss']:.4f} | FP32 Test Acc: {fp32_metrics['acc']:.2f}%")
	print(f"INT8 Test Loss: {int8_metrics['loss']:.4f} | INT8 Test Acc: {int8_metrics['acc']:.2f}%")
	print(f"Accuracy Drop (FP32-INT8): {accuracy_drop:.2f}%")
	print(f"Saved state_dict: {quantized_state_dict_path}")
	print(f"Saved exported program: {quantized_exported_program_path}")
	print(f"Saved metrics: {metrics_file_path}")
	print(
		"Note: PT2E INT8 uses int8 activations/weights in kernels; "
		"bias is accumulated in higher precision by backend kernels."
	)

	if args.export_pte:
		maybe_export_pte(
			model=quantized_model,
			example_input=example_input,
			output_path=args.output_dir / "best_model_subject1_val_pt2e_int8.pte",
		)


if __name__ == "__main__":
	main()
