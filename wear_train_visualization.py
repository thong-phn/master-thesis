"""
Visualization utilities for WEAR two-stage channel-pruning model.

This script generates five figures for a trained stage-2 checkpoint:
1) Hard channel masks per block (block2, block3, block4)
2) Logit-drop channel importance (ablation on pre-mask feature maps)
3) Prototype input samples (high-activation examples for kept/pruned groups)
4) Class-conditional activation heatmaps (kept vs pruned)
5) Input saliency heatmap (mean |d logit / d input|)

Example:
	python -m lib.wear_train_visualization \
		--checkpoint models/wear_best_model_two_stage_channel_subject0_val.pth \
		--preprocessing fft \
		--split test \
		--output_dir figures/channel_pruning_vis_subject0
"""

from pathlib import Path
import argparse
import os
from contextlib import contextmanager
from types import MethodType
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.model import GumbelChannelPruningCNN
from lib.wear_train import WEAR_Dataset
from wear_main_loso_baseline import set_seed

def resolve_paths(root_path_arg: str | None):
	if root_path_arg is not None:
		project_root = Path(__file__).resolve().parent.parent
		root_path = Path(root_path_arg).expanduser().resolve()
		return project_root, root_path

	if os.path.exists("/kaggle/input"):
		root_path = Path("/kaggle/input/datasets/thongp/wearthesis/wear")
		project_root = Path("/kaggle/working")
	else:
		project_root = Path(__file__).resolve().parent.parent
		root_path = project_root / "wear"
	return project_root, root_path


def build_model(checkpoint_path: Path, freq_bins: int, dropout: float, tau_start: float, tau_end: float, device):
	model = GumbelChannelPruningCNN(
		num_classes=8,
		num_channels=6,
		freq_bins=freq_bins,
		dropout=dropout,
		tau_start=tau_start,
		tau_end=tau_end,
	).to(device)

	state = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(state)
	model.eval()
	return model


@torch.no_grad()
def get_hard_masks_numpy(model: GumbelChannelPruningCNN):
	hard_masks = model.get_hard_masks()
	return {
		"block2": hard_masks["block2"].detach().cpu().numpy(),
		"block3": hard_masks["block3"].detach().cpu().numpy(),
		"block4": hard_masks["block4"].detach().cpu().numpy(),
	}

def _get_block_bn_layer(model: GumbelChannelPruningCNN, block_name: str):
	if block_name == "block2":
		return model.bn2
	if block_name == "block3":
		return model.bn3
	if block_name == "block4":
		return model.bn4
	raise ValueError(f"Unsupported block name: {block_name}")


@contextmanager
def _temporary_full_channel_masks(model: GumbelChannelPruningCNN):
	"""Temporarily force all pruning masks to 1 so ablations are measured from a dense baseline."""
	original_get_channel_mask = model._get_channel_mask

	def _all_ones_mask(self, logits, batch_size):
		return torch.ones(batch_size, logits.size(0), device=logits.device, dtype=logits.dtype)

	model._get_channel_mask = MethodType(_all_ones_mask, model)
	try:
		yield
	finally:
		model._get_channel_mask = original_get_channel_mask


def collect_logit_drop_importance(model, dataloader, device, max_batches: int):
	"""Measure channel importance by zeroing one channel at a time from a dense (all-mask-on) baseline.

	This is the right setting if the goal is to decide which channels should be pruned from the start,
	because the reference pass is not biased by the model's current learned hard mask.
	"""
	block_channels = {"block2": 64, "block3": 128, "block4": 128}
	importance_sums = {k: np.zeros(c, dtype=np.float64) for k, c in block_channels.items()}
	processed_batches = 0

	with torch.no_grad():
		with _temporary_full_channel_masks(model):
			for batch_idx, (x, _) in enumerate(dataloader):
				if batch_idx >= max_batches:
					break

				x = x.to(device)
				base_logits = model(x)
				target = base_logits.argmax(dim=1)
				base_selected = base_logits.gather(1, target.unsqueeze(1)).squeeze(1)

				for block_name, channels in block_channels.items():
					bn_layer = _get_block_bn_layer(model, block_name)
					for ch in range(channels):
						def ablate_single_channel(_, __, output, channel_idx=ch):
							masked = output.clone()
							masked[:, channel_idx, :] = 0.0
							return masked

						hook = bn_layer.register_forward_hook(ablate_single_channel)
						ablated_logits = model(x)
						hook.remove()

						ablated_selected = ablated_logits.gather(1, target.unsqueeze(1)).squeeze(1)
						drop = (base_selected - ablated_selected).mean().item()
						importance_sums[block_name][ch] += drop

				processed_batches += 1

	processed_batches = max(processed_batches, 1)
	return {k: (v / processed_batches).astype(np.float32) for k, v in importance_sums.items()}


@contextmanager
def _temporary_intervention_masks(model: GumbelChannelPruningCNN, intervention_masks):
	"""Apply fixed channel masks via BN hooks while model masks are forced to all ones."""

	hooks = []
	for block_name in ["block2", "block3", "block4"]:
		bn_layer = _get_block_bn_layer(model, block_name)
		mask_np = intervention_masks[block_name].astype(np.float32)

		def _mask_hook(_, __, output, mask_values=mask_np):
			mask_t = torch.as_tensor(mask_values, device=output.device, dtype=output.dtype).view(1, -1, 1)
			return output * mask_t

		hooks.append(bn_layer.register_forward_hook(_mask_hook))

	try:
		yield
	finally:
		for h in hooks:
			h.remove()


def _evaluate_with_intervention_masks(model, dataloader, device, intervention_masks, max_batches: int):
	correct = 0
	total = 0
	selected_logit_sum = 0.0

	with torch.no_grad():
		with _temporary_full_channel_masks(model):
			with _temporary_intervention_masks(model, intervention_masks):
				for batch_idx, (x, y) in enumerate(dataloader):
					if batch_idx >= max_batches:
						break

					x = x.to(device)
					y = y.to(device)
					logits = model(x)
					pred = logits.argmax(dim=1)
					selected = logits.gather(1, pred.unsqueeze(1)).squeeze(1)

					correct += (pred == y).sum().item()
					total += y.size(0)
					selected_logit_sum += selected.sum().item()

	total = max(total, 1)
	return {
		"accuracy": float(correct / total),
		"mean_selected_logit": float(selected_logit_sum / total),
		"num_samples": int(total),
	}


def collect_prune_vs_keep_intervention(
	model,
	dataloader,
	device,
	hard_masks,
	max_batches: int,
	random_trials: int,
	seed: int,
):
	"""Compare dense baseline vs pruning mask=0 channels vs pruning kept channels with matched counts."""
	rng = np.random.default_rng(seed)
	blocks = ["block2", "block3", "block4"]

	dense_masks = {b: np.ones_like(hard_masks[b], dtype=np.float32) for b in blocks}
	prune_masks = {b: hard_masks[b].astype(np.float32).copy() for b in blocks}

	dense_metrics = _evaluate_with_intervention_masks(
		model=model,
		dataloader=dataloader,
		device=device,
		intervention_masks=dense_masks,
		max_batches=max_batches,
	)
	prune_metrics = _evaluate_with_intervention_masks(
		model=model,
		dataloader=dataloader,
		device=device,
		intervention_masks=prune_masks,
		max_batches=max_batches,
	)

	keep_trial_metrics = []
	for _ in range(max(random_trials, 1)):
		keep_matched_masks = {b: np.ones_like(hard_masks[b], dtype=np.float32) for b in blocks}
		for block_name in blocks:
			pruned_count = int((hard_masks[block_name] == 0).sum())
			kept_idx = np.where(hard_masks[block_name] == 1)[0]
			if pruned_count == 0 or len(kept_idx) == 0:
				continue
			num_to_zero = min(pruned_count, len(kept_idx))
			chosen = rng.choice(kept_idx, size=num_to_zero, replace=False)
			keep_matched_masks[block_name][chosen] = 0.0

		metrics = _evaluate_with_intervention_masks(
			model=model,
			dataloader=dataloader,
			device=device,
			intervention_masks=keep_matched_masks,
			max_batches=max_batches,
		)
		keep_trial_metrics.append(metrics)

	keep_acc = np.array([m["accuracy"] for m in keep_trial_metrics], dtype=np.float32)
	keep_logit = np.array([m["mean_selected_logit"] for m in keep_trial_metrics], dtype=np.float32)

	return {
		"dense": dense_metrics,
		"prune_mask_zero": prune_metrics,
		"keep_matched_random": {
			"accuracy_mean": float(keep_acc.mean()),
			"accuracy_std": float(keep_acc.std(ddof=0)),
			"mean_selected_logit_mean": float(keep_logit.mean()),
			"mean_selected_logit_std": float(keep_logit.std(ddof=0)),
			"trials": int(len(keep_trial_metrics)),
		},
	}


def collect_prototype_samples(
	model,
	dataloader,
	device,
	hard_masks,
	logit_drop_importance,
	max_batches: int,
	block_name: str,
	top_k: int,
	top_channels_per_group: int,
):
	activations = {}

	def save_activation(name):
		def hook_fn(_, __, output):
			activations[name] = output.detach().cpu()

		return hook_fn

	block_bn = _get_block_bn_layer(model, block_name)
	hook = block_bn.register_forward_hook(save_activation(block_name))

	mask = hard_masks[block_name]
	importance = logit_drop_importance[block_name]
	keep_idx = np.where(mask == 1)[0]
	prune_idx = np.where(mask == 0)[0]

	def pick_top_channels(group_idx):
		if len(group_idx) == 0:
			return np.array([], dtype=np.int64)
		group_importance = importance[group_idx]
		order = np.argsort(-group_importance)
		return group_idx[order[: min(top_channels_per_group, len(group_idx))]]

	selected_channels = {
		"kept": pick_top_channels(keep_idx),
		"pruned": pick_top_channels(prune_idx),
	}

	candidates = {"kept": [], "pruned": []}

	try:
		with torch.no_grad():
			for batch_idx, (x, y) in enumerate(dataloader):
				if batch_idx >= max_batches:
					break

				x_device = x.to(device)
				_ = model(x_device)
				a = activations[block_name]  # [B, C, T]

				x_cpu = x.detach().cpu().numpy()
				y_cpu = y.detach().cpu().numpy()

				for group_name in ["kept", "pruned"]:
					ch_idx = selected_channels[group_name]
					if len(ch_idx) == 0:
						scores = np.zeros(a.shape[0], dtype=np.float32)
					else:
						scores = a[:, ch_idx, :].abs().mean(dim=(1, 2)).numpy()

					for i in range(a.shape[0]):
						candidates[group_name].append((float(scores[i]), x_cpu[i], int(y_cpu[i])))
	finally:
		hook.remove()

	prototypes = {}
	for group_name in ["kept", "pruned"]:
		candidates[group_name].sort(key=lambda t: t[0], reverse=True)
		prototypes[group_name] = candidates[group_name][:top_k]

	return {
		"block": block_name,
		"selected_channels": selected_channels,
		"prototypes": prototypes,
	}


def collect_class_conditional_profiles(model, dataloader, device, hard_masks, num_classes: int, max_batches: int):
	activations = {}

	def save_activation(name):
		def hook_fn(_, __, output):
			activations[name] = output.detach().cpu()

		return hook_fn

	hooks = [
		model.bn2.register_forward_hook(save_activation("block2")),
		model.bn3.register_forward_hook(save_activation("block3")),
		model.bn4.register_forward_hook(save_activation("block4")),
	]

	stats = {}
	for block_name in ["block2", "block3", "block4"]:
		stats[block_name] = {
			"kept_sum": None,
			"pruned_sum": None,
			"counts": np.zeros(num_classes, dtype=np.int64),
		}

	try:
		with torch.no_grad():
			for batch_idx, (x, y) in enumerate(dataloader):
				if batch_idx >= max_batches:
					break

				x = x.to(device)
				_ = model(x)
				y_cpu = y.detach().cpu().numpy()

				for block_name in ["block2", "block3", "block4"]:
					a = activations[block_name]  # [B, C, T]
					mask = hard_masks[block_name]
					keep_idx = np.where(mask == 1)[0]
					prune_idx = np.where(mask == 0)[0]

					if stats[block_name]["kept_sum"] is None:
						steps = a.shape[-1]
						stats[block_name]["kept_sum"] = np.zeros((num_classes, steps), dtype=np.float64)
						stats[block_name]["pruned_sum"] = np.zeros((num_classes, steps), dtype=np.float64)

					for i in range(a.shape[0]):
						cls = int(y_cpu[i])
						if cls < 0 or cls >= num_classes:
							continue

						if len(keep_idx) > 0:
							kept_profile = a[i, keep_idx, :].abs().mean(dim=0).numpy()
						else:
							kept_profile = np.zeros(a.shape[-1], dtype=np.float32)

						if len(prune_idx) > 0:
							pruned_profile = a[i, prune_idx, :].abs().mean(dim=0).numpy()
						else:
							pruned_profile = np.zeros(a.shape[-1], dtype=np.float32)

						stats[block_name]["kept_sum"][cls] += kept_profile
						stats[block_name]["pruned_sum"][cls] += pruned_profile
						stats[block_name]["counts"][cls] += 1
	finally:
		for h in hooks:
			h.remove()

	profiles = {}
	for block_name, block_stats in stats.items():
		counts = block_stats["counts"]
		den = np.maximum(counts[:, None], 1)
		profiles[block_name] = {
			"kept": (block_stats["kept_sum"] / den).astype(np.float32),
			"pruned": (block_stats["pruned_sum"] / den).astype(np.float32),
			"counts": counts,
		}

	return profiles


def collect_input_saliency(model, dataloader, device, max_batches: int):
	grad_sum = None
	input_sum = None
	total_samples = 0

	for batch_idx, (x, _) in enumerate(dataloader):
		if batch_idx >= max_batches:
			break

		x = x.to(device)
		x.requires_grad_(True)

		model.zero_grad(set_to_none=True)
		logits = model(x)
		target = logits.argmax(dim=1)
		selected = logits.gather(1, target.unsqueeze(1)).sum()
		selected.backward()

		grads = x.grad.detach().abs().cpu()  # [B, 6, T]
		inputs = x.detach().abs().cpu()

		bsz = x.size(0)
		if grad_sum is None:
			grad_sum = grads.sum(dim=0)
			input_sum = inputs.sum(dim=0)
		else:
			grad_sum += grads.sum(dim=0)
			input_sum += inputs.sum(dim=0)
		total_samples += bsz

	total_samples = max(total_samples, 1)
	saliency_map = (grad_sum / total_samples).numpy()
	input_map = (input_sum / total_samples).numpy()
	return input_map, saliency_map


def plot_mask_overview(hard_masks, out_path: Path):
	fig, axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)
	block_order = ["block2", "block3", "block4"]

	for ax, block_name in zip(axes, block_order):
		mask = hard_masks[block_name]
		x = np.arange(len(mask))
		colors = np.where(mask > 0.5, "#FFC444", "#D32F2F")
		ax.bar(x, mask, color=colors, width=1.0)
		ax.set_xticks(np.arange(0, len(mask), 5))
		kept = int(mask.sum())
		total = len(mask)
		ax.set_ylim(0, 1.05)
		ax.set_yticks([0, 1])
		ax.set_ylabel("Mask")
		ax.set_title(f"{block_name}: kept {kept}/{total}, pruned {total - kept}/{total}")

	axes[-1].set_xlabel("Channel index")
	fig.suptitle("Figure 1: Channel Masks")
	fig.savefig(out_path, dpi=220)
	plt.close(fig)


def plot_logit_drop_importance(logit_drop, hard_masks, out_path: Path):
	fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
	block_order = ["block2", "block3", "block4"]

	for ax, block_name in zip(axes, block_order):
		importance = logit_drop[block_name]
		mask = hard_masks[block_name]
		x = np.arange(len(importance))
		colors = np.where(mask > 0.5, "#FFC444", "#9E9E9E")
		ax.set_xticks(np.arange(0, len(mask), 5))
		ax.bar(x, importance, color=colors, width=1.0)
		ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
		ax.set_title(f"{block_name}: dense-baseline logit drop")
		ax.set_ylabel("Delta logit")
		ax.grid(axis="y", alpha=0.2)

	axes[-1].set_xlabel("Channel index")
	fig.suptitle("Figure 2: Logit-Drop Channel Importance")
	fig.savefig(out_path, dpi=220)
	plt.close(fig)


def plot_prune_vs_keep_intervention(intervention_result, out_path: Path):
	dense_acc = intervention_result["dense"]["accuracy"]
	prune_acc = intervention_result["prune_mask_zero"]["accuracy"]
	keep_acc_mean = intervention_result["keep_matched_random"]["accuracy_mean"]
	keep_acc_std = intervention_result["keep_matched_random"]["accuracy_std"]

	dense_logit = intervention_result["dense"]["mean_selected_logit"]
	prune_logit = intervention_result["prune_mask_zero"]["mean_selected_logit"]
	keep_logit_mean = intervention_result["keep_matched_random"]["mean_selected_logit_mean"]
	keep_logit_std = intervention_result["keep_matched_random"]["mean_selected_logit_std"]

	labels = ["Dense", "Prune mask=0", "Prune kept (matched)"]
	x = np.arange(len(labels))

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

	acc_values = [dense_acc, prune_acc, keep_acc_mean]
	acc_errors = [0.0, 0.0, keep_acc_std]
	axes[0].bar(x, acc_values, yerr=acc_errors, capsize=4, color=["#1f77b4", "#2ca02c", "#d62728"])
	axes[0].set_xticks(x)
	axes[0].set_xticklabels(labels, rotation=10)
	axes[0].set_ylabel("Accuracy")
	axes[0].set_title("Intervention accuracy")
	axes[0].grid(axis="y", alpha=0.2)

	logit_values = [dense_logit, prune_logit, keep_logit_mean]
	logit_errors = [0.0, 0.0, keep_logit_std]
	axes[1].bar(x, logit_values, yerr=logit_errors, capsize=4, color=["#1f77b4", "#2ca02c", "#d62728"])
	axes[1].set_xticks(x)
	axes[1].set_xticklabels(labels, rotation=10)
	axes[1].set_ylabel("Mean selected logit")
	axes[1].set_title("Intervention confidence")
	axes[1].grid(axis="y", alpha=0.2)

	fig.suptitle("Figure 3: Prune-vs-Keep Intervention Test")
	fig.savefig(out_path, dpi=220)
	plt.close(fig)


def plot_prototype_samples(prototype_result, preprocessing: str, out_path: Path):
	x_label = "Frequency bin" if preprocessing in {"fft", "dct", "ihw"} else "Time index"
	top_k = max(len(prototype_result["prototypes"]["kept"]), len(prototype_result["prototypes"]["pruned"]))
	top_k = max(top_k, 1)
	fig, axes = plt.subplots(2, top_k, figsize=(3.4 * top_k, 7), constrained_layout=True)
	if top_k == 1:
		axes = np.array([[axes[0]], [axes[1]]])

	block_name = prototype_result["block"]
	for row_idx, group_name in enumerate(["kept", "pruned"]):
		group_items = prototype_result["prototypes"][group_name]
		selected_channels = prototype_result["selected_channels"][group_name]
		for col_idx in range(top_k):
			ax = axes[row_idx, col_idx]
			if col_idx >= len(group_items):
				ax.axis("off")
				continue

			score, sample_input, label = group_items[col_idx]
			im = ax.imshow(sample_input, aspect="auto", cmap="viridis", origin="lower")
			ax.set_title(f"{group_name} #{col_idx + 1}: score={score:.3f}, y={label}")
			ax.set_xlabel(x_label)
			ax.set_ylabel("Input ch")
			fig.colorbar(im, ax=ax, fraction=0.046)

		if len(selected_channels) > 0:
			ch_text = ",".join(str(int(c)) for c in selected_channels[:10])
		else:
			ch_text = "none"
		axes[row_idx, 0].text(
			0.0,
			1.08,
			f"{group_name.upper()} selected channels ({block_name}): {ch_text}",
			transform=axes[row_idx, 0].transAxes,
			fontsize=10,
		)

	fig.suptitle("Figure 3: Prototype Input Samples (High Activation)")
	fig.savefig(out_path, dpi=220)
	plt.close(fig)


def plot_class_conditional_profiles(class_profiles, preprocessing: str, out_path: Path):
	x_label = "Frequency bin" if preprocessing in {"fft", "dct", "ihw"} else "Time index"
	fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
	block_order = ["block2", "block3", "block4"]

	for row_idx, block_name in enumerate(block_order):
		for col_idx, group_name in enumerate(["kept", "pruned"]):
			ax = axes[row_idx, col_idx]
			heat = class_profiles[block_name][group_name]
			im = ax.imshow(heat, aspect="auto", cmap="magma", origin="lower")
			ax.set_title(f"{block_name} | {group_name}")
			ax.set_xlabel(x_label)
			ax.set_ylabel("Class index")
			fig.colorbar(im, ax=ax, fraction=0.046)

	fig.suptitle("Figure 4: Class-Conditional Mean |Activation| (Pre-mask)")
	fig.savefig(out_path, dpi=220)
	plt.close(fig)


def plot_saliency(input_map, saliency_map, preprocessing: str, out_path: Path):
	x_label = "Frequency bin" if preprocessing in {"fft", "dct", "ihw"} else "Time index"
	fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

	im0 = axes[0].imshow(input_map, aspect="auto", cmap="Blues", origin="lower")
	axes[0].set_title("Mean |input|")
	axes[0].set_xlabel(x_label)
	axes[0].set_ylabel("Input channel (sensor)")
	fig.colorbar(im0, ax=axes[0], fraction=0.046)

	im1 = axes[1].imshow(saliency_map, aspect="auto", cmap="magma", origin="lower")
	axes[1].set_title("Mean |d logit / d input| (saliency)")
	axes[1].set_xlabel(x_label)
	axes[1].set_ylabel("Input channel (sensor)")
	fig.colorbar(im1, ax=axes[1], fraction=0.046)

	fig.suptitle("Figure 3: Input Attribution Map")
	fig.savefig(out_path, dpi=220)
	plt.close(fig)	

def main():
	parser = argparse.ArgumentParser(description="Visualize kept/pruned channel behavior for WEAR stage-2 model")
	parser.add_argument("--checkpoint", type=str, required=True, help="Stage-2 checkpoint (.pth)")
	parser.add_argument("--root_path", type=str, default='/home/qphan/master-thesis/wear', help="Path to WEAR dataset root (contains train/ and test/)")
	parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
	parser.add_argument("--preprocessing", type=str, default="fft", choices=["fft", "dct", "ihw", "no"])
	parser.add_argument("--subject_ids", type=int, nargs="*", default=None) # Default Subject 0
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--max_batches_logit_drop", type=int, default=4, help="Number of batches used for logit-drop importance")
	parser.add_argument("--max_batches_intervention", type=int, default=20, help="Number of batches used for prune-vs-keep intervention test")
	parser.add_argument("--intervention_random_trials", type=int, default=20, help="Random trials for prune-kept matched-count intervention")
	parser.add_argument("--max_batches_prototype", type=int, default=20, help="Number of batches used for prototype mining")
	parser.add_argument("--max_batches_class_conditional", type=int, default=20, help="Number of batches used for class-conditional activation")
	parser.add_argument("--max_batches_saliency", type=int, default=10, help="Number of batches used for saliency")
	parser.add_argument("--prototype_block", type=str, default="block3", choices=["block2", "block3", "block4"])
	parser.add_argument("--prototype_top_k", type=int, default=5)
	parser.add_argument("--prototype_top_channels", type=int, default=8)
	parser.add_argument("--dropout", type=float, default=0.4)
	parser.add_argument("--tau_start", type=float, default=10.0)
	parser.add_argument("--tau_end", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output_dir", type=str, default="/home/qphan/master-thesis/fig")
	args = parser.parse_args()

	set_seed(args.seed)

	project_root, root_path = resolve_paths(args.root_path)
	checkpoint_path = Path(args.checkpoint).expanduser()
	if not checkpoint_path.is_absolute():
		checkpoint_path = (project_root / checkpoint_path).resolve()

	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	dataset = WEAR_Dataset(
		root_path=root_path,
		split=args.split,
		subject_ids=args.subject_ids,
		preprocessing=args.preprocessing,
	)

	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
	freq_bins = dataset[0][0].shape[-1]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	print(f"Dataset size: {len(dataset)} samples")
	print(f"Pretrained model: {checkpoint_path}")

	model = build_model(
		checkpoint_path=checkpoint_path,
		freq_bins=freq_bins,
		dropout=args.dropout,
		tau_start=args.tau_start,
		tau_end=args.tau_end,
		device=device,
	)

	hard_masks = get_hard_masks_numpy(model)
	logit_drop = collect_logit_drop_importance(
		model=model,
		dataloader=dataloader,
		device=device,
		max_batches=args.max_batches_logit_drop,
	)
	intervention_result = collect_prune_vs_keep_intervention(
		model=model,
		dataloader=dataloader,
		device=device,
		hard_masks=hard_masks,
		max_batches=args.max_batches_intervention,
		random_trials=args.intervention_random_trials,
		seed=args.seed,
	)
	prototype_result = collect_prototype_samples(
		model=model,
		dataloader=dataloader,
		device=device,
		hard_masks=hard_masks,
		logit_drop_importance=logit_drop,
		max_batches=args.max_batches_prototype,
		block_name=args.prototype_block,
		top_k=args.prototype_top_k,
		top_channels_per_group=args.prototype_top_channels,
	)
	class_profiles = collect_class_conditional_profiles(
		model=model,
		dataloader=dataloader,
		device=device,
		hard_masks=hard_masks,
		num_classes=model.fc2.out_features,
		max_batches=args.max_batches_class_conditional,
	)
	input_map, saliency_map = collect_input_saliency(
		model=model,
		dataloader=dataloader,
		device=device,
		max_batches=args.max_batches_saliency,
	)

	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = (project_root / output_dir).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	fig1 = output_dir / "01_channel_mask_overview.png"
	fig2 = output_dir / "02_logit_drop_importance.png"
	fig3 = output_dir / "03_prune_vs_keep_intervention.png"
	# fig3 = output_dir / "03_prototype_samples.png"
	# fig4 = output_dir / "04_class_conditional_activation.png"
	# fig5 = output_dir / "05_input_saliency.png"

	plot_mask_overview(hard_masks, fig1)
	plot_logit_drop_importance(logit_drop, hard_masks, fig2)
	plot_prune_vs_keep_intervention(intervention_result, fig3)
	# plot_prototype_samples(prototype_result, args.preprocessing, fig3)
	# plot_class_conditional_profiles(class_profiles, args.preprocessing, fig4)
	# plot_saliency(input_map, saliency_map, args.preprocessing, fig5)

	print("Saved figures:")
	print(f"  - {fig1}")
	print(f"  - {fig2}")
	print(f"  - {fig3}")
	# print(f"  - {fig3}")
	# print(f"  - {fig4}")
	# print(f"  - {fig5}")


if __name__ == "__main__":
	main()
