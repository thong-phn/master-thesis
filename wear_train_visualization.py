"""
Visualization utilities for WEAR two-stage channel-pruning model.

This script generates three figures for a trained stage-2 checkpoint:
1) Hard channel masks per block (block2, block3, block4)
2) Pre-mask activation profile: kept channels vs pruned channels
3) Input saliency heatmap (mean |d logit / d input|)

Example:
	python -m lib.wear_train_visualization \
		--checkpoint models/wear_best_model_two_stage_channel_subject0_val.pth \
		--preprocessing fft \
		--split test \
		--output_dir figures/channel_pruning_vis_subject0
"""

from pathlib import Path
import argparse
import random
import os
import importlib
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


def collect_pre_mask_activation_profiles(model, dataloader, device, hard_masks, max_batches: int):
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

	# Sum over batches then normalize for stable profile estimate.
	profile_sums = {
		"block2": {"kept": None, "pruned": None, "count": 0},
		"block3": {"kept": None, "pruned": None, "count": 0},
		"block4": {"kept": None, "pruned": None, "count": 0},
	}

	try:
		with torch.no_grad():
			for batch_idx, (x, _) in enumerate(dataloader):
				if batch_idx >= max_batches:
					break

				x = x.to(device)
				_ = model(x)

				for block_name in ["block2", "block3", "block4"]:
					a = activations[block_name]  # [B, C, T]
					mask = hard_masks[block_name]
					keep_idx = np.where(mask == 1)[0]
					prune_idx = np.where(mask == 0)[0]

					# Mean absolute activation profile over channels and batch.
					if len(keep_idx) > 0:
						kept_profile = a[:, keep_idx, :].abs().mean(dim=(0, 1)).numpy()
					else:
						kept_profile = np.zeros(a.shape[-1], dtype=np.float32)

					if len(prune_idx) > 0:
						pruned_profile = a[:, prune_idx, :].abs().mean(dim=(0, 1)).numpy()
					else:
						pruned_profile = np.zeros(a.shape[-1], dtype=np.float32)

					if profile_sums[block_name]["kept"] is None:
						profile_sums[block_name]["kept"] = kept_profile
						profile_sums[block_name]["pruned"] = pruned_profile
					else:
						profile_sums[block_name]["kept"] += kept_profile
						profile_sums[block_name]["pruned"] += pruned_profile
					profile_sums[block_name]["count"] += 1
	finally:
		for h in hooks:
			h.remove()

	profiles = {}
	for block_name, data in profile_sums.items():
		count = max(data["count"], 1)
		profiles[block_name] = {
			"kept": data["kept"] / count,
			"pruned": data["pruned"] / count,
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
		colors = np.where(mask > 0.5, "#2E7D32", "#D32F2F")
		ax.bar(x, mask, color=colors, width=1.0)
		kept = int(mask.sum())
		total = len(mask)
		ax.set_ylim(0, 1.05)
		ax.set_ylabel("Mask")
		ax.set_title(f"{block_name}: kept {kept}/{total}, pruned {total - kept}/{total}")

	axes[-1].set_xlabel("Channel index")
	fig.suptitle("Figure 1: Hard Channel Masks (Stage 2)")
	fig.savefig(out_path, dpi=220)
	plt.close(fig)


def plot_activation_profiles(profiles, preprocessing: str, out_path: Path):
	x_label = "Frequency bin" if preprocessing in {"fft", "dct", "ihw"} else "Time index"
	fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
	block_order = ["block2", "block3", "block4"]

	for ax, block_name in zip(axes, block_order):
		kept = profiles[block_name]["kept"]
		pruned = profiles[block_name]["pruned"]
		x = np.arange(len(kept))

		ax.plot(x, kept, color="#1565C0", linewidth=2, label="Kept channels")
		ax.plot(x, pruned, color="#EF6C00", linewidth=2, label="Pruned channels")
		ax.set_title(f"{block_name}: pre-mask mean |activation| profile")
		ax.set_ylabel("Activation")
		ax.grid(alpha=0.25)
		ax.legend(loc="upper right")

	axes[-1].set_xlabel(x_label)
	fig.suptitle("Figure 2: Kept vs Pruned Channels Capture (Pre-mask Activation)")
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


def parse_args():
	parser = argparse.ArgumentParser(description="Visualize kept/pruned channel behavior for WEAR stage-2 model")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to stage-2 checkpoint (.pth)")
	parser.add_argument("--root_path", type=str, default=None, help="Path to WEAR dataset root (contains train/ and test/)")
	parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split for analysis")
	parser.add_argument("--preprocessing", type=str, default="fft", choices=["fft", "dct", "ihw", "no"])
	parser.add_argument("--subject_ids", type=int, nargs="*", default=None, help="Optional subject IDs to analyze (for train split)")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--max_batches_activation", type=int, default=20, help="Number of batches used for activation profiles")
	parser.add_argument("--max_batches_saliency", type=int, default=10, help="Number of batches used for saliency")
	parser.add_argument("--dropout", type=float, default=0.4)
	parser.add_argument("--tau_start", type=float, default=10.0)
	parser.add_argument("--tau_end", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output_dir", type=str, default="./figures/")
	return parser.parse_args()

def main():
	args = parse_args()
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
	if len(dataset) == 0:
		raise RuntimeError("Dataset is empty for the selected split/subject_ids.")

	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
	freq_bins = dataset[0][0].shape[-1]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	print(f"Dataset size: {len(dataset)} samples")
	print(f"Checkpoint: {checkpoint_path}")

	model = build_model(
		checkpoint_path=checkpoint_path,
		freq_bins=freq_bins,
		dropout=args.dropout,
		tau_start=args.tau_start,
		tau_end=args.tau_end,
		device=device,
	)

	hard_masks = get_hard_masks_numpy(model)
	profiles = collect_pre_mask_activation_profiles(
		model=model,
		dataloader=dataloader,
		device=device,
		hard_masks=hard_masks,
		max_batches=args.max_batches_activation,
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
	fig2 = output_dir / "02_activation_kept_vs_pruned.png"
	fig3 = output_dir / "03_input_saliency.png"

	plot_mask_overview(hard_masks, fig1)
	plot_activation_profiles(profiles, args.preprocessing, fig2)
	plot_saliency(input_map, saliency_map, args.preprocessing, fig3)

	print("Saved figures:")
	print(f"  - {fig1}")
	print(f"  - {fig2}")
	print(f"  - {fig3}")


if __name__ == "__main__":
	main()
