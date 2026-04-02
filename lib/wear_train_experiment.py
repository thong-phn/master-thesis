import json
import random
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.wear_train import (
	WEAR_Dataset,
	_copy_batchnorm_subset,
	_copy_linear_input_subset,
	_copy_separable_conv_subset,
	_count_parameters,
	_evaluate_classifier,
	_load_stage1_weights_to_channel_gumbel_model,
)

def _to_binary_mask_tensor(values, expected_len, block_name, device):
	mask = torch.as_tensor(values, dtype=torch.float32, device=device).flatten()
	if mask.numel() != expected_len:
		raise ValueError(
			f"Invalid mask length for {block_name}: expected {expected_len}, got {mask.numel()}"
		)

	unique = set(mask.detach().cpu().tolist())
	if not unique.issubset({0.0, 1.0}):
		raise ValueError(f"Mask for {block_name} must be binary (0/1), got values: {sorted(unique)}")

	if int(mask.sum().item()) == 0:
		raise ValueError(f"Mask for {block_name} prunes all channels; at least one channel must be kept.")

	return mask


def _resolve_subject_mask_entry(mask_payload, val_subject):
	if isinstance(mask_payload, dict) and all(k in mask_payload for k in ("block2", "block3", "block4")):
		return mask_payload

	subject_key = str(int(val_subject))
	if not isinstance(mask_payload, dict) or subject_key not in mask_payload:
		raise ValueError(
			f"No mask entry found for validation subject {val_subject}. "
			f"Expected key '{subject_key}' in fixed mask JSON."
		)
	return mask_payload[subject_key]


def _swap_fixed_mask_one_for_one(mask, block_name, rng):
	"""Apply one-for-one swap: turn ON all original OFF channels, then randomly turn OFF
	the same number of channels from the original ON set.
	"""
	mask_cpu = mask.detach().cpu().clone()
	off_indices = torch.where(mask_cpu < 0.5)[0].tolist()
	on_indices = torch.where(mask_cpu > 0.5)[0].tolist()
	off_count = len(off_indices)

	if off_count == 0:
		return mask, {
			"original_on": len(on_indices),
			"original_off": 0,
			"changed_indices": 0,
		}

	if off_count > len(on_indices):
		raise ValueError(
			f"Cannot apply one-for-one fixed-mask swap for {block_name}: "
			f"off_count={off_count} is greater than on_count={len(on_indices)}."
		)

	indices_to_turn_off = rng.sample(on_indices, off_count)
	swapped_mask = torch.ones_like(mask_cpu)
	swapped_mask[indices_to_turn_off] = 0.0
	changed_indices = int((swapped_mask != mask_cpu).sum().item())

	return swapped_mask.to(mask.device), {
		"original_on": len(on_indices),
		"original_off": off_count,
		"changed_indices": changed_indices,
	}


def _load_fixed_masks(fixed_mask_path, val_subject, device, swap_enabled=False, swap_seed=42):
	if fixed_mask_path is None:
		return None

	mask_path = Path(fixed_mask_path).expanduser()
	if not mask_path.exists():
		raise FileNotFoundError(f"Fixed mask file does not exist: {mask_path}")

	with open(mask_path, "r", encoding="utf-8") as f:
		payload = json.load(f)

	subject_mask = _resolve_subject_mask_entry(payload, val_subject)
	required = {"block2": 64, "block3": 128, "block4": 128}
	rng = random.Random(int(swap_seed) + int(val_subject))

	fixed_masks = {}
	for block_name, expected_len in required.items():
		if block_name not in subject_mask:
			raise ValueError(f"Missing '{block_name}' in fixed mask entry for subject {val_subject}.")
		mask = _to_binary_mask_tensor(
			values=subject_mask[block_name],
			expected_len=expected_len,
			block_name=block_name,
			device=device,
		)
		if swap_enabled:
			mask, swap_info = _swap_fixed_mask_one_for_one(mask, block_name, rng)
			print(
				"Fixed-mask one-for-one swap "
				f"[{block_name}] for val subject {val_subject}: "
				f"original_on={swap_info['original_on']}, "
				f"original_off={swap_info['original_off']}, "
				f"changed_indices={swap_info['changed_indices']}"
			)
		fixed_masks[block_name] = mask

	return fixed_masks


@contextmanager
def _temporary_fixed_channel_masks(model, fixed_masks):
	if not fixed_masks:
		yield
		return

	hooks = []

	def _make_mask_hook(mask):
		mask_view = mask.view(1, -1, 1)

		def _hook(_module, _inp, output):
			return output * mask_view

		return _hook

	hooks.append(model.bn2.register_forward_hook(_make_mask_hook(fixed_masks["block2"])))
	hooks.append(model.bn3.register_forward_hook(_make_mask_hook(fixed_masks["block3"])))
	hooks.append(model.bn4.register_forward_hook(_make_mask_hook(fixed_masks["block4"])))

	try:
		yield
	finally:
		for hook in hooks:
			hook.remove()


def _combine_hard_and_fixed_masks(hard_masks, fixed_masks):
	if fixed_masks is None:
		return {k: v.detach().clone() for k, v in hard_masks.items()}

	combined = {}
	for block_name, hard_mask in hard_masks.items():
		fixed_mask = fixed_masks[block_name].to(hard_mask.device)
		combined[block_name] = (hard_mask * fixed_mask).detach()
	return combined


def _set_gumbel_logits_to_allow_all_channels(model):
	"""Bias Gumbel logits toward the ON state for all channels.

	This avoids any accidental collapse if a caller wants the internal pruning masks
	to start from a permissive configuration.
	"""
	with torch.no_grad():
		for attr_name in ("chan_logits_2", "chan_logits_3", "chan_logits_4"):
			logits = getattr(model, attr_name, None)
			if logits is None:
				continue
			logits.zero_()
			logits[:, 0] = -5.0
			logits[:, 1] = 5.0


def _build_pruned_channel_model_from_stage2_with_fixed(stage2_model, fixed_masks, num_classes, dropout, device):
	from lib.model import PrunedSeparableConvCNN

	learned_hard_masks = stage2_model.get_hard_masks()
	effective_masks = (
		{k: v.detach().clone() for k, v in fixed_masks.items()}
		if fixed_masks is not None
		else _combine_hard_and_fixed_masks(learned_hard_masks, fixed_masks)
	)

	keep_indices = {}
	for block_name in ("block2", "block3", "block4"):
		mask = effective_masks[block_name].detach().cpu()
		indices = torch.where(mask > 0.5)[0]
		if indices.numel() == 0:
			raise ValueError(
				f"Effective stage-2 mask for {block_name} pruned all channels; stage 3 cannot proceed."
			)
		keep_indices[block_name] = indices

	pruned_model = PrunedSeparableConvCNN(
		num_classes=num_classes,
		num_channels=6,
		block2_channels=int(keep_indices["block2"].numel()),
		block3_channels=int(keep_indices["block3"].numel()),
		block4_channels=int(keep_indices["block4"].numel()),
		dropout=dropout,
	).to(device)

	with torch.no_grad():
		pruned_model.bn0.load_state_dict(stage2_model.bn0.state_dict())
		_copy_separable_conv_subset(stage2_model.sep_conv1, pruned_model.sep_conv1)
		pruned_model.bn1.load_state_dict(stage2_model.bn1.state_dict())

		_copy_separable_conv_subset(
			stage2_model.sep_conv2,
			pruned_model.sep_conv2,
			keep_in_indices=None,
			keep_out_indices=keep_indices["block2"],
		)
		_copy_batchnorm_subset(stage2_model.bn2, pruned_model.bn2, keep_indices["block2"])

		_copy_separable_conv_subset(
			stage2_model.sep_conv3,
			pruned_model.sep_conv3,
			keep_in_indices=keep_indices["block2"],
			keep_out_indices=keep_indices["block3"],
		)
		_copy_batchnorm_subset(stage2_model.bn3, pruned_model.bn3, keep_indices["block3"])

		_copy_separable_conv_subset(
			stage2_model.sep_conv4,
			pruned_model.sep_conv4,
			keep_in_indices=keep_indices["block3"],
			keep_out_indices=keep_indices["block4"],
		)
		_copy_batchnorm_subset(stage2_model.bn4, pruned_model.bn4, keep_indices["block4"])

		_copy_linear_input_subset(stage2_model.fc1, pruned_model.fc1, keep_indices["block4"])
		pruned_model.fc2.load_state_dict(stage2_model.fc2.state_dict())

	return pruned_model, keep_indices, effective_masks


def train_loso_wear_three_stage_channel(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
	from lib.model import GumbelChannelPruningCNN, SeparableConvCNN

	epochs_stage1 = train_kwargs.get("epochs_stage1", 60)
	epochs_stage2 = train_kwargs.get("epochs_stage2", 60)
	epochs_stage3 = train_kwargs.get("epochs_stage3", 10)
	lr = train_kwargs.get("lr", 1e-3)
	batch_size = train_kwargs.get("batch_size", 64)
	device = train_kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	patience = train_kwargs.get("patience", 10)
	min_delta = train_kwargs.get("min_delta", 1e-3)
	sparsity_weight = train_kwargs.get("sparsity_weight", 0.01)
	stage3_loaded_lr_factor = train_kwargs.get("stage3_loaded_lr_factor", 0.1)
	preprocessing = train_kwargs.get("preprocessing", "fft")
	dropout = train_kwargs.get("dropout", 0.4)
	tau_start = train_kwargs.get("tau_start", 10.0)
	tau_end = train_kwargs.get("tau_end", 1.0)
	fixed_mask_path = train_kwargs.get("fixed_mask_path")
	fixed_mask_swap_enabled = bool(train_kwargs.get("fixed_mask_swap_enabled", False))
	fixed_mask_swap_seed = int(train_kwargs.get("fixed_mask_swap_seed", 42))

	model_path = Path(train_kwargs.get("model_path", "./models/experiments_pruning_channel.pth"))
	model_path.parent.mkdir(parents=True, exist_ok=True)

	stage1_model_path_arg = train_kwargs.get("stage1_model_path")
	stage1_model_path = Path(stage1_model_path_arg).expanduser() if stage1_model_path_arg else model_path.parent / f"{model_path.stem}_stage1.pth"
	stage1_model_path.parent.mkdir(parents=True, exist_ok=True)

	stage2_model_path_arg = train_kwargs.get("stage2_model_path")
	stage2_model_path = Path(stage2_model_path_arg).expanduser() if stage2_model_path_arg else model_path.parent / f"{model_path.stem}_stage2_channel.pth"
	stage2_model_path.parent.mkdir(parents=True, exist_ok=True)

	stage3_model_path_arg = train_kwargs.get("stage3_model_path")
	stage3_model_path = Path(stage3_model_path_arg).expanduser() if stage3_model_path_arg else model_path.parent / f"{model_path.stem}_stage3_pruned_channel.pth"
	stage3_model_path.parent.mkdir(parents=True, exist_ok=True)

	use_pretrained_stage1 = stage1_model_path_arg is not None
	use_pretrained_stage2 = stage2_model_path_arg is not None
	use_pretrained_stage3 = stage3_model_path_arg is not None

	if use_pretrained_stage1 and not stage1_model_path.exists():
		raise FileNotFoundError(f"Provided stage1 model path does not exist: {stage1_model_path}")
	if use_pretrained_stage2 and not stage2_model_path.exists():
		raise FileNotFoundError(f"Provided stage2 model path does not exist: {stage2_model_path}")
	if use_pretrained_stage3 and not stage3_model_path.exists():
		raise FileNotFoundError(f"Provided stage3 model path does not exist: {stage3_model_path}")

	val_subject = int(val_subjects[0])
	fixed_masks = _load_fixed_masks(
		fixed_mask_path,
		val_subject=val_subject,
		device=device,
		swap_enabled=fixed_mask_swap_enabled,
		swap_seed=fixed_mask_swap_seed,
	)
	if fixed_masks is not None:
		keep_counts = {k: int(v.sum().item()) for k, v in fixed_masks.items()}
		swap_mode_text = "ON" if fixed_mask_swap_enabled else "OFF"
		print(
			"Fixed masks loaded for val subject "
			f"{val_subject}: block2={keep_counts['block2']}/64, "
			f"block3={keep_counts['block3']}/128, block4={keep_counts['block4']}/128; "
			f"swap_mode={swap_mode_text}, swap_seed={fixed_mask_swap_seed}"
		)

	train_dataset = WEAR_Dataset(root_path, split="train", subject_ids=train_subjects, preprocessing=preprocessing)
	val_dataset = WEAR_Dataset(root_path, split="train", subject_ids=val_subjects, preprocessing=preprocessing)
	test_dataset = WEAR_Dataset(root_path, split="test", subject_ids=None, preprocessing=preprocessing)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	print(f"Train samples: {len(train_dataset)}")
	print(f"Val samples: {len(val_dataset)}")
	print(f"Test samples: {len(test_dataset)}")

	freq_bins = train_dataset[0][0].shape[-1]
	criterion = nn.CrossEntropyLoss()

	print("\n" + "=" * 60)
	if use_pretrained_stage1:
		print("STAGE 1: Loading pretrained SeparableConvCNN")
		print(f"Checkpoint: {stage1_model_path}")
	else:
		print("STAGE 1: Training SeparableConvCNN")
	print("=" * 60)

	model_stage1 = SeparableConvCNN(num_classes=8, num_channels=6, freq_bins=freq_bins, dropout=dropout).to(device)
	stage1_best_val_loss = None
	stage1_best_epoch = None

	if not use_pretrained_stage1:
		optimizer = torch.optim.Adam(model_stage1.parameters(), lr=lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
		stage1_best_val_loss = float("inf")
		stage1_best_epoch = 0
		no_improve = 0

		for epoch in range(epochs_stage1):
			# Train
			model_stage1.train()
			train_loss_sum, train_correct, train_total = 0.0, 0, 0

			for x, y in train_loader:
				x, y = x.to(device), y.to(device)
				out = model_stage1(x)
				loss = criterion(out, y)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				bs = y.size(0)
				train_loss_sum += loss.item() * bs
				_, pred = out.max(1)
				train_total += bs
				train_correct += pred.eq(y).sum().item()

			train_loss = train_loss_sum / max(train_total, 1)
			train_acc = 100.0 * train_correct / max(train_total, 1)
			
			# Val
			val_loss, val_acc, _ = _evaluate_classifier(model_stage1, val_loader, criterion, device)
			scheduler.step(val_loss)
			# Early stopping
			if val_loss < stage1_best_val_loss - min_delta:
				stage1_best_val_loss = val_loss
				stage1_best_epoch = epoch + 1
				torch.save(model_stage1.state_dict(), stage1_model_path)
				no_improve = 0
			else:
				no_improve += 1

			print(
				f"Epoch [{epoch+1}/{epochs_stage1}]: "
				f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
				f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}"
			)

			if wandb_run is not None:
				wandb_run.log({
					"stage": 1,
					"epoch": epoch + 1,
					"train_loss": train_loss,
					"train_acc": train_acc,
					"val_loss": val_loss,
					"val_acc": val_acc,
					"best_val_loss": stage1_best_val_loss,
					"lr": optimizer.param_groups[0]["lr"],
				})

			if no_improve >= patience:
				print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage1}] (patience={patience}).")
				break
	# Test
	model_stage1.load_state_dict(torch.load(stage1_model_path, map_location=device))
	stage1_test_loss, stage1_test_acc, stage1_test_f1 = _evaluate_classifier(model_stage1, test_loader, criterion, device)

	print("-" * 50)
	print("Stage 1 Summary:")
	if stage1_best_val_loss is not None:
		print(f"Best Val Loss: {stage1_best_val_loss:.4f} at Epoch {stage1_best_epoch}")
	else:
		print("Best Val Loss: not available (loaded pretrained stage1 checkpoint)")
	print(f"Test Loss: {stage1_test_loss:.4f} | Test Acc: {stage1_test_acc:.2f}% | Test F1 Macro: {stage1_test_f1:.4f}")

	print("\n" + "=" * 60)
	if use_pretrained_stage2:
		print("STAGE 2: Loading pretrained GumbelChannelPruningCNN")
		print(f"Checkpoint: {stage2_model_path}")
	else:
		print("STAGE 2: Training GumbelChannelPruningCNN (fixed masks + learned masks)")
	print("=" * 60)

	model_stage2 = GumbelChannelPruningCNN(
		num_classes=8,
		num_channels=6,
		freq_bins=freq_bins,
		dropout=dropout,
		tau_start=tau_start,
		tau_end=tau_end,
	).to(device)

	stage2_best_val_loss = None
	stage2_best_epoch = None

	if use_pretrained_stage2:
		model_stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
	else:
		_load_stage1_weights_to_channel_gumbel_model(model_stage2, torch.load(stage1_model_path, map_location=device))

		stage2_backbone_lr = lr * train_kwargs.get("stage2_backbone_lr_factor", 0.1)
		stage2_named_params = list(model_stage2.named_parameters())
		gumbel_params = [p for n, p in stage2_named_params if n.startswith("chan_logits_")]
		gumbel_param_ids = {id(p) for p in gumbel_params}
		backbone_params = [p for _, p in stage2_named_params if id(p) not in gumbel_param_ids]

		# Keep Gumbel parameters trainable unless explicitly requested otherwise.
		freeze_gumbel_masks = bool(train_kwargs.get("freeze_gumbel_masks", False))
		if freeze_gumbel_masks:
			for p in gumbel_params:
				p.requires_grad = False
			optimizer = torch.optim.Adam([
				{"params": backbone_params, "lr": stage2_backbone_lr},
			])
		else:
			if fixed_masks is not None:
				_set_gumbel_logits_to_allow_all_channels(model_stage2)
			optimizer = torch.optim.Adam(
				[
					{"params": backbone_params, "lr": stage2_backbone_lr},
					{"params": gumbel_params, "lr": lr},
				]
			)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

		stage2_best_val_loss = float("inf")
		stage2_best_epoch = 0
		no_improve = 0

		for epoch in range(epochs_stage2):
			model_stage2.train()
			if hasattr(model_stage2, "set_tau") and not freeze_gumbel_masks:
				model_stage2.set_tau(epoch, epochs_stage2)

			train_loss_sum, train_correct, train_total = 0.0, 0, 0
			with _temporary_fixed_channel_masks(model_stage2, fixed_masks):
				for x, y in train_loader:
					x, y = x.to(device), y.to(device)
					out = model_stage2(x)
					loss = criterion(out, y)
					if (not freeze_gumbel_masks) and model_stage2.mask_l1 is not None:
						loss = loss + sparsity_weight * model_stage2.mask_l1

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					bs = y.size(0)
					train_loss_sum += loss.item() * bs
					_, pred = out.max(1)
					train_total += bs
					train_correct += pred.eq(y).sum().item()

			train_loss = train_loss_sum / max(train_total, 1)
			train_acc = 100.0 * train_correct / max(train_total, 1)

			model_stage2.eval()
			with _temporary_fixed_channel_masks(model_stage2, fixed_masks):
				val_loss, val_acc, _ = _evaluate_classifier(model_stage2, val_loader, criterion, device)
			scheduler.step(val_loss)

			if val_loss < stage2_best_val_loss - min_delta:
				stage2_best_val_loss = val_loss
				stage2_best_epoch = epoch + 1
				torch.save(model_stage2.state_dict(), stage2_model_path)
				no_improve = 0
			else:
				no_improve += 1

			mask_info = f"; Mask: {model_stage2.mask_l1.item():.2%}" if model_stage2.mask_l1 is not None else ""
			print(
				f"Epoch [{epoch+1}/{epochs_stage2}]: "
				f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
				f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}" + mask_info
			)

			if wandb_run is not None:
				log_payload = {
					"stage": 2,
					"epoch": epoch + 1,
					"train_loss": train_loss,
					"train_acc": train_acc,
					"val_loss": val_loss,
					"val_acc": val_acc,
					"best_val_loss": stage2_best_val_loss,
					"lr_backbone": optimizer.param_groups[0]["lr"],
					"mask_l1": model_stage2.mask_l1.item() if model_stage2.mask_l1 is not None else None,
					"stage2_mask_source": "fixed_json" if freeze_gumbel_masks else "learned_gumbel",
					"stage2_gumbel_frozen": freeze_gumbel_masks,
				}
				if not freeze_gumbel_masks and len(optimizer.param_groups) > 1:
					log_payload["lr_gumbel"] = optimizer.param_groups[1]["lr"]
				wandb_run.log(log_payload)

			if no_improve >= patience:
				print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage2}] (patience={patience}).")
				break

	model_stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
	model_stage2.eval()
	with _temporary_fixed_channel_masks(model_stage2, fixed_masks):
		stage2_test_loss, stage2_test_acc, stage2_test_f1 = _evaluate_classifier(model_stage2, test_loader, criterion, device)

	learned_hard_masks = model_stage2.get_hard_masks()
	# If fixed masks are provided, use them directly as the effective masks.
	if fixed_masks is not None:
		effective_hard_masks = {k: v.detach().clone() for k, v in fixed_masks.items()}
	else:
		effective_hard_masks = _combine_hard_and_fixed_masks(learned_hard_masks, fixed_masks)
	keep_counts = {name: int((mask > 0.5).sum().item()) for name, mask in effective_hard_masks.items()}
	total_channels = 64 + 128 + 128
	pruning_stats = {
		"Block2_Pruned_%": (1 - keep_counts["block2"] / 64.0) * 100.0,
		"Block3_Pruned_%": (1 - keep_counts["block3"] / 128.0) * 100.0,
		"Block4_Pruned_%": (1 - keep_counts["block4"] / 128.0) * 100.0,
		"Total_Pruned_%": (1 - (keep_counts["block2"] + keep_counts["block3"] + keep_counts["block4"]) / total_channels) * 100.0,
	}

	print("-" * 50)
	print("Stage 2 Summary:")
	if stage2_best_val_loss is not None:
		print(f"Best Val Loss: {stage2_best_val_loss:.4f} at Epoch {stage2_best_epoch}")
	else:
		print("Best Val Loss: not available (loaded pretrained stage2 checkpoint)")
	print(f"Test Loss: {stage2_test_loss:.4f} | Test Acc: {stage2_test_acc:.2f}% | Test F1 Macro: {stage2_test_f1:.4f}")
	print(
		"Effective hard channel keeps: "
		f"block2={keep_counts['block2']}/64, "
		f"block3={keep_counts['block3']}/128, "
		f"block4={keep_counts['block4']}/128"
	)

	print("\n" + "=" * 60)
	if use_pretrained_stage3:
		print("STAGE 3: Loading pretrained compact channel-pruned model")
		print(f"Checkpoint: {stage3_model_path}")
	else:
		print("STAGE 3: Building compact pruned model and fine-tuning")
	print("=" * 60)

	stage3_model, keep_indices, _ = _build_pruned_channel_model_from_stage2_with_fixed(
		stage2_model=model_stage2,
		fixed_masks=fixed_masks,
		num_classes=8,
		dropout=dropout,
		device=device,
	)

	dense_reference_model = SeparableConvCNN(num_classes=8, num_channels=6, freq_bins=freq_bins, dropout=dropout)
	dense_param_count = _count_parameters(dense_reference_model)
	pruned_param_count = _count_parameters(stage3_model)
	param_reduction_pct = (1.0 - pruned_param_count / max(dense_param_count, 1)) * 100.0

	print(
		f"Stage 3 model size: {pruned_param_count:,} params vs dense {dense_param_count:,} params "
		f"({param_reduction_pct:.2f}% reduction)"
	)

	stage3_best_val_loss = None
	stage3_best_epoch = None

	if use_pretrained_stage3:
		stage3_model.load_state_dict(torch.load(stage3_model_path, map_location=device))
	else:
		optimizer = torch.optim.Adam([{"params": list(stage3_model.parameters()), "lr": lr * stage3_loaded_lr_factor}])
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

		stage3_best_val_loss = float("inf")
		stage3_best_epoch = 0
		no_improve = 0

		for epoch in range(epochs_stage3):
			stage3_model.train()
			train_loss_sum, train_correct, train_total = 0.0, 0, 0

			for x, y in train_loader:
				x, y = x.to(device), y.to(device)
				out = stage3_model(x)
				loss = criterion(out, y)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				bs = y.size(0)
				train_loss_sum += loss.item() * bs
				_, pred = out.max(1)
				train_total += bs
				train_correct += pred.eq(y).sum().item()

			train_loss = train_loss_sum / max(train_total, 1)
			train_acc = 100.0 * train_correct / max(train_total, 1)
			val_loss, val_acc, _ = _evaluate_classifier(stage3_model, val_loader, criterion, device)
			scheduler.step(val_loss)

			if val_loss < stage3_best_val_loss - min_delta:
				stage3_best_val_loss = val_loss
				stage3_best_epoch = epoch + 1
				torch.save(stage3_model.state_dict(), stage3_model_path)
				no_improve = 0
			else:
				no_improve += 1

			print(
				f"Epoch [{epoch+1}/{epochs_stage3}]: "
				f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
				f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}"
			)

			if wandb_run is not None:
				wandb_run.log({
					"stage": 3,
					"epoch": epoch + 1,
					"train_loss": train_loss,
					"train_acc": train_acc,
					"val_loss": val_loss,
					"val_acc": val_acc,
					"best_val_loss": stage3_best_val_loss,
					"lr_loaded_weights": optimizer.param_groups[0]["lr"],
					"model_param_count": pruned_param_count,
					"model_param_reduction_pct": param_reduction_pct,
				})

			if no_improve >= patience:
				print(f"Early Stopping at Epoch [{epoch+1}/{epochs_stage3}] (patience={patience}).")
				break

		stage3_model.load_state_dict(torch.load(stage3_model_path, map_location=device))

	stage3_test_loss, stage3_test_acc, stage3_test_f1 = _evaluate_classifier(stage3_model, test_loader, criterion, device)

	if wandb_run is not None:
		wandb_run.log({
			"stage1_test_loss": stage1_test_loss,
			"stage1_test_acc": stage1_test_acc,
			"stage1_test_f1": stage1_test_f1,
			"stage2_test_loss": stage2_test_loss,
			"stage2_test_acc": stage2_test_acc,
			"stage2_test_f1": stage2_test_f1,
			"stage3_test_loss": stage3_test_loss,
			"stage3_test_acc": stage3_test_acc,
			"stage3_test_f1": stage3_test_f1,
			"model_param_count_dense": dense_param_count,
			"model_param_count_pruned": pruned_param_count,
			"model_param_reduction_pct": param_reduction_pct,
			**pruning_stats,
		})

	return {
		"stage1": {
			"model": "SeparableConvCNN",
			"best_val_loss": stage1_best_val_loss,
			"best_epoch": stage1_best_epoch,
			"test_loss": stage1_test_loss,
			"test_acc": stage1_test_acc,
			"test_f1_macro": stage1_test_f1,
			"model_path": str(stage1_model_path),
			"loaded_from_checkpoint": use_pretrained_stage1,
		},
		"stage2": {
			"model": "GumbelChannelPruningCNN",
			"best_val_loss": stage2_best_val_loss,
			"best_epoch": stage2_best_epoch,
			"test_loss": stage2_test_loss,
			"test_acc": stage2_test_acc,
			"test_f1_macro": stage2_test_f1,
			"model_path": str(stage2_model_path),
			"hard_masks": {k: v.detach().cpu().numpy() for k, v in learned_hard_masks.items()},
			"final_mask": {k: v.detach().cpu().numpy() for k, v in effective_hard_masks.items()},
			"pruning_stats": pruning_stats,
		},
		"stage3": {
			"model": "PrunedSeparableConvCNN",
			"best_val_loss": stage3_best_val_loss,
			"best_epoch": stage3_best_epoch,
			"test_loss": stage3_test_loss,
			"test_acc": stage3_test_acc,
			"test_f1_macro": stage3_test_f1,
			"model_path": str(stage3_model_path),
			"loaded_from_checkpoint": use_pretrained_stage3,
			"dense_param_count": dense_param_count,
			"pruned_param_count": pruned_param_count,
			"param_reduction_pct": param_reduction_pct,
			"block2_keep": int(keep_indices["block2"].numel()),
			"block3_keep": int(keep_indices["block3"].numel()),
			"block4_keep": int(keep_indices["block4"].numel()),
		},
	}
