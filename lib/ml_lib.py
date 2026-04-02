import torch
from pathlib import Path
from sklearn.metrics import f1_score
import json

def _load_weights_to_gumbel_model(gumbel_model, stage1_state_dict):
    """
    Load weights from SeparableConvCNN (stage 1) into GumbelMaskSeparableConvCNN (stage 2). 
    The Gumbel model has an additional 'bin_logits' parameter that doesn't exist in the
    base SeparableConvCNN model, so we selectively load weights.
    
    Args:
        gumbel_model: GumbelMaskSeparableConvCNN model to load weights into
        stage1_state_dict: State dict from trained SeparableConvCNN model
    """
    gumbel_state_dict = gumbel_model.state_dict()
    
    # Load all weights that exist in both models
    for name, param in stage1_state_dict.items():
        if name in gumbel_state_dict:
            gumbel_state_dict[name] = param
            print(f"  Loaded: {name}")
        else:
            print(f"  Skipped (not in Gumbel model): {name}")
    
    # The bin_logits will be initialized randomly (not loaded from stage 1)
    print(f"  Kept random initialization: bin_logits (Gumbel-specific parameter)")
    
    gumbel_model.load_state_dict(gumbel_state_dict)
    return gumbel_model


def _val_one_epoch(model, dataloader, criterion, 
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    args:
        model, dataloader, criterion
    return:
        val_loss, val_acc, val_f1
    """
    model.eval()
    loss_sum, total, correct = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            bs = y.size(0)
            loss_sum += loss.item() * bs
            _, pred = out.max(1)
            total += bs
            correct += pred.eq(y).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    val_loss = loss_sum / max(total, 1)
    val_acc = 100.0 * correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0
    return val_loss, val_acc, f1

def _train_one_epoch(model, optimizer, dataloader, criterion, 
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    args:
        model
        dataloader
        criterion
    return:
        train_loss, train_acc
    """
    
    model.train()
    train_loss_sum, train_correct, train_total = 0.0, 0, 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
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
    return train_loss, train_acc


def stage1_pipeline(model, train_loader, val_loader, test_loader, 
                    criterion, optimizer, scheduler, checkpoint_path,
                    num_epochs=60, patience=10, min_delta=1e-3,
                    wandb_run=None, use_pretrained=False,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Run stage-1 training (or load pretrained checkpoint) and evaluate on test set.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_best_val_loss = None
    stage1_best_epoch = None

    # Use pretrained model
    if use_pretrained:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"pre-trained path not found: {checkpoint_path}")
        # Load pretrain model
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        stage1_test_loss, stage1_test_acc, stage1_test_f1 = _val_one_epoch(model, test_loader, criterion, device)
        print("-" * 50 + "\nStage 1 Summary:")
        print("Use pretrained model:")
        print(f"Test Loss: {stage1_test_loss:.4f} | Test Acc: {stage1_test_acc:.2f}% | Test F1 Macro: {stage1_test_f1:.4f}")
        if wandb_run is not None:
            wandb_run.log({
                "stage1_test_loss": stage1_test_loss,
                "stage1_test_acc": stage1_test_acc,
                "stage1_test_f1": stage1_test_f1,
                "stage1/test_loss": stage1_test_loss,
                "stage1/test_acc": stage1_test_acc,
                "stage1/test_f1": stage1_test_f1,
                "stage1/use_pretrain_model": False,
            })
        return {
            "best_val_loss": stage1_best_val_loss,
            "best_epoch": stage1_best_epoch,
            "test_loss": stage1_test_loss,
            "test_acc": stage1_test_acc,
            "test_f1": stage1_test_f1,
            "model_path": str(checkpoint_path),
            "loaded_from_checkpoint": True,
        }
    
    # Default stage 1 pipeline  
    stage1_best_val_loss = float('inf')
    stage1_best_epoch = 0
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = _train_one_epoch(model, optimizer, train_loader, criterion, device)
        val_loss, val_acc, _ = _val_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < stage1_best_val_loss - min_delta:
            stage1_best_val_loss = val_loss
            stage1_best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch [{epoch+1}/{num_epochs}]: "
            f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
            f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "stage1/stage_number": 1,
                "stage1/purpose": "train baseline model",
                "stage1/epoch": epoch + 1,
                "stage1/train_loss": train_loss,
                "stage1/train_acc": train_acc,
                "stage1/val_loss": val_loss,
                "stage1/val_acc": val_acc,
                "stage1/best_val_loss": stage1_best_val_loss,
                "stage1/lr": optimizer.param_groups[0]["lr"],
            })

        if no_improve >= patience:
            print(f"Early Stopping at Epoch [{epoch+1}/{num_epochs}] (patience={patience}).")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    stage1_test_loss, stage1_test_acc, stage1_test_f1 = _val_one_epoch(model, test_loader, criterion, device)
    
    print("-" * 50 + "\nStage 1 Summary:")
    print(f"Best Val Loss: {stage1_best_val_loss:.4f} at Epoch {stage1_best_epoch}")
    print(f"Test Loss: {stage1_test_loss:.4f} | Test Acc: {stage1_test_acc:.2f}% | Test F1 Macro: {stage1_test_f1:.4f}")
    
    if wandb_run is not None:
        wandb_run.log({
            "stage1_test_loss": stage1_test_loss,
            "stage1_test_acc": stage1_test_acc,
            "stage1_test_f1": stage1_test_f1,
            "stage1/test_loss": stage1_test_loss,
            "stage1/test_acc": stage1_test_acc,
            "stage1/test_f1": stage1_test_f1,
            "stage1/use_pretrain_model": False,
        })

    return {
        "best_val_loss": stage1_best_val_loss,
        "best_epoch": stage1_best_epoch,
        "test_loss": stage1_test_loss,
        "test_acc": stage1_test_acc,
        "test_f1": stage1_test_f1,
        "model_path": str(checkpoint_path),
        "loaded_from_checkpoint": False,
    }

def stage2_channel_gumbel_pruning_pipeline(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    checkpoint_path,
    stage1_checkpoint_path=None,
    lr=1e-3,
    backbone_lr_factor=0.1,
    sparsity_weight=0.01,
    num_epochs=60,
    patience=10,
    min_delta=1e-3,
    wandb_run=None,
    use_pretrained=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Run stage-2 channel-pruning training (or load pretrained checkpoint) and evaluate on test set.

    Expected model behavior:
    - optional set_tau(epoch, num_epochs)
    - optional mask_l1 attribute for sparsity regularization
    - get_hard_masks() returning dict-like masks (e.g., block2/block3/block4)
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    stage2_best_val_loss = None
    stage2_best_epoch = None

    if use_pretrained: # pre-trained stage 2
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"pre-trained path not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else: # load stage 1 weight
        if stage1_checkpoint_path is not None:
            print("\nLoading Stage 1 weights into Stage 2 model:")
            stage1_state = torch.load(Path(stage1_checkpoint_path).expanduser(), map_location=device)
            _load_weights_to_gumbel_model(model, stage1_state)

        named_params = list(model.named_parameters()) # all model param
        gumbel_params = [p for n, p in named_params if n.startswith("chan_logits_")] # gumbel model param
        gumbel_param_ids = {id(p) for p in gumbel_params}
        backbone_params = [p for _, p in named_params if id(p) not in gumbel_param_ids] # backbond model param

        optimizer = torch.optim.Adam([{"params": backbone_params, "lr": lr * backbone_lr_factor}, {"params": gumbel_params, "lr": lr}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

        stage2_best_val_loss = float("inf")
        stage2_best_epoch = 0
        no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            if hasattr(model, "set_tau"):
                model.set_tau(epoch, num_epochs)

            train_loss_sum, train_correct, train_total = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                if getattr(model, "mask_l1", None) is not None:
                    loss = loss + sparsity_weight * model.mask_l1

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
            val_loss, val_acc, _ = _val_one_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < stage2_best_val_loss - min_delta:
                stage2_best_val_loss = val_loss
                stage2_best_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path)
                no_improve = 0
            else:
                no_improve += 1

            mask_info = f"; Mask: {model.mask_l1.item():.2%}" if getattr(model, "mask_l1", None) is not None else ""
            print(
                f"Epoch [{epoch+1}/{num_epochs}]: "
                f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; "
                f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}" + mask_info
            )

            if wandb_run is not None:
                wandb_run.log({
                    "stage2/stage_number": 2,
                    "stage2/purpose": "train channel-pruning model with masking",
                    "stage2/epoch": epoch + 1,
                    "stage2/train_loss": train_loss,
                    "stage2/train_acc": train_acc,
                    "stage2/val_loss": val_loss,
                    "stage2/val_acc": val_acc,
                    "stage2/best_val_loss": stage2_best_val_loss,
                    "stage2/lr_backbone": optimizer.param_groups[0]["lr"],
                    "stage2/lr_gumbel": optimizer.param_groups[1]["lr"],
                    "stage2/mask_l1": model.mask_l1.item() if getattr(model, "mask_l1", None) is not None else None,
                })

            if no_improve >= patience:
                print(f"Early Stopping at Epoch [{epoch+1}/{num_epochs}] (patience={patience}).")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    stage2_test_loss, stage2_test_acc, stage2_test_f1 = _val_one_epoch(model, test_loader, criterion, device)

    hard_masks = model.get_hard_masks() if hasattr(model, "get_hard_masks") else {}
    keep_counts = {
        name: int((mask > 0.5).sum().item())
        for name, mask in hard_masks.items()
    }

    total_channels = int(sum(mask.numel() for mask in hard_masks.values())) if len(hard_masks) > 0 else 0
    pruning_stats = {}
    for name, mask in hard_masks.items():
        block_total = float(mask.numel())
        pruning_stats[f"{name}_Pruned_%"] = (1.0 - keep_counts[name] / max(block_total, 1.0)) * 100.0
    if total_channels > 0:
        pruning_stats["Total_Pruned_%"] = (1.0 - sum(keep_counts.values()) / float(total_channels)) * 100.0

    print("-" * 50)
    print("Stage 2 Summary:")
    if stage2_best_val_loss is not None:
        print(f"Best Val Loss: {stage2_best_val_loss:.4f} at Epoch {stage2_best_epoch}")
    else:
        print("Best Val Loss: not available (loaded pretrained stage2 checkpoint)")
    print(f"Test Loss: {stage2_test_loss:.4f} | Test Acc: {stage2_test_acc:.2f}% | Test F1 Macro: {stage2_test_f1:.4f}")

    if len(keep_counts) > 0:
        ordered = [k for k in ("block2", "block3", "block4") if k in keep_counts]
        ordered.extend([k for k in keep_counts.keys() if k not in ordered])
        keep_str = ", ".join(f"{k}={keep_counts[k]}/{hard_masks[k].numel()}" for k in ordered)
        print(f"Hard channel keeps: {keep_str}")

    if wandb_run is not None:
        log_payload = {
            "stage2/test_loss": stage2_test_loss,
            "stage2/test_acc": stage2_test_acc,
            "stage2/test_f1": stage2_test_f1,
            "stage2/best_val_loss": stage2_best_val_loss,
            "stage2/best_epoch": stage2_best_epoch,
        }
        log_payload.update({f"stage2/{k}": v for k, v in pruning_stats.items()})
        
        # Best checkpoint is already loaded above, so these are hard masks of best epoch.
        for name, mask in hard_masks.items():
            hard_mask_np = (mask > 0.5).float().detach().cpu().numpy()
            log_payload[f"stage2/raw_hard_mask_{name}"] = json.dumps(hard_mask_np.tolist())
            log_payload[f"stage2/final_mask_keep_ratio_{name}"] = float(hard_mask_np.mean())
        
        wandb_run.log(log_payload)

    return {
        "best_val_loss": stage2_best_val_loss,
        "best_epoch": stage2_best_epoch,
        "test_loss": stage2_test_loss,
        "test_acc": stage2_test_acc,
        "test_f1": stage2_test_f1,
        "model_path": str(checkpoint_path),
        "hard_masks": hard_masks,
        "keep_counts": keep_counts,
        "pruning_stats": pruning_stats,
        "loaded_from_checkpoint": use_pretrained,
    }