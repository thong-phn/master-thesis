## Plan: UCI Five-Stage LOSO Port

Recreate the WEAR five-stage runner and training flow for UCI-HAR by adding a new entry script (`uci_main_loso_five_stage.py`) with full CLI parity and implementing a UCI-specific five-stage pipeline inside `lib/uci_train.py` (not shared refactor). Keep UCI dataset semantics (6 classes, UCI subject files), preserve WEAR stage logic, and expose W&B project routing via CLI.

**Steps**
1. Phase 1: Establish UCI main-script scaffold from WEAR (can start immediately)
2. Create `uci_main_loso_five_stage.py` by adapting `wear_main_loso_five_stage.py` structure: seed setup, subject loading/parsing, fold loop, per-fold W&B run, per-fold metrics append, overall cross-fold summary, and artifact upload helper.
3. Port full CLI surface from WEAR: `--epochs_stage1..5`, LR factors, sparsity weights, tau range, `--performance`, checkpoint overrides (`--stage1_model_path`..`--stage5_model_path`), `--subjects`, `--run_name`, `--wandb`; add UCI-specific W&B project argument (e.g., `--wandb_project`) per user request.
4. Keep path conventions UCI-specific: dataset root `uci-har`, model output naming `uci_best_model_five_stage_subject{val_subject}_val.pth`, log naming `uci_loso_five_stage_results_{preprocessing}.txt`; retain `{subject}` placeholder expansion for checkpoint override parity with WEAR. *parallel with step 5*
5. In `uci_main_loso_five_stage.py`, add stage-1 checkpoint resolution logic identical to WEAR: compute `resolved_stage1_model_path` per fold, and if `--stage1_model_path` contains `{subject}`, substitute current `val_subject` before passing into W&B config and `train_loso_uci_multi_stage(...)`.
6. Phase 2: Build UCI five-stage training engine in `lib/uci_train.py` (depends on step 2 for call contract)
7. Add utility helpers required by five-stage flow (copied/adapted from `lib/wear_train.py`): `_resolve_device`, `_evaluate_classifier`, `_get_hard_bin_mask_from_model`, `_save_confusion_matrix_artifact`, `_count_parameters`, `_copy_batchnorm_subset`, `_copy_separable_conv_subset`, `_copy_linear_input_subset`, `_build_pruned_channel_model_from_stage2`, `_load_matching_weights`, and a UCI slicing dataset wrapper (analogous to `SlicedWEARDataset`).
8. Implement `train_loso_uci_multi_stage(root_path, train_subjects, val_subjects, wandb_run=None, **train_kwargs)` with WEAR-equivalent five stages and checkpoint-skip behavior:
9. Stage 1: train/load `SeparableConvCNN` on full UCI input.
10. Stage 2: init `GumbelMaskSeparableConvCNN` from stage 1, learn input-bin mask using dual LR groups and sparsity bin penalty.
11. Stage 3: apply hard bin mask to datasets, train/load `SeparableConvCNN` on pruned input.
12. Stage 4: init `GumbelChannelPruningCNN` from stage 3, learn channel masks with channel sparsity penalty.
13. Stage 5: build compact `PrunedSeparableConvCNN` from stage 4 hard channel masks and fine-tune.
14. Ensure UCI-specific constants are used consistently: `num_classes=6`, `num_channels=6`, `freq_bins` from UCI data shape, and class labels compatible with current UCI dataset indexing (0-5).
15. Preserve return payload compatibility with WEAR five-stage main expectations: nested `stage1..stage5` dicts with `test_acc`, `test_f1_macro`, `best_val_loss`, `best_epoch`, paths, checkpoint flags, and stage-specific pruning metadata.
16. Phase 3: Wire main script to new UCI training API and finalize reporting (depends on steps 2-15)
17. Import and call `train_loso_uci_multi_stage` from `lib/uci_train.py`; update per-fold logging to include stage metrics, Stage 5 vs Stage 1 improvement, hard bin mask, channel masks, and parameter reduction summaries.
18. Keep W&B behavior controllable by CLI: allow disable (`--wandb False`) and explicit project selection (`--wandb_project`), while preserving fold-level + summary artifact workflow.
19. Phase 4: Verification and regression checks (depends on all prior steps)
20. Static validation: ensure script executes `--help`, argument parsing works, and all required kwargs are forwarded without missing keys.
21. Dry-run sanity for one subject fold (or reduced epochs): verify stage transitions, checkpoint loading/saving, and that pruned datasets have non-empty bins/channels.
22. Behavioral checks: verify stage return keys consumed by main exist; verify no mismatch for metric key names (`test_f1_macro` vs `test_f1`) and class count assumptions.
23. W&B/log artifact checks: confirm result log file and confusion matrix artifacts are produced for stage 1/3/5 and summary artifact upload works under chosen project name.

**Relevant files**
- `/home/qphan/master-thesis/wear_main_loso_five_stage.py` — source template for five-stage CLI, subject filtering, fold loop, and summary logging.
- `/home/qphan/master-thesis/uci_main_loso_five_stage.py` — new UCI entrypoint to create with WEAR parity + configurable W&B project.
- `/home/qphan/master-thesis/lib/wear_train.py` — source template for five-stage orchestration and pruning helper utilities.
- `/home/qphan/master-thesis/lib/uci_train.py` — target for UCI-specific five-stage pipeline and helper additions.
- `/home/qphan/master-thesis/lib/model.py` — confirm constructor signatures and class defaults for UCI/WEAR model compatibility.
- `/home/qphan/master-thesis/uci_main_loso_baseline.py` — preserve UCI conventions for dataset root, logging style, and project naming defaults.

**Verification**
1. Run `python uci_main_loso_five_stage.py --help` and verify all WEAR-parity args plus `--wandb_project` appear.
2. Run one constrained fold (single subject selection and low epochs) and confirm all five stages execute and produce checkpoints.
3. Inspect generated log file under `log/uci_loso_five_stage_results_<preprocessing>.txt` for per-stage entries and overall summary statistics.
4. Validate that stage payloads include expected keys by printing one fold metrics dict before summary aggregation.
5. If W&B enabled, confirm fold runs, summary artifact upload, and confusion matrix artifacts are visible in the selected project.

**Decisions**
- Full CLI parity with WEAR five-stage script is included.
- Five-stage logic will be implemented directly in `lib/uci_train.py` (no cross-dataset refactor).
- W&B project name must be user-configurable via CLI (not hardcoded).
- Scope includes UCI five-stage runner + UCI train pipeline only; no refactor of existing WEAR paths.

**Further Considerations**
1. Default W&B project string recommendation: `thesis-uci-five-stage` to separate from baseline runs while keeping discoverability.
2. Keep `--subjects` parser behavior identical to WEAR (comma/dot/space separators) to simplify experiment scripting reuse.
3. Preserve current UCI baseline APIs (`train_loso`) to avoid breaking existing single-stage scripts while adding new five-stage API alongside it.
