# GumbelMaskSeparableConvCNN Improvement Plan

## Current Status
- **Initial Accuracy**: 87.51%
- **After Exp-G1 + G5**: 89.48% (+1.97%)
- **After Exp-G4 attempt (constant tau=2.0)**: 90.53% (+1.05%)
- **After Exp-G4 with annealing**: 88.80% - **REJECTED** ❌
- **After Exp-G3 (sparsity reg=0.01)**: **90.97% (+0.44%)** - 35 bins (53.8%) ✅ BEST
- **After Exp-S1 (sparsity reg=0.10)**: 86.56% (-4.41%) - 18 bins (27.7%) - Too aggressive ⚠️
- **After Exp-S2 (sparsity reg=0.05)**: 89.72% (-1.25%) - 23 bins (35.4%) - Good balance ✅
- **Exp-S3 (target sparsity)**: Paused
- **Current training config**: Reverted to **Exp-G3 (sparsity_weight=0.01)**
- **Target**: 90.0% (match baseline SeparableConvCNN)
- **Status**: Best = 90.97% with 35 bins ✅

## Model Overview
`GumbelMaskSeparableConvCNN` learns to mask/select frequency bins using Gumbel-Softmax:
- Learnable `bin_logits` (freq_bins × 2) for each bin's [off, on] probabilities
- Differentiable masking via Gumbel-Softmax (training) / Softmax (inference)
- Same SeparableConv architecture after masking

---

## Potential Issues & Hypotheses

### 1. **Over-masking (too many bins turned off)**
- **Hypothesis**: Model masks too many discriminative bins to minimize training loss
- **Check**: Log `mask_l1` (fraction of bins kept) during training
- **Expected**: Should keep 50-80% of bins; <30% suggests over-masking

### 2. **Gumbel temperature (tau) suboptimal**
- **Current**: `gumbel_tau=1.0` (default)
- **Issue**: Fixed tau doesn't allow exploration→exploitation transition
- **Effect**: 
  - High tau (>1): soft/stochastic masking, better exploration
  - Low tau (<0.5): hard/deterministic masking, better exploitation

### 3. **No explicit sparsity regularization**
- **Issue**: `mask_l1` is computed but **not used in loss**
- **Effect**: No incentive to be selective; may keep all bins (defeating purpose) or drop critical ones

### 4. **Initialization bias**
- **Current**: `bin_logits = zeros(freq_bins, 2)` → 50% on/off initially
- **Issue**: No prior knowledge; uniform initialization may be suboptimal
- **Better**: Initialize with slight bias toward "on" (e.g., logits=[0, 0.5])

### 5. **Joint optimization instability**
- **Issue**: Mask and network weights learned together
- **Effect**: Mask may overfit to training distribution, hurting generalization
- **Solution**: Two-stage training or different learning rates

### 6. **Train/test mode discrepancy**
- **Training**: Gumbel-Softmax (stochastic, hard=True)
- **Test**: Regular softmax (deterministic, soft probabilities)
- **Issue**: Training sees binary masks, test sees weighted masks
- **Effect**: Train/test behavior mismatch hurts performance

### 7. **FFT normalization interaction**
- **Current setup**: Using log1p + lr=5e-4 from Exp-7
- **Issue**: Gumbel masking may need different normalization than baseline
- **Effect**: Masking already selects features; normalization may be redundant or harmful

---

## Improvement Experiments (Priority Order)

### Phase 1: Diagnostics (Understand Current Behavior)

#### Exp-G1: Log masking statistics
**Goal**: Understand what the model is currently doing

**Implementation**:
```python
# In training loop, after each epoch:
wandb.log({
    "mask_fraction": model.mask_l1.item(),  # fraction of bins kept
    "mask_entropy": -(mask * torch.log(mask + 1e-8)).sum(),  # uncertainty
})

# After training, log final mask:
final_mask = model.last_mask.cpu().numpy()
wandb.log({"final_mask": wandb.Histogram(final_mask)})
```

**Expected insights**:
- Is the model keeping too few bins (<30%)?
- Is the mask confident or uncertain?
- Which bins are being selected? (low freq vs high freq)

---

#### Exp-G2: Baseline with static mask (sanity check)
**Goal**: Confirm that learned masking is the bottleneck

**Config**: 
- Keep all bins (mask = 1 everywhere)
- Or use top-K bins from baseline importance

**Code**:
```python
# In forward():
# mask = torch.ones(freq_bins)  # disable masking
x = x  # no masking applied
```

**Expected**: Should match SeparableConvCNN (~90%)
**If not**: Issue is elsewhere (architecture, hyperparameters, etc.)

---

### Phase 2: Fix Training Dynamics

#### Exp-G3: Add sparsity regularization
**Goal**: Encourage selective masking with L1 penalty

**Implementation**:
```python
# In train loop:
outputs = model(fft_mag)
loss = criterion(outputs, labels)

# Add sparsity regularization
if hasattr(model, 'mask_l1'):
    sparsity_weight = 0.01  # tune this
    loss = loss + sparsity_weight * model.mask_l1

loss.backward()
```

**Tune**: Try `sparsity_weight` in [0.001, 0.01, 0.05, 0.1]
**Expected**: Encourages fewer bins, may improve generalization if current model over-relies on noisy bins

---

#### Exp-G4: Temperature annealing schedule
**Goal**: Start with exploration, end with exploitation

**Implementation**:
```python
class GumbelMaskSeparableConvCNN:
    def __init__(self, ..., tau_start=1.0, tau_end=0.1):
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.current_tau = tau_start
    
    def set_tau(self, epoch, max_epochs):
        # Linear annealing
        self.current_tau = self.tau_start - (self.tau_start - self.tau_end) * (epoch / max_epochs)

# In training loop:
model.set_tau(epoch, epochs)
```

**Schedule**: `tau: 2.0 → 0.3` over 60 epochs
**Expected**: Better exploration early, sharper decisions late

---

#### Exp-G5: Fix train/test discrepancy
**Goal**: Make test behavior match training

**Option A - Use hard mask in test**:
```python
def forward(self, x):
    if self.training:
        probs = F.gumbel_softmax(self.bin_logits, tau=self.gumbel_tau, hard=True)
    else:
        # Use argmax (hard) instead of soft probabilities
        probs_soft = torch.softmax(self.bin_logits, dim=-1)
        probs = F.one_hot(probs_soft.argmax(dim=-1), num_classes=2).float()
    
    mask = probs[:, 1]
    ...
```

**Option B - Use soft mask in training** (set `hard=False` in Gumbel)

**Expected**: Reduces train/test mismatch

---

#### Exp-G6: Two-stage training
**Goal**: Decouple mask learning from network learning

**Stage 1** (10 epochs): Train only `bin_logits`, freeze conv/fc weights
```python
optimizer_mask = torch.optim.Adam([model.bin_logits], lr=1e-3)
# Train for 10 epochs
```

**Stage 2** (50 epochs): Freeze mask (or use very low LR), train network
```python
optimizer_net = torch.optim.Adam(
    [p for n, p in model.named_parameters() if 'bin_logits' not in n],
    lr=5e-4
)
```

**Expected**: Mask learns useful bins first, then network optimizes for those bins

---

### Phase 3: Architecture & Initialization

#### Exp-G7: Better initialization
**Goal**: Start with informed prior (keep most bins initially)

**Implementation**:
```python
# In __init__:
# Initialize to favor "on" state
self.bin_logits = nn.Parameter(torch.zeros(freq_bins, 2))
nn.init.constant_(self.bin_logits[:, 1], 0.5)  # bias toward "on"
```

**Or** use frequency-based init:
```python
# Keep low-freq bins (0-10 Hz more important in HAR)
low_freq_bins = freq_bins // 4
self.bin_logits[:low_freq_bins, 1] = 1.0  # strong bias for low freq
```

**Expected**: Prevents random early pruning of important bins

---

#### Exp-G8: Separate learning rates
**Goal**: Mask and network may need different LR

**Implementation**:
```python
optimizer = torch.optim.Adam([
    {'params': [model.bin_logits], 'lr': 1e-4},  # slower for mask
    {'params': [p for n, p in model.named_parameters() if 'bin_logits' not in n], 'lr': 5e-4}
])
```

**Rationale**: Mask affects all downstream layers; needs gentler updates
**Expected**: More stable joint training

---

#### Exp-G9: Remove masking layer (ablation)
**Goal**: Confirm masking adds value vs. just being a bottleneck

**Implementation**: Use baseline SeparableConvCNN with same hyperparameters
**Expected**: If baseline >> GumbelMask, then masking approach needs rethinking

---

### Phase 4: Advanced Techniques

#### Exp-G10: Straight-through estimator
**Goal**: Alternative to Gumbel-Softmax for discrete masking

**Implementation**: Replace Gumbel with straight-through:
```python
if self.training:
    probs_soft = torch.softmax(self.bin_logits / self.tau, dim=-1)
    probs_hard = F.one_hot(probs_soft.argmax(dim=-1), 2).float()
    probs = probs_hard - probs_soft.detach() + probs_soft  # STE
else:
    probs = F.one_hot(torch.softmax(self.bin_logits, dim=-1).argmax(dim=-1), 2).float()
```

---

#### Exp-G11: Relaxed masking (soft attention)
**Goal**: Instead of hard on/off, use learned soft weights

**Implementation**:
```python
# Replace binary mask with continuous attention
attention = torch.sigmoid(self.bin_logits[:, 1] - self.bin_logits[:, 0])
x = x * attention.view(1, 1, -1)
```

**Expected**: Smoother optimization, may preserve more information

---

#### Exp-G12: FFT normalization ablation for Gumbel
**Goal**: Test if log1p normalization helps or hurts masking

**Configs to try**:
- No normalization (raw FFT)
- log1p only (current Exp-7)
- Channel-wise z-score

**Hypothesis**: Masking may work better on raw magnitudes (clearer energy differences)

---

## Recommended Experiment Sequence

### Week 1: Diagnostics
1. **Exp-G1**: ✅ DONE - Log masking statistics (prints fraction of bins kept each epoch)
2. **Exp-G2**: Sanity check with no masking (confirm issue is masking-related)

### Week 2: Quick Wins (Low-Hanging Fruit)
3. **Exp-G5**: ✅ DONE - Fix train/test discrepancy (use hard mask via argmax in test) - **+1.97% gain**
4. **Exp-G7**: Better initialization (bias toward keeping bins)
5. **Exp-G4**: ⚠️ TESTED & REJECTED - Temperature annealing hurts performance
   - **Constant tau=2.0**: 90.53% ✅
   - **Annealing 2.0→0.3**: 88.80% ❌ (-1.73%)
   - **Finding**: High exploration (soft masking) throughout training is better for generalization

### Week 3: Training Improvements
6. **Exp-G3**: ✅ SUCCESS - Add sparsity regularization (weight=0.01) - **+0.44% gain (90.97%!)**
   - **Mask sparsity**: 53.8% of bins kept (35/65) vs 70% before
   - **Pattern discovered**: Keeps all low-freq bins (first 10), drops all high-freq bins (last 10)
   - **Finding**: Low frequencies (0-10 Hz) are most discriminative for HAR
   - **Benefit**: More interpretable + slightly better accuracy
7. **Exp-G8**: Separate learning rates for mask/network
8. **Exp-G6**: Two-stage training (if joint training still unstable)

### Week 4: Alternatives (if gap persists)
9. **Exp-G11**: Try soft attention instead of hard masking
10. **Exp-G12**: FFT normalization ablation

---

## Success Metrics

- **Primary**: Test accuracy ≥ 90%
- **Secondary**:
  - Mask sparsity: 30-70% bins kept (interpretable, efficient)
  - Stability: Val loss should not spike
  - Consistency: Final mask should be reproducible across runs

---

## Implementation Checklist

- [x] Add masking statistics logging (Exp-G1) - prints to console each epoch
- [ ] Create GumbelMask experiment tracking table (like FFT normalization)
- [x] Implement temperature annealing (Exp-G4) - TESTED & REJECTED (constant better)
- [x] Add sparsity loss term (Exp-G3) - weight=0.01, **SUCCESS! 90.97%**
- [x] Fix train/test mask mode discrepancy (Exp-G5) - use hard mask in test
- [ ] Try better initialization strategies
- [ ] Set up separate LR for bin_logits
- [ ] Log final learned mask visualization to W&B
- [ ] Compare learned mask to feature importance from baseline

---

## Summary & Key Findings

### Final Results ✅
- **Target**: 90.0%
- **Achieved**: **90.97%** (exceeds baseline SeparableConvCNN 90.36%)
- **Total improvement**: +3.46% from initial 87.51%

### Successful Experiments
1. **Exp-G5** (train/test fix): +1.97% - Critical fix for hard masking consistency
2. **Exp-G3** (sparsity regularization weight=0.01): +0.44% - Improved selectivity and interpretability (35 bins)
3. **Exp-S2** (sparsity weight=0.05): -1.25% but 35% sparsity - Good accuracy/sparsity trade-off (23 bins)
4. **Constant tau=2.0**: High exploration throughout training works best

### Failed/Suboptimal Experiments
1. **Exp-G4** (temperature annealing): -1.73% - Annealing 2.0→0.3 hurts generalization
2. **Exp-S1** (sparsity weight=0.10): -4.41% - Too aggressive, 18 bins not enough (27.7% kept)

### Learned Mask Pattern (Interpretability)

**Best Accuracy (Exp-G3, sparsity_weight=0.01):**
- **Bins kept**: 35/65 (53.8%), **Accuracy: 90.97%**
- **Low-frequency bins (0-10 Hz)**: ALL kept (100%)
- **High-frequency bins (last 10)**: ALL dropped (0%)
- **Interpretation**: Human activities have most discriminative information in low frequencies (<10 Hz)
- Link: https://wandb.ai/thongp-ubicomp/thesis/runs/0dkov216

**Best Sparsity/Accuracy Balance (Exp-S2, sparsity_weight=0.05):**
- **Bins kept**: 23/65 (35.4%), **Accuracy: 89.72%** (-1.25% from best)
- **Pattern**: First 17 bins + selective mid-freq bins (19, 20, 23, 24, 27, 28)
- **Finding**: Model selectively keeps important mid-frequency components (5-10 Hz) for activity discrimination
- **Trade-off**: 34% fewer bins with only 1.25% accuracy drop

**Too Aggressive (Exp-S1, sparsity_weight=0.10):**
- **Bins kept**: 18/65 (27.7%), **Accuracy: 86.56%** (-4.41%)
- **Pattern**: First 17 bins + bin 19 only
- **Finding**: Only ultra-low frequencies (<5 Hz) retained, too restrictive

### Best Configuration
```python
GumbelMaskSeparableConvCNN(
    num_channels=6,
    freq_bins=65,
    gumbel_tau=2.0,  # constant, no annealing
    dropout=0.4
)

# Training:
# - lr=5e-4
# - sparsity_weight=0.01
# - log1p FFT normalization
```

---

## Reducing Bin Sparsity (Fewer Bins Kept)

**Current State**: 35/65 bins kept (53.8%)  
**Goal**: Achieve higher sparsity (e.g., 20-30 bins = 30-45% kept) while maintaining accuracy

### Strategy S1: Increase Sparsity Regularization Weight
**Current**: `sparsity_weight = 0.01` achieved 53.8% bins kept

**Implementation**:
```python
# In train.py, try progressively higher weights:
sparsity_weight = 0.02  # Expected: ~40-45% bins kept
sparsity_weight = 0.05  # Expected: ~30-35% bins kept
sparsity_weight = 0.10  # Expected: ~20-25% bins kept

loss = loss + sparsity_weight * model.mask_l1
```

**Expected**: Higher weight → stronger penalty for keeping bins → fewer bins selected  
**Risk**: Too high (>0.1) may degrade accuracy significantly  
**Recommendation**: Try [0.02, 0.03, 0.05] first

---

### Strategy S2: Target Sparsity Loss
**Goal**: Penalize deviation from specific target sparsity (e.g., 30% bins)

**Implementation**:
```python
# In train.py:
target_sparsity = 0.30  # target 30% of bins kept
sparsity_target_weight = 0.1

# Add to loss:
sparsity_delta = torch.abs(model.mask_l1 - target_sparsity)
loss = loss + sparsity_target_weight * sparsity_delta
```

**Advantage**: Direct control over sparsity level (not just "fewer")  
**Expected**: Converges to exactly ~20 bins (30% of 65)

---

### Strategy S3: L0/L1 Regularization on Logits
**Goal**: Penalize non-zero logit differences directly

**Implementation**:
```python
# In train.py:
# L1 on logit differences (encourages "off" state)
logit_diff = model.bin_logits[:, 1] - model.bin_logits[:, 0]  # positive = "on"
logit_l1 = torch.abs(logit_diff).mean()
loss = loss + 0.01 * logit_l1
```

**Alternative - L0 approximation** (continuous relaxation):
```python
# Hoyer regularization (promotes exact zeros)
l1_norm = torch.abs(logit_diff).sum()
l2_norm = torch.sqrt((logit_diff ** 2).sum())
hoyer = l1_norm / (l2_norm + 1e-8)
loss = loss + 0.01 * hoyer
```

**Expected**: More aggressive sparsity than L1 on mask

---

### Strategy S4: Entropy Regularization
**Goal**: Encourage confident on/off decisions (reduce "uncertain" bins sitting at 0.5)

**Implementation**:
```python
# In model.py forward():
probs_soft = torch.softmax(self.bin_logits, dim=-1)
entropy = -(probs_soft * torch.log(probs_soft + 1e-8)).sum(dim=-1).mean()

# Lower entropy = more confident decisions
self.mask_entropy = entropy

# In train.py:
entropy_weight = 0.05
loss = loss - entropy_weight * model.mask_entropy  # negative to minimize entropy
```

**Expected**: Forces bins to choose "clearly on" or "clearly off" (no middleground)  
**May help**: Push 53.8% down by eliminating marginal bins

---

### Strategy S5: Progressive Sparsity Annealing
**Goal**: Start lenient (allow many bins), gradually enforce stricter sparsity

**Implementation**:
```python
# In train.py training loop:
def get_sparsity_weight(epoch, max_epochs, start_weight=0.001, end_weight=0.1):
    # Linear increase in sparsity penalty
    return start_weight + (end_weight - start_weight) * (epoch / max_epochs)

# Each epoch:
sparsity_weight = get_sparsity_weight(epoch, epochs)
loss = loss + sparsity_weight * model.mask_l1
```

**Schedule example**: 0.001 → 0.1 over 60 epochs  
**Expected**: Gentle start (model learns which bins are useful), aggressive end (force sparsity)

---

### Strategy S6: Top-K Hard Constraint
**Goal**: Force exactly K bins to be kept (no more, no less)

**Implementation**:
```python
# In model.py forward():
K = 20  # keep exactly 20 bins

# Get top-K bins by logit score
logit_scores = self.bin_logits[:, 1] - self.bin_logits[:, 0]  # "on" preference
_, top_k_indices = torch.topk(logit_scores, K)

# Create hard mask
mask = torch.zeros(freq_bins)
mask[top_k_indices] = 1.0
```

**Advantage**: Guarantees exact sparsity (e.g., 20/65 = 30.8%)  
**Disadvantage**: Not differentiable (need straight-through estimator)  
**Use case**: When you need exact bin budget (e.g., inference constraints)

---

### Strategy S7: Initialization Bias Toward "Off"
**Goal**: Start with most bins off, let training turn essential ones back on

**Implementation**:
```python
# In model.py __init__:
# Bias toward "off" state
self.bin_logits = nn.Parameter(torch.zeros(freq_bins, 2))
nn.init.constant_(self.bin_logits[:, 0], 0.5)   # bias toward "off"
nn.init.constant_(self.bin_logits[:, 1], -0.5)  # discourage "on"

# Or more aggressive:
self.bin_logits[:, 0] = 1.0   # strong "off" bias
self.bin_logits[:, 1] = -1.0
```

**Expected**: Starts with ~20-30% bins, training adds essential bins  
**Risk**: May converge to local minimum with too few bins

---

### Strategy S8: Combine Multiple Strategies
**Recommended combination** for aggressive sparsity:

```python
# In train.py:
# 1. Target sparsity (30%)
target_sparsity = 0.30
loss = loss + 0.1 * torch.abs(model.mask_l1 - target_sparsity)

# 2. Entropy minimization (confident decisions)
loss = loss - 0.02 * model.mask_entropy

# 3. Progressive annealing (easier start)
sparsity_weight = get_sparsity_weight(epoch, epochs, 0.005, 0.03)
loss = loss + sparsity_weight * model.mask_l1
```

**Expected**: Converges to ~20 bins with high confidence

---

### Strategy S9: Post-Training Pruning
**Goal**: Train with current sparsity, then prune lowest-scoring bins

**Implementation**:
```python
# After training:
K = 20  # desired bins
logit_scores = model.bin_logits[:, 1] - model.bin_logits[:, 0]
_, top_k_indices = torch.topk(logit_scores, K)

# Set pruned bins to hard "off"
model.bin_logits.data[:, 1] = -10  # all off
model.bin_logits.data[top_k_indices, 1] = 10  # top-K on
model.bin_logits.data[top_k_indices, 0] = -10

# Fine-tune for a few epochs
# Network adapts to fixed mask
```

**Advantage**: Maintains 90.97% during training, only increases sparsity at end  
**Use case**: When you want guaranteed performance, then compress

---

## Recommended Sparsity Reduction Plan

### Phase 1: Gentle Increase (Exp-S1)
**Goal**: Reach 40-45% bins kept (26-29 bins)

```python
sparsity_weight = 0.02
```

**Expected**: 89.5-90.5% accuracy (minimal drop)

### Phase 2: Moderate Sparsity (Exp-S2)
**Goal**: Reach 30-35% bins kept (20-23 bins)

```python
target_sparsity = 0.33
loss = loss + 0.1 * torch.abs(model.mask_l1 - target_sparsity)
```

**Expected**: 88.5-89.5% accuracy (~1-2% drop)

### Phase 3: Aggressive Sparsity (Exp-S3)
**Goal**: Reach 20-25% bins kept (13-16 bins)

```python
# Combine strategies:
target_sparsity = 0.23
loss = loss + 0.15 * torch.abs(model.mask_l1 - target_sparsity)
loss = loss - 0.03 * model.mask_entropy  # confident decisions
```

**Expected**: 87-89% accuracy (~2-4% drop)  
**Benefit**: Highly interpretable (only most critical bins)

### Phase 4: Post-Pruning (Exp-S4)
**Goal**: Keep 90.97% accuracy, then prune to 20 bins and fine-tune

1. Train with current config (35 bins, 90.97%)
2. Prune to top-20 bins by logit score
3. Fine-tune network with frozen mask (5-10 epochs)

**Expected**: 89-90% accuracy (smaller drop than training sparse from scratch)

---

## Trade-off Analysis

| Bins Kept | % Kept | Strategy | Actual Accuracy | Accuracy Drop | Notes |
|-----------|--------|----------|-----------------|---------------|-------|
| **35** | 53.8% | sparsity_weight=0.01 | **90.97%** ✅ | baseline | Best accuracy |
| **23** | 35.4% | sparsity_weight=0.05 | **89.72%** ✅ | -1.25% | **Best balance** |
| **18** | 27.7% | sparsity_weight=0.10 | **86.56%** ⚠️ | -4.41% | Too aggressive |
| **~20** (expected) | ~30% | target_sparsity=0.30 | Testing now... | Expected: -1.5 to -2% | Exp-S3 |

**Key Findings**: 
- **Sweet spot**: sparsity_weight=0.05 achieves 35% sparsity with minimal accuracy loss (1.25%)
- **Diminishing returns**: Going from 35→23 bins saves only 1.25%, but 23→18 bins costs 3.16% more

---

## Implementation Priority

### High Priority (Easy Wins)
- [x] **Exp-G3**: sparsity_weight=0.01 → 35 bins (53.8%), **90.97%** ✅ Best accuracy
- [x] **Exp-S1**: sparsity_weight=0.10 → 18 bins (27.7%), **86.56%** ⚠️ Too aggressive
- [x] **Exp-S2**: sparsity_weight=0.05 → 23 bins (35.4%), **89.72%** ✅ Best balance
- [ ] **Exp-S3**: Target sparsity loss (target=0.30) → paused (reverted to Exp-G3)

### Medium Priority (More Control)
- [ ] **Exp-S4**: Progressive annealing (0.01→0.05)

### Advanced (If Needed)
- [ ] **Exp-S5**: Entropy regularization + target sparsity
- [ ] **Exp-S6**: Post-training pruning + fine-tuning (preserves accuracy best)
- [ ] **Exp-S7**: Top-K hard constraint (for guaranteed bin budget)

---

## Open Questions

1. **Why use masking?** 
   - Efficiency (fewer bins → faster inference)?
   - Interpretability (which freq bins matter)?
   - Regularization (prevent overfitting)?
   
2. **Is 87.51% with masking acceptable if mask is interpretable?**
   - Trade-off: -2.5% accuracy for explainability?

3. **Should masking be per-channel or global?**
   - Current: same mask for all channels
   - Alternative: different masks for accel vs. gyro

4. **What is the optimal sparsity target?**
   - Computational efficiency: 20-30% bins (faster inference)
   - Interpretability: 15-25% bins (clearer insights)
   - Accuracy preservation: 40-50% bins (safer)

---

Last updated: 2026-02-25
