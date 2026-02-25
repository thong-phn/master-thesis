# Lightweight GumbelMask Model Plan

## Objective
Design a parameter-efficient model with integrated Gumbel masking that achieves 90-91% accuracy with significantly fewer parameters than the current GumbelMaskSeparableConvCNN.

## Current Baseline
- **Model**: GumbelMaskSeparableConvCNN
- **Accuracy**: 90.97%
- **Architecture**: 4 separable conv blocks (32→64→128→128 channels) + 2 FC layers (128→64→6)
- **Parameter count**: **~37,000 parameters**
- **Key insight**: Only 35/65 frequency bins needed (53.8%), focused on low frequencies

## Target Goals
- **Primary**: Achieve 90-91% accuracy (max 1% drop from 90.97%)
- **Secondary**: Reduce parameters by 50-70% (target: 11k-18.5k params)
  - 50% reduction: ~18.5k parameters
  - 60% reduction: ~14.8k parameters
  - 70% reduction: ~11.1k parameters
- **Tertiary**: Maintain interpretability via Gumbel masking

---

## Parameter Analysis

### Current Model Breakdown
```
COMPONENT                    | PARAMETERS
-----------------------------|------------
Gumbel bin_logits (65×2)     | 130
Separable Conv blocks        | ~25,000 (estimated)
FC layers (128→64→6)         | ~8,600 (128×64 + 64×6)
BatchNorm                    | ~3,300 (estimated)
-----------------------------|------------
TOTAL                        | ~37,000
```

**Bottleneck**: Separable conv blocks (~68% of params) and FC layers (~23%)

---

## Lightweight Architecture Strategies

### Strategy 1: Channel Reduction
**Rationale**: Reduce channel widths while maintaining architectural depth
- Current: 32→64→128→128
- Proposed variants:
  - **L1-Narrow**: 16→32→64→64 (50% reduction)
  - **L2-Minimal**: 12→24→48→48 (62.5% reduction)
  - **L3-Tiny**: 8→16→32→32 (75% reduction)

### Strategy 2: Layer Reduction
**Rationale**: Fewer conv blocks, rely on Gumbel mask to select important features
- Current: 4 separable conv blocks
- Proposed variants:
  - **L4-Shallow**: 3 blocks (32→64→128)
  - **L5-Minimal**: 2 blocks (32→64)

### Strategy 3: Frequency-Aware Design
**Rationale**: Exploit learned mask insight (only low-freq bins matter)
- **L6-FreqFocus**: Process only first 40 bins (mask shows 35/65 needed)
  - Reduces input dimension, fewer computations
  - Add optional frequency bin selection layer before Gumbel mask

### Strategy 4: Hybrid Separable + Standard Convs
**Rationale**: Mix separable (efficient) with standard (expressive) convolutions
- **L7-Hybrid**: 
  - First 2 blocks: Separable (for efficiency)
  - Last 1-2 blocks: Standard 1x1 pointwise (for feature mixing)

### Strategy 5: Knowledge Distillation
**Rationale**: Transfer knowledge from current 90.97% model to lightweight student
- **L8-Distilled**: 
  - Train lightweight model (L1-L3) with soft targets from current model
  - Loss = CrossEntropy + KL-divergence(student_logits, teacher_logits)

### Strategy 6: MobileNet-Inspired Inverted Residuals
**Rationale**: Use efficient inverted bottleneck blocks (expand→depthwise→project)
- **L9-MobileStyle**:
  - Inverted residual blocks with expansion factor 2-4
  - Skip connections for gradient flow
  - Very parameter-efficient

---

## Experiment Plan

### Phase 1: Baseline Parameter Analysis (Week 1)
**Exp-L0**: Calculate exact parameter count of current GumbelMaskSeparableConvCNN
- Add `count_parameters()` utility function
- Log to W&B for comparison
- Identify bottleneck layers (where most parameters are)

### Phase 2: Simple Channel Reduction (Week 1)
**Exp-L1**: Test 16→32→64→64 architecture (50% channel reduction)
- Keep 4 separable conv blocks
- Keep Gumbel mask + sparsity reg (weight=0.01)
- Keep tau=2.0, lr=5e-4, log1p normalization
- **Expected**: 89-90% accuracy, ~18.5k params (50% reduction from 37k)

**Exp-L2**: Test 12→24→48→48 architecture (62.5% reduction)
- Same config as L1
- **Expected**: 88-89% accuracy, ~13-15k params (60-65% reduction)

**Exp-L3**: Test 8→16→32→32 architecture (75% reduction)
- Same config as L1
- **Expected**: 85-88% accuracy, ~9-10k params (may be too aggressive)

### Phase 3: Layer Reduction (Week 2)
**Exp-L4**: 3-block architecture (32→64→128)
- Remove 4th block, adjust pooling
- **Expected**: 89-90% accuracy, ~25-27k params (30% reduction)

**Exp-L5**: 2-block architecture (32→64)
- Minimal depth, rely on Gumbel mask for feature selection
- May need to increase channel width to compensate
- **Expected**: 87-89% accuracy, ~18.5k params (50% reduction)

### Phase 4: Frequency-Aware Optimization (Week 2)
**Exp-L6**: Process only first 40 frequency bins
- Truncate FFT output to 40 bins (covers learned mask's 35 bins)
- Use best architecture from Phase 2-3
- **Expected**: Similar accuracy, faster inference

### Phase 5: Advanced Architectures (Week 3)
**Exp-L7**: Hybrid separable + standard conv
- First 2 blocks: Separable
- Last block: 1x1 standard conv for feature mixing
- **Expected**: Similar accuracy, slight parameter reduction

**Exp-L9**: MobileNet-inspired inverted residuals
- 3 inverted residual blocks (expansion=3)
- Gumbel mask at input
- **Expected**: 89-91% accuracy, ~11-15k params (60-70% reduction, most promising)

### Phase 6: Knowledge Distillation (Week 3-4)
**Exp-L8**: Distill from current 90.97% model
- Use best lightweight architecture from Phase 2-5
- Loss = CE + 0.5 * KL(student || teacher, T=4)
- **Expected**: +1-2% boost over standalone lightweight model

---

## Implementation Details

### Base Lightweight Model Template
```python
class LightweightGumbelCNN(nn.Module):
    def __init__(self, channels=[16, 32, 64, 64], num_blocks=4, 
                 num_channels=6, freq_bins=65, dropout=0.3, gumbel_tau=2.0):
        super().__init__()
        
        # Gumbel mask (minimal parameters)
        self.bin_logits = nn.Parameter(torch.zeros(freq_bins, 2))
        self.gumbel_tau = gumbel_tau
        self.mask_l1 = None
        
        # Flexible channel progression
        self.blocks = nn.ModuleList()
        in_ch = num_channels
        for out_ch in channels:
            self.blocks.append(SeparableConvBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # Lightweight classifier
        self.fc = nn.Linear(channels[-1], 6)
```

### MobileNet-Style Inverted Residual (Most Promising)
```python
class InvertedResidual(nn.Module):
    """Efficient inverted bottleneck: expand → depthwise → project"""
    def __init__(self, in_ch, out_ch, expansion=3, stride=1):
        super().__init__()
        hidden = in_ch * expansion
        self.conv1 = nn.Conv1d(in_ch, hidden, 1, bias=False)  # Expand
        self.bn1 = nn.BatchNorm1d(hidden)
        self.dwconv = nn.Conv1d(hidden, hidden, 3, stride, 1, 
                                groups=hidden, bias=False)  # Depthwise
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, out_ch, 1, bias=False)  # Project
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.use_residual = (stride == 1 and in_ch == out_ch)
    
    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv2(out))
        return out + x if self.use_residual else out
```

### Knowledge Distillation Training Loop
```python
def train_with_distillation(student, teacher, dataloader, temperature=4, alpha=0.5):
    """
    alpha: weight for distillation loss vs hard label loss
    temperature: softmax temperature for soft targets
    """
    for X, y in dataloader:
        # Student forward
        student_logits = student(X)
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = teacher(X)
        
        # Hard label loss
        loss_ce = F.cross_entropy(student_logits, y)
        
        # Soft target loss (KL divergence)
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Combined loss
        loss = (1 - alpha) * loss_ce + alpha * loss_kd
        loss.backward()
```

---

## Success Criteria

### Minimum Viable Model
- **Accuracy**: ≥ 89.5% (max 1.5% drop)
- **Parameters**: ≤ 18.5k (50% of 37k)
- **Inference speed**: ≥ 1.5x faster

### Optimal Model
- **Accuracy**: ≥ 90.0% (max 1% drop)
- **Parameters**: ≤ 14.8k (40% of 37k)
- **Interpretability**: Maintains Gumbel masking insights

### Stretch Goal
- **Accuracy**: ≥ 90.5% (within 0.5% of best)
- **Parameters**: ≤ 11.1k (30% of 37k)
- **Edge deployment**: Suitable for mobile/embedded devices

---

## Implementation Checklist

- [x] **Exp-L0**: Add parameter counting utility, benchmark current model (**37k params**)
- [ ] Create `lightweight_models.py` with modular architectures
- [ ] **Exp-L1**: Test 16→32→64→64 narrow architecture
- [ ] **Exp-L2**: Test 12→24→48→48 minimal architecture
- [ ] **Exp-L4**: Test 3-block depth reduction
- [ ] **Exp-L9**: Implement MobileNet-style inverted residuals (priority)
- [ ] **Exp-L6**: Test frequency-aware 40-bin input
- [ ] **Exp-L8**: Implement knowledge distillation training
- [ ] Create experiment tracking table (accuracy vs parameters vs speed)
- [ ] Visualize parameter-accuracy Pareto frontier
- [ ] Profile inference speed on CPU (simulate edge device)
- [ ] Log model size (MB) and FLOPs to W&B

---

## Expected Outcomes

### Most Promising Candidates
1. **Exp-L9** (MobileNet inverted residuals): 89-91% accuracy, ~11-15k params (60-70% reduction)
2. **Exp-L1** (16→32→64→64): 89-90% accuracy, ~18.5k params (50% reduction)
3. **Exp-L8** (Distillation on L9 or L1): +1-2% boost over standalone

### Risk Mitigation
- If accuracy drops below 89%: Use knowledge distillation (Exp-L8)
- If parameter reduction insufficient: Combine strategies (e.g., L1 + L4 + L6)
- If Gumbel mask becomes unstable: Adjust tau or add noise regularization

---

## Visualization & Analysis

### Plots to Generate
1. **Parameter-Accuracy Scatter**: X=params, Y=accuracy (Pareto frontier)
2. **Architecture Comparison**: Bar chart of params/accuracy for each experiment
3. **Inference Speed**: Box plot of inference time per batch
4. **Learned Masks**: Heatmap comparing Gumbel masks across lightweight models
5. **Model Size**: Pie chart showing parameter distribution by layer

### Metrics to Track in W&B
```python
wandb.log({
    'test_accuracy': acc,
    'model_parameters': count_params,
    'parameter_reduction_%': reduction,
    'inference_time_ms': time_ms,
    'model_size_mb': size_mb,
    'bins_kept': mask.sum(),
    'compression_ratio': original_params / new_params
})
```

---

## Open Questions
1. Should we freeze Gumbel mask from teacher model or re-learn it?
2. What's the minimum channel width before accuracy degrades significantly?
3. Can we quantize the lightweight model for further compression?
4. Should we use different sparsity weights for lightweight models?
5. Does the learned mask pattern change with fewer parameters?

---

## Timeline
- **Week 1**: Phase 1-2 (baseline + channel reduction experiments)
- **Week 2**: Phase 3-4 (layer reduction + frequency optimization)
- **Week 3**: Phase 5 (advanced architectures, prioritize MobileNet style)
- **Week 4**: Phase 6 (knowledge distillation) + final analysis
