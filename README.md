Note:
Accuracy 
    Body acc works better than total acc
    Without tau annealing
        - With self.bn0: 91.38% - 41% mask
        - Without self.bn0: 91.89% - 47% mask
    Tau annealing
        - With self.bn0: 92.09% - 46% mask
        - Without self.bn0: 89.55% - 41.5% mask

Size
    32 -> 64 -> 128 -> 128: 92%
    16 -> 32 -> 64 -> 64: 89 (half of parameters remove)

Results:
Gumbel mask (tau annealing from 3.0 to 2.0)
    Test Accuracy: 89.08% ± 2.30%
    Test F1 Macro: 0.8914 ± 0.0230
Gumbel mask (tau annealing from 5.0 to 1.0)
    Test Accuracy: 89.32% ± 1.92%
    Test F1 Macro: 0.8934 ± 0.0196
Gumbel mask (tau annealing from 5.0 to 1.0, remove weigh-sparity ramp up, sparsity_weight = 0.01)
    Test Accuracy: 89.67% ± 1.66%
    Test F1 Macro: 0.8974 ± 0.0165
Gumbel mask (tau annealing from 5.0 to 1.0, remove weigh-sparity ramp up, sparsity_weight = 0.005)
    Test Accuracy: 89.67% ± 1.66%
    Test F1 Macro: 0.8974 ± 0.0165


Add WEAR debug:
    Observation
        Gummbel Mask: Stop at epoch 1
        Raw time-domain without Gummbel Mask: Stop at epoch 4; 73.95%
        Raw FFT without Gummbel Mask: Stop at epoch 1; 72.38%
        GUMBEL_MASK
        --preprocessing 'fft' --sparsity_weight 0.0001: Test Loss: 0.7397 | Test Acc: 72.38% | Test F1 Macro: 0.6733
    Issue: FFT data seem not working
    Cause: 
        FFT compresses 100 samples → 51 bins (losing temporal sequence info),
        Large DC/low-frequency dominance: The DC component (gravity) and first few bins dominate the FFT spectrum. 
        Same hyperparameters tuned for UCI-HAR: 
            Metric	UCI-HAR FFT ✅	WEAR FFT ❌	WEAR raw ✅
            Shape	(6, 65)	(6, 51)	(6, 100)
            Mean	0.010	0.107	0.433
            Std	0.035	0.198	1.156
            Max	0.685	1.955	6.133
            DC (bin 0)	0.022	0.557	—
    Solution: Gravity filter
        Raw FFT without Gummbel stop at epoch 4; Accuracy: 71.06% 
        FFT with Gumbel stop at epoch 4; Accuracy: 57.88% -> Using UCI-HAR give comparable accuracy
Add WEAR debug 2: Try with 2 stage training (pre-trained the model without Gumbel)
        FFT with Gumbel stop at epoch 2; Accuracy: 70.63%; Mask: 80%
        FFT with Gumbel stop at epoch 4; LR reduce in 2nd stage; Accuracy: 71.18%; Mask: 90%
        FFT with Gumbel stop at epoch 4; LR reduce in 2nd stage; sparsity weight: 0.1; Accuracy: 70.46%; Mask: 60
        FFT with Gumbel stop at epoch 4; LR reduce in 2nd stage; sparsity weight: 0.5; Accuracy: 45.80%; Mask: 60
        FFT with Gumbel stop at epoch 4; LR reduce in 2nd stage; sparsity weight: 1; Accuracy: 20.46%; Mask: 0 -> limit

Note: Typical pipeline: Accel -> Remove g -> FFT -> Normalized
TODO: 
- Add LRReduce
- Try with different model
- Why joint optimization not work but 2 stage work?
Dataset: 
UCI-HAR: Locomotion
Wetlab: Controlled Experiment
Handwashing: Daily life

WEAR: Sport

Weighted CEL
    Subject 0, no weighted:
        STAGE 1
        Test Accuracy: 71.14% ± 0.00%
        Test F1 Macro: 0.6580 ± 0.0000

        STAGE2:
        Test Accuracy: 68.41% ± 0.00%
        Test F1 Macro: 0.6327 ± 0.0000

        STAGE3:
        Test Accuracy: 72.81% ± 0.00%
        Test F1 Macro: 0.6633 ± 0.0000

        STAGE4:
        Test Accuracy: 70.63% ± 0.00%
        Test F1 Macro: 0.6508 ± 0.0000

        STAGE5:
        Test Accuracy: 70.89% ± 0.00%
        Test F1 Macro: 0.6518 ± 0.0000

        Improvement (Stage5 - Stage1):
        Accuracy: -0.25%
        F1 Macro: -0.0062
    Subject 0, weighted, keep less input?:
        STAGE1:
        Test Accuracy: 79.20% ± 0.00%
        Test F1 Macro: 0.6841 ± 0.0000

        STAGE2:
        Test Accuracy: 73.01% ± 0.00%
        Test F1 Macro: 0.6336 ± 0.0000

        STAGE3:
        Test Accuracy: 77.89% ± 0.00%
        Test F1 Macro: 0.6823 ± 0.0000

        STAGE4:
        Test Accuracy: 79.97% ± 0.00%
        Test F1 Macro: 0.6851 ± 0.0000

        STAGE5:
        Test Accuracy: 79.73% ± 0.00%
        Test F1 Macro: 0.6884 ± 0.0000

        Improvement (Stage5 - Stage1):
        Accuracy: 0.54%
        F1 Macro: 0.0043

About bn0 issue
    Debug thêm với subject 7: 
        No bn0: Prune quá nhiều 
            Baseline stage 1: Test Acc: da co
        With bn0: Prune vừa phải
            Baseline da co
        No bn0 + Zscore normalization: 54.39% ± 0.00%
        With bn0 + Zscore norm: 56
    Input pruning S0: Train samples: 50686; Val samples: 2793; Test samples: 9705.
    NOTE: Stage 3 (Fine tuning pruned input)
        No bn0, sparsity weight = 0.1 -> Maintain accuracy
            - Stage 1: Test Acc: 71.14% | Test F1 Macro: 0.6580
            - Stage 2: Test Acc: 68.41% | Test F1 Macro: 0.6327 | Hard input bins kept: 30/51 (58.8%)
            - Stage 3: Test Acc: 71.55% | Test F1 Macro: 0.6606
        With bn0, sparsity weight = 0.1 -> Higher at the begin but drop
            - Stage 1: Test Acc: 73.33% | Test F1 Macro: 0.6610
            - Stage 2: Test Acc: 69.58% | Test F1 Macro: 0.6342 | Hard input bins kept: 36/51 (70.6%)
            - Stage 3: Test Acc: 70.93% | Test F1 Macro: 0.6570 
    Input pruning S12: Train samples: 50210; Val samples: 3269; Test samples: 9705
        No bn0, sparsity weight = 0.1
            - Stage 1: Test Acc: 71.68% | Test F1 Macro: 0.6592
            - Stage 2: Test Acc: 71.54% | Test F1 Macro: 0.6619 | Hard input bins kept: 27/51 (52.9%)
            - Stage 3: Test Acc: 72.46% | Test F1 Macro: 0.6677
        With bn0, sparsity weight = 0.1
            - Stage 1: Test Acc: 72.34% | Test F1 Macro: 0.6660
            - Stage 2: Test Acc: 70.48% | Test F1 Macro: 0.6580 | Hard input bins kept: 25/51 (49.0%)
            - Stage 3: Test Acc: 70.47% | Test F1 Macro: 0.6547

SPARSITY_WEIGHT
- FFT: 0.09
- IHW: 0.07
- DCT: 


wCEL additional input pruning tuning
S0:
Stage1: Test Accuracy: 79.20% ± 0.00% | Test F1 Macro: 0.6841 ± 0.0000
- SPARSITY_WEIGHT: 0.2 ->  Test Accuracy: 79.22% ± 0.00% | Test F1 Macro: 0.6881 ± 0.0000 | Input bins: 26/51 (51.0%)
- SPARSITY_WEIGHT: 0.25 -> kept bins=11/51
- SPARSITY_WEIGHT: 0.3 ->  Input bins: 10/51 (19.6%)

S7: 
Stage 1: Test Accuracy: 72.68% ± 0.00% | Test F1 Macro: 0.6554 ± 0.0000
- SPARSITY_WEIGHT: 0.2 -> Test Accuracy: 74.57% ± 0.00% | Test F1 Macro: 0.6692 ± 0.0000 | Hard input bins kept: 26/51 (51.0%)
- SPARSITY_WEIGHT: 0.3 -> Test Acc: 77.28%% | Test F1 Macro: 0.6757 | Hard input bins kept: 16/51 (31.4%)

Baseline 
WEAR
TD: Test Accuracy: 81.22% ± 2.13% | Test F1 Macro: 0.7180 ± 0.0313 (link: )
DCT: Test Accuracy: 78.80% ± 2.17% | Test F1 Macro: 0.6864 ± 0.0144

FFT: Test Accuracy: 77.89% ± 1.77% | Test F1 Macro: 0.6788 ± 0.0115
Five stage