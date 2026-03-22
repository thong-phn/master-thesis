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
        FFT with Gumbel stop at epoch 4; LR reduce in 2nd stage; sparsity weight: 0.01; Accuracy: 71.18%; Mask: 90%
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
