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