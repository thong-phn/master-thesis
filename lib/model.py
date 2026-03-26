
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelMaskSeparableConvCNN(nn.Module):
    """
    SeparableConv-based CNN with Gumbel-Softmax bin on/off masking.
    """
    def __init__(self, num_classes=6, num_channels=6, freq_bins=65, dropout=0.4, gumbel_tau=2.0, tau_start=5.0, tau_end=1.0):
        super(GumbelMaskSeparableConvCNN, self).__init__()

        # Two-class logits per bin: [off, on]
        self.bin_logits = nn.Parameter(torch.zeros(freq_bins, 2))
        
        # Tau annealing configuration
        self.gumbel_tau = gumbel_tau
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.current_tau = tau_start  # Initialize with tau_start
        self.mask_l1 = None
        self.last_mask = None

        # Stem block
        self.bn0 = nn.BatchNorm1d(num_channels)
        self.sep_conv1 = SeparableConv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Separable conv blocks
        self.sep_conv2 = SeparableConv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.sep_conv3 = SeparableConv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.sep_conv4 = SeparableConv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, freq_bins)

        batch_size = x.size(0)

        if self.training:
            # Exp-G4: Use annealed temperature
            logits_expanded = self.bin_logits.unsqueeze(0).expand(batch_size, -1, -1)
            probs = F.gumbel_softmax(logits_expanded, tau=self.current_tau, hard=True)
        else:
            # Exp-G5: Use hard mask in test (argmax) to match training behavior
            logits_expanded = self.bin_logits.unsqueeze(0).expand(batch_size, -1, -1)
            probs_soft = torch.softmax(logits_expanded, dim=-1)
            probs = F.one_hot(probs_soft.argmax(dim=-1), num_classes=2).float()

        mask = probs[:, :, 1] # shape (batch, freq_bins)
        
        # Track statistics based on the expected mask or mean batch mask
        self.mask_l1 = mask.mean()
        self.last_mask = mask.mean(dim=0).detach() # store the mean frequency mask across the batch

        x = x * mask.unsqueeze(1) # shape (batch, 1, freq_bins)

        # Stem
        x = self.bn0(x)
        x = F.relu(self.sep_conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.sep_conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.sep_conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.sep_conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def set_tau(self, epoch, max_epochs):
        """Exp-G4: Anneal temperature from tau_start to tau_end over training."""
        # Linear annealing
        progress = epoch / max(max_epochs, 1)
        self.current_tau = self.tau_start - (self.tau_start - self.tau_end) * progress

class SeparableConv1d(nn.Module):
    """Depthwise Separable Convolution (Depthwise + Pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(SeparableConv1d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SeparableConvCNN(nn.Module):
    """
    SeparableConv-based CNN for UCI-HAR FFT data
    Based on depthwise separable convolutions for efficiency
    """
    def __init__(self, num_classes=6, num_channels=6, freq_bins=65, dropout=0.4):
        super(SeparableConvCNN, self).__init__()
        
        # Input shape: (batch, num_channels, 31) where num_channels is 3 (accel) or 6 (accel+gyro)
        
        # Stem block
        self.bn0 = nn.BatchNorm1d(num_channels)
        self.sep_conv1 = SeparableConv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)  # 31 -> 15
        
        # Separable conv blocks
        self.sep_conv2 = SeparableConv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)  # 32 -> 16
        
        self.sep_conv3 = SeparableConv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)  # 16 -> 8
        
        self.sep_conv4 = SeparableConv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)  # 8 -> 4
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: (batch, num_channels, 31)
        
        # Stem
        x = self.bn0(x)
        x = F.relu(self.sep_conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.sep_conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.sep_conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.sep_conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)
        
        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class GumbelChannelPruningCNN(nn.Module):
    """
    SeparableConv-based CNN with Gumbel-Softmax channel masking after conv blocks.
    """

    def __init__(self, num_classes=6, num_channels=6, freq_bins=65, dropout=0.4, tau_start=5.0, tau_end=1.0):
        super(GumbelChannelPruningCNN, self).__init__()

        self.tau_start = tau_start
        self.tau_end = tau_end
        self.current_tau = tau_start
        # Compatibility with existing training logs that inspect model.mask_l1.
        self.mask_l1 = None # sparsity mask for l1 regularization

        # Block 1: Stem block (no pruning due to feature extract & not much params compared to Block 2,3,4)
        self.bn0 = nn.BatchNorm1d(num_channels)
        self.sep_conv1 = SeparableConv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Block 2
        self.sep_conv2 = SeparableConv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.chan_logits_2 = nn.Parameter(torch.zeros(64, 2))

        # Block 3
        self.sep_conv3 = SeparableConv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.chan_logits_3 = nn.Parameter(torch.zeros(128, 2))

        # Block 4
        self.sep_conv4 = SeparableConv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)
        self.chan_logits_4 = nn.Parameter(torch.zeros(128, 2))

        self.register_buffer('last_mask_2', torch.ones(64))
        self.register_buffer('last_mask_3', torch.ones(128))
        self.register_buffer('last_mask_4', torch.ones(128))

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _get_channel_mask(self, logits, batch_size):
        """
        logits: (C, 2) -> (B, C, 2)
        training: 
        """
        logits_expanded = logits.unsqueeze(0).expand(batch_size, -1, -1)
        # Train: Different mask for differnt samples (due to different gumbel noise)
        # The amount of randomness reduced as tau reduced
        if self.training:
            probs = F.gumbel_softmax(logits_expanded, tau=self.current_tau, hard=True)
            return probs[:, :, 1]
        # Eval: Same mask for all samples
        probs_soft = torch.softmax(logits_expanded, dim=-1)
        return F.one_hot(probs_soft.argmax(dim=-1), num_classes=2).float()[:, :, 1]

    def forward(self, x):
        batch_size = x.size(0)

        # Stem
        x = self.bn0(x)
        x = F.relu(self.bn1(self.sep_conv1(x)))
        x = self.pool1(x)

        # Block 2: apply mask after BN+ReLU
        x = F.relu(self.bn2(self.sep_conv2(x)))
        mask2 = self._get_channel_mask(self.chan_logits_2, batch_size)
        self.last_mask_2 = mask2.mean(dim=0).detach()
        x = x * mask2.unsqueeze(-1)
        x = self.pool2(x)

        # Block 3: apply mask after BN+ReLU
        x = F.relu(self.bn3(self.sep_conv3(x)))
        mask3 = self._get_channel_mask(self.chan_logits_3, batch_size)
        self.last_mask_3 = mask3.mean(dim=0).detach()
        x = x * mask3.unsqueeze(-1)
        x = self.pool3(x)

        # Block 4: apply mask after BN+ReLU
        x = F.relu(self.bn4(self.sep_conv4(x)))
        mask4 = self._get_channel_mask(self.chan_logits_4, batch_size)
        self.last_mask_4 = mask4.mean(dim=0).detach()
        x = x * mask4.unsqueeze(-1)
        x = self.pool4(x)

        # Average "on" probability for training logs.
        self.mask_l1 = self.get_sparsity_loss()

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def set_tau(self, epoch, max_epochs):
        progress = epoch / max(max_epochs, 1)
        self.current_tau = self.tau_start - (self.tau_start - self.tau_end) * progress

    def get_sparsity_loss(self):
        tau = max(self.current_tau, 1e-6)
        p2 = torch.softmax(self.chan_logits_2 / tau, dim=-1)[:, 1].mean()
        p3 = torch.softmax(self.chan_logits_3 / tau, dim=-1)[:, 1].mean()
        p4 = torch.softmax(self.chan_logits_4 / tau, dim=-1)[:, 1].mean()
        return (p2 + p3 + p4) / 3.0

    @torch.no_grad()
    def get_hard_masks(self):
        """Return deterministic hard channel masks from logits via argmax([off, on])."""
        m2 = torch.softmax(self.chan_logits_2, dim=-1).argmax(dim=-1).float()
        m3 = torch.softmax(self.chan_logits_3, dim=-1).argmax(dim=-1).float()
        m4 = torch.softmax(self.chan_logits_4, dim=-1).argmax(dim=-1).float()
        return {
            'block2': m2,
            'block3': m3,
            'block4': m4,
        }

    @torch.no_grad()
    def get_pruning_stats(self):
        hard_masks = self.get_hard_masks()
        c2 = hard_masks['block2'].sum().item()
        c3 = hard_masks['block3'].sum().item()
        c4 = hard_masks['block4'].sum().item()
        total = 64 + 128 + 128
        return {
            'Block2_Pruned_%': (1 - c2 / 64.0) * 100.0,
            'Block3_Pruned_%': (1 - c3 / 128.0) * 100.0,
            'Block4_Pruned_%': (1 - c4 / 128.0) * 100.0,
            'Total_Pruned_%': (1 - (c2 + c3 + c4) / total) * 100.0,
        }


