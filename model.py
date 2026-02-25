
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelMaskSeparableConvCNN(nn.Module):
    """
    SeparableConv-based CNN with Gumbel-Softmax bin on/off masking.
    """
    def __init__(
        self,
        num_classes=6,
        num_channels=3,
        freq_bins=65,
        dropout=0.4,
        gumbel_tau=2.0,
        tau_start=2.0,
        tau_end=2.0,
        channels=(32, 64, 128, 128),
        fc_hidden=64,
    ):
        super(GumbelMaskSeparableConvCNN, self).__init__()

        if not (2 <= len(channels) <= 4):
            raise ValueError(f"Expected 2 to 4 channel values, got {len(channels)}: {channels}")
        self.channels = tuple(channels)
        c1, c2 = self.channels[0], self.channels[1]
        self.has_block3 = len(self.channels) >= 3
        self.has_block4 = len(self.channels) >= 4
        if self.has_block3:
            c3 = self.channels[2]
        if self.has_block4:
            c4 = self.channels[3]

        # Two-class logits per bin: [off, on]
        self.bin_logits = nn.Parameter(torch.zeros(freq_bins, 2))
        # Exp-G4 tested: Constant tau=2.0 works better than annealing
        self.gumbel_tau = gumbel_tau
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.current_tau = gumbel_tau  # Use constant tau
        self.mask_l1 = None
        self.last_mask = None

        # Stem block
        self.bn0 = nn.BatchNorm1d(num_channels)
        self.sep_conv1 = SeparableConv1d(num_channels, c1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(c1)
        self.pool1 = nn.MaxPool1d(2)

        # Separable conv blocks
        self.sep_conv2 = SeparableConv1d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool2 = nn.MaxPool1d(2)

        if self.has_block3:
            self.sep_conv3 = SeparableConv1d(c2, c3, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(c3)
            self.pool3 = nn.MaxPool1d(2)

        if self.has_block4:
            self.sep_conv4 = SeparableConv1d(c3, c4, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm1d(c4)
            self.pool4 = nn.MaxPool1d(2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.channels[-1], fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, freq_bins)

        if self.training:
            # Exp-G4: Use annealed temperature
            probs = F.gumbel_softmax(self.bin_logits, tau=self.current_tau, hard=True)
        else:
            # Exp-G5: Use hard mask in test (argmax) to match training behavior
            probs_soft = torch.softmax(self.bin_logits, dim=-1)
            probs = F.one_hot(probs_soft.argmax(dim=-1), num_classes=2).float()

        mask = probs[:, 1]
        self.mask_l1 = mask.mean()
        self.last_mask = mask.detach()
        x = x * mask.view(1, 1, -1)

        # Stem
        x = self.bn0(x)
        x = F.relu(self.sep_conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.sep_conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        # Block 3 (optional)
        if self.has_block3:
            x = F.relu(self.sep_conv3(x))
            x = self.bn3(x)
            x = self.pool3(x)

        # Block 4 (optional)
        if self.has_block4:
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


class InvertedResidual1d(nn.Module):
    """MobileNet-style inverted residual block for 1D signals."""

    def __init__(self, in_ch, out_ch, expansion=3, stride=1):
        super().__init__()
        hidden = in_ch * expansion
        self.use_residual = (stride == 1 and in_ch == out_ch)

        self.expand = nn.Conv1d(in_ch, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)

        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(hidden)

        self.project = nn.Conv1d(hidden, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = F.relu6(self.bn1(self.expand(x)))
        out = F.relu6(self.bn2(self.depthwise(out)))
        out = self.bn3(self.project(out))
        if self.use_residual:
            out = out + x
        return out


class GumbelMaskMobileStyleCNN(nn.Module):
    """MobileNet-inspired 1D CNN with integrated Gumbel on/off frequency masking."""

    def __init__(
        self,
        num_classes=6,
        num_channels=3,
        freq_bins=65,
        dropout=0.4,
        gumbel_tau=2.0,
        tau_start=2.0,
        tau_end=2.0,
        channels=(24, 32, 40),
        expansion=3,
        fc_hidden=32,
    ):
        super().__init__()

        if len(channels) != 3:
            raise ValueError(f"Expected 3 channel values for L9, got {len(channels)}: {channels}")

        self.bin_logits = nn.Parameter(torch.zeros(freq_bins, 2))
        self.gumbel_tau = gumbel_tau
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.current_tau = gumbel_tau
        self.mask_l1 = None
        self.last_mask = None

        c1, c2, c3 = channels
        self.bn0 = nn.BatchNorm1d(num_channels)

        self.block1 = InvertedResidual1d(num_channels, c1, expansion=expansion, stride=1)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = InvertedResidual1d(c1, c2, expansion=expansion, stride=1)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = InvertedResidual1d(c2, c3, expansion=expansion, stride=1)
        self.pool3 = nn.MaxPool1d(2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c3, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        if self.training:
            probs = F.gumbel_softmax(self.bin_logits, tau=self.current_tau, hard=True)
        else:
            probs_soft = torch.softmax(self.bin_logits, dim=-1)
            probs = F.one_hot(probs_soft.argmax(dim=-1), num_classes=2).float()

        mask = probs[:, 1]
        self.mask_l1 = mask.mean()
        self.last_mask = mask.detach()
        x = x * mask.view(1, 1, -1)

        x = self.bn0(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def set_tau(self, epoch, max_epochs):
        """Keep same API as other masked models."""
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
    def __init__(self, num_classes=6, num_channels=3, freq_bins=65, dropout=0.4):
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

