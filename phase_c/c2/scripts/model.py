"""
Phase C2: Small 3D ResNet model for the controlled preprocessing experiment.

Architecture: Small 3D CNN with residual connections.
Input: (B, 1, 32, 256, 256) — single-channel 3D MRI volume
Output: (B, 12) — logits for 12 binary targets

Design constraints:
    - Must fit in 8 GB VRAM with batch_size=2 and mixed precision
    - ~1-2M parameters — enough capacity for fair comparison but small enough
      that preprocessing quality (not model capacity) is the bottleneck
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """Simple 3D residual block with two conv layers and optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out, inplace=True)


class SmallResNet3D(nn.Module):
    """Small 3D ResNet for knee MRI multi-label classification.

    Architecture summary:
        stem: Conv3d(1→16) + BN + ReLU + MaxPool
        layer1: 2x ResBlock3D(16→32, stride=2 for first)
        layer2: 2x ResBlock3D(32→64, stride=2 for first)
        layer3: 2x ResBlock3D(64→128, stride=2 for first)
        head: AdaptiveAvgPool3d → Dropout → Linear(128→12)

    With input (B, 1, 32, 256, 256):
        After stem+pool: (B, 16, 16, 128, 128)
        After layer1:    (B, 32, 8, 64, 64)
        After layer2:    (B, 64, 4, 32, 32)
        After layer3:    (B, 128, 2, 16, 16)
        After GAP:       (B, 128)
    """

    def __init__(self, in_channels: int = 1, num_targets: int = 12, dropout: float = 0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        # Residual layers
        self.layer1 = nn.Sequential(
            ResBlock3D(16, 32, stride=2),
            ResBlock3D(32, 32, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(32, 64, stride=2),
            ResBlock3D(64, 64, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResBlock3D(64, 128, stride=2),
            ResBlock3D(128, 128, stride=1),
        )

        # Classification head
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(128, num_targets)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # Raw logits — apply sigmoid externally


def count_parameters(model: nn.Module) -> int:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = SmallResNet3D()
    total, trainable = count_parameters(model)
    print(f"SmallResNet3D: {total:,} total params, {trainable:,} trainable")

    # Verify forward pass shape
    x = torch.randn(2, 1, 32, 256, 256)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # Estimate VRAM with mixed precision
    model = model.cuda()
    x = x.cuda()
    with torch.cuda.amp.autocast():
        out = model(x)
    print(f"Peak VRAM after forward: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
