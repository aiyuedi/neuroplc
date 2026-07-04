#!/usr/bin/env python3
"""
NeuroPLC — Teacher 1D-CNN with Self-Attention
===============================================
Large teacher model for CWRU bearing fault diagnosis.
Trained on raw waveform, then used to distill knowledge into Student KAN.

Architecture:
    Input (1, 1024)
      → Conv1D(1→16, k=15) → BN → ReLU → MaxPool(4)
      → Conv1D(16→32, k=9) → BN → ReLU → MaxPool(2)
      → Conv1D(32→64, k=5) → BN → ReLU → MaxPool(2)
      → MultiHeadSelfAttention(4 heads, d=64)
      → AdaptiveAvgPool → Flatten
      → FC(128) → ReLU → Dropout(0.3)
      → FC(64)  → ReLU → Dropout(0.3)
      → FC(4)   → softmax

Parameters: ~50K
Expected accuracy: 99%+ on CWRU single-load (standard benchmark)

Reference:
    WDCNN (Zhang et al., Sensors 2017) — wide first-layer kernel concept
    ResNet-1D (Zhang et al., IEEE Access 2020)

Usage:
    from models.teacher_cnn import TeacherCNN
    model = TeacherCNN(num_classes=4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention1D(nn.Module):
    """Multi-head self-attention for 1-D feature maps."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) — batch, channels, length
        Returns:
            (B, C, L) — same shape
        """
        B, C, L = x.shape
        # (B, L, C) for attention
        x_t = x.transpose(1, 2)  # (B, L, C)

        qkv = self.qkv(x_t)  # (B, L, 3C)
        qkv = qkv.reshape(B, L, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, L, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)

        return out.transpose(1, 2)  # back to (B, C, L)


class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class TeacherCNN(nn.Module):
    """
    1D-CNN + Self-Attention teacher for bearing fault diagnosis.

    Args:
        input_channels:  number of input channels (default 1)
        num_classes:     number of fault classes (default 4)
        conv_channels:   conv layer output channels
        kernel_sizes:    conv kernel sizes
        pool_sizes:      max-pool sizes
        attn_heads:      self-attention heads
        attn_dim:        attention dimension
        fc_dims:         FC hidden layer sizes
        dropout:         dropout rate
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 4,
        conv_channels: list = None,
        kernel_sizes: list = None,
        pool_sizes: list = None,
        attn_heads: int = 4,
        attn_dim: int = 64,
        fc_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 32, 64]
        if kernel_sizes is None:
            kernel_sizes = [15, 9, 5]
        if pool_sizes is None:
            pool_sizes = [4, 2, 2]
        if fc_dims is None:
            fc_dims = [128, 64]

        # ── Convolutional backbone ──
        self.conv_blocks = nn.ModuleList()
        in_ch = input_channels
        for out_ch, k, p in zip(conv_channels, kernel_sizes, pool_sizes):
            self.conv_blocks.append(ConvBlock(in_ch, out_ch, k, p))
            in_ch = out_ch

        # ── Self-attention ──
        self.attention = SelfAttention1D(attn_dim, attn_heads, dropout)

        # ── Classifier head ──
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        fc_layers = []
        fc_in = conv_channels[-1]  # 64
        for fc_out in fc_dims:
            fc_layers.extend([
                nn.Linear(fc_in, fc_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            fc_in = fc_out
        fc_layers.append(nn.Linear(fc_in, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        self._init_weights()
        self._stats = {
            "input_channels": input_channels,
            "conv_channels": conv_channels,
            "num_classes": num_classes,
        }

    def _init_weights(self):
        """Kaiming init for conv and FC layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: (B, 1, 1024) — raw vibration waveform
            return_features: if True, also return the feature vector before FC
        Returns:
            logits:     (B, num_classes)
            features:   (B, 64) if return_features=True
        """
        # Convolutional backbone
        for conv_block in self.conv_blocks:
            x = conv_block(x)  # (B, C, L')

        # Self-attention
        x = self.attention(x)  # (B, 64, L')

        # Pooling and flatten
        x = self.avg_pool(x).squeeze(-1)  # (B, 64)

        features = x  # (B, 64) — for KD feature alignment

        # Classifier
        logits = self.classifier(x)  # (B, num_classes)

        if return_features:
            return logits, features
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before FC (for KD feature alignment)."""
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.attention(x)
        x = self.avg_pool(x).squeeze(-1)
        return x  # (B, 64)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a multi-line summary string."""
        p = self.parameter_count
        return (
            f"TeacherCNN — {p:,} parameters\n"
            f"  Conv: {self._stats['conv_channels']}\n"
            f"  Attention: 4 heads, d=64\n"
            f"  FC: [128, 64, {self._stats['num_classes']}]"
        )
