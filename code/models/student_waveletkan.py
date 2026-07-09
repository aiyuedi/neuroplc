#!/usr/bin/env python3
"""
NeuroPLC — Student WaveletKAN (Wavelet-Kolmogorov-Arnold Network)
===================================================================
SVNN-compliant KAN variant using Mexican hat wavelet basis functions.
Each edge phi(x) = sum_{j=1}^J c_j * psi((x - b_j) / a_j)
where psi(t) = C * (1 - t^2) * exp(-t^2/2)  (Mexican hat / Ricker wavelet)

M2 formula (Proposition 9d):
    M2 = max_j (|c_j| / a_j^2) * sup_t |psi''(t)|
    sup_t |psi''(t)| ≈ 2.602 (computed numerically)

All psi are C^infinity → Condition 2 satisfied.
Computable from wavelet parameters alone → SVNN.

Architecture:
    WaveletKAN([28, 16, 4], n_scales=4)

Parameters:
    Layer 0: 28*16*(4+1+1) = 28*16*6 ≈ 2,688
    Layer 1: 16*4*(4+1+1)  = 16*4*6  ≈ 384
    Total: ~3,072 (very compact, fewer than B-spline KAN's 6,148)

Usage:
    from models.student_waveletkan import StudentWaveletKAN
    model = StudentWaveletKAN([28, 16, 4], n_scales=4)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mexican hat wavelet constant: sup_t |psi''(t)|
MEXICAN_HAT_M2_SUP = 2.6020
MEXICAN_HAT_NORM = 2.0 / (math.sqrt(3.0) * math.pi**0.25)  # ~0.867


def mexican_hat(t: torch.Tensor) -> torch.Tensor:
    """Mexican hat (Ricker) wavelet: psi(t) = C * (1 - t^2) * exp(-t^2/2)."""
    return MEXICAN_HAT_NORM * (1.0 - t**2) * torch.exp(-t**2 / 2.0)


class WaveletKANLinear(nn.Module):
    """Single WaveletKAN layer: y_j = sum_i phi_{j,i}(x_i) + b_j

    where phi_{j,i}(x) = sum_{s=1}^S c_{j,i,s} * psi((x - mu_s) / a_s)
          + w_{j,i} * x  (linear base)
    """

    def __init__(self, in_features: int, out_features: int,
                 n_scales: int = 4,
                 domain_lo: float = -3.0, domain_hi: float = 3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_scales = n_scales
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi

        # Wavelet coefficients per edge: (out, in, n_scales)
        self.wavelet_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, n_scales) * 0.05
        )

        # Scales (shared across all edges for simplicity)
        # a_s = a_min * (a_ratio)^s
        a_min = 0.3
        a_max = 2.5
        scales = torch.logspace(math.log10(a_min), math.log10(a_max), n_scales)
        self.register_buffer('scales', scales)  # (n_scales,)

        # Shifts mu_s = uniform over domain
        shifts = torch.linspace(domain_lo + 0.5, domain_hi - 0.5, n_scales)
        self.register_buffer('shifts', shifts)  # (n_scales,)

        # Linear base path
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.wavelet_coeffs, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        batch, in_d = x.shape

        x = torch.clamp(x, self.domain_lo - 1.0, self.domain_hi + 1.0)

        # Base path
        base_out = F.linear(x, self.base_weight)  # (batch, out)

        # Wavelet path
        # For each scale s: t_{i,s} = (x_i - mu_s) / a_s
        # psi(t) evaluated, then weighted by coeffs
        # x_exp: (batch, in, 1), scales: (1, 1, S), shifts: (1, 1, S)
        x_exp = x.unsqueeze(-1)  # (batch, in, 1)
        a = self.scales.view(1, 1, -1)   # (1, 1, S)
        b = self.shifts.view(1, 1, -1)   # (1, 1, S)

        t = (x_exp - b) / a  # (batch, in, S)
        psi_vals = mexican_hat(t)  # (batch, in, S)

        # Weighted sum: (batch, in, S) * (out, in, S) → sum over in, S → (batch, out)
        wavelet_out = torch.einsum('bis,ois->bo', psi_vals, self.wavelet_coeffs)

        return base_out + wavelet_out + self.bias

    def compute_m2_bounds(self) -> torch.Tensor:
        """Compute M2 per edge.

        M2 = max_s (|c_s| / a_s^2) * sup|psi''| + |w_base|
        where sup|psi''| = MEXICAN_HAT_M2_SUP

        Returns: (out_features, in_features) tensor of M2 values.
        """
        # Per-scale contribution: |c_s| / a_s^2
        # wavelet_coeffs: (out, in, S), scales: (S,)
        scale_contrib = self.wavelet_coeffs.abs() / (self.scales.view(1, 1, -1)**2)  # (out, in, S)
        max_contrib = scale_contrib.max(dim=-1).values  # (out, in)
        m2_wavelet = max_contrib * MEXICAN_HAT_M2_SUP
        m2_base = self.base_weight.abs()  # linear base: M2 = 0, but |weight| for Lipschitz

        return m2_wavelet + m2_base


class StudentWaveletKAN(nn.Module):
    """WaveletKAN classifier for CWRU bearing fault diagnosis."""

    def __init__(self, layers_hidden, n_scales=4, num_classes=4, dropout=0.0):
        super().__init__()
        self.layers_hidden = layers_hidden
        self.n_scales = n_scales

        self.kan_layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.kan_layers.append(
                WaveletKANLinear(
                    layers_hidden[i], layers_hidden[i+1],
                    n_scales=n_scales,
                    domain_lo=-3.0, domain_hi=3.0
                )
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        for layer in self.kan_layers:
            x = layer(x)
            x = self.dropout(x)
        return x

    def compute_all_m2(self):
        """Return dict of {layer_idx: M2_tensor} for all layers."""
        m2_dict = {}
        for i, layer in enumerate(self.kan_layers):
            m2_dict[i] = layer.compute_m2_bounds()
        return m2_dict

    @property
    def n_edges(self):
        return sum(l.in_features * l.out_features for l in self.kan_layers)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = StudentWaveletKAN([28, 16, 4], n_scales=4)
    x = torch.randn(4, 28)
    y = model(x)
    m2 = model.compute_all_m2()
    print(f"WaveletKAN [28,16,4]: params={model.n_params}, edges={model.n_edges}")
    print(f"Output shape: {y.shape}")
    print(f"M2 bounds: L0 mean={m2[0].mean():.4f}, L1 mean={m2[1].mean():.4f}")
    print(f"MEXICAN_HAT_M2_SUP = {MEXICAN_HAT_M2_SUP:.4f}")
