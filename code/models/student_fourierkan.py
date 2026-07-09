#!/usr/bin/env python3
"""
NeuroPLC — Student FourierKAN (Fourier-Kolmogorov-Arnold Network)
===================================================================
SVNN-compliant KAN variant using Fourier basis functions.
Each edge phi(x) = sum_{k=1}^K [c_k * sin(k*w*x) + d_k * cos(k*w*x)]

M2 formula (Proposition 9c):
    M2 = w^2 * sum_{k=1}^K k^2 * (|c_k| + |d_k|)

Computation: O(K) per edge, K typically 8.
All operations are C^2 and M2 is computable from coefficients alone → SVNN.

Architecture:
    FourierKAN([28, 16, 4], n_harmonics=8, omega=1.0)

Parameters:
    Layer 0: 28*16*(2*8+1) = 28*16*17 = 7,616 (base+bias + Fourier coeffs)
    Layer 1: 16*4*(2*8+1)  = 16*4*17  = 1,088
    Total: ~8,704 (slightly more than B-spline KAN's 6,148)

Usage:
    from models.student_fourierkan import StudentFourierKAN
    model = StudentFourierKAN([28, 16, 4], n_harmonics=8)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierKANLinear(nn.Module):
    """Single FourierKAN layer: y_j = sum_i phi_{j,i}(x_i) + b_j

    where phi_{j,i}(x) = sum_{k=1}^K [c_{j,i,k} * sin(k*w*x) + d_{j,i,k} * cos(k*w*x)]
          + c_{j,i,0} * x  (linear base path)
    """

    def __init__(self, in_features: int, out_features: int,
                 n_harmonics: int = 8, omega: float = 1.0,
                 domain_lo: float = -3.0, domain_hi: float = 3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_harmonics = n_harmonics
        self.omega = omega
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi

        # Fourier coefficients: sin terms + cos terms per harmonic
        # Shape: (out, in, 2*n_harmonics)
        self.fourier_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, 2 * n_harmonics) * 0.1
        )

        # Linear base path (like SiLU base in B-spline KAN)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.base_activation = nn.SiLU()

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Precompute harmonic indices (1..K)
        k = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        self.register_buffer('k', k)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.fourier_coeffs, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        batch, in_d = x.shape

        # Clamp to domain for numerical stability
        x = torch.clamp(x, self.domain_lo - 1.0, self.domain_hi + 1.0)

        # Base path: SiLU(x) * base_weight
        base_out = F.linear(self.base_activation(x), self.base_weight)

        # Fourier path
        # Compute sin/cos for each harmonic
        # x_expanded: (batch, in, 1)
        x_exp = x.unsqueeze(-1)  # (batch, in, 1)

        # k: (K,) → angles: (batch, in, K)
        angles = self.omega * self.k.view(1, 1, -1) * x_exp  # (batch, in, K)

        sins = torch.sin(angles)  # (batch, in, K)
        coss = torch.cos(angles)  # (batch, in, K)

        # Split coefficients: (out, in, K) each
        coeffs_sin = self.fourier_coeffs[:, :, :self.n_harmonics]  # (out, in, K)
        coeffs_cos = self.fourier_coeffs[:, :, self.n_harmonics:]  # (out, in, K)

        # Per-edge sum: for each (out, in), sum over K
        # sin_part: (batch, in, K) * (out, in, K) → sum over K, in → (batch, out)
        sin_part = torch.einsum('bik,oik->bo', sins, coeffs_sin)
        cos_part = torch.einsum('bik,oik->bo', coss, coeffs_cos)

        out = base_out + sin_part + cos_part + self.bias
        return out

    def compute_m2_bounds(self) -> torch.Tensor:
        """Compute M2 per edge: M2 = omega^2 * sum k^2*(|c_k|+|d_k|).
        Returns: (out_features, in_features) tensor of M2 values.
        """
        k = self.k  # (K,)
        coeffs_sin = self.fourier_coeffs[:, :, :self.n_harmonics]  # (out, in, K)
        coeffs_cos = self.fourier_coeffs[:, :, self.n_harmonics:]  # (out, in, K)

        m2 = self.omega**2 * torch.sum(
            k.view(1, 1, -1)**2 * (coeffs_sin.abs() + coeffs_cos.abs()),
            dim=-1
        )  # (out, in)
        return m2


class StudentFourierKAN(nn.Module):
    """Fourier KAN classifier for CWRU bearing fault diagnosis."""

    def __init__(self, layers_hidden, n_harmonics=8, omega=1.0,
                 num_classes=4, dropout=0.0):
        super().__init__()
        self.layers_hidden = layers_hidden
        self.n_harmonics = n_harmonics
        self.omega = omega

        self.kan_layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.kan_layers.append(
                FourierKANLinear(
                    layers_hidden[i], layers_hidden[i+1],
                    n_harmonics=n_harmonics, omega=omega,
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
    model = StudentFourierKAN([28, 16, 4], n_harmonics=8)
    x = torch.randn(4, 28)
    y = model(x)
    m2 = model.compute_all_m2()
    print(f"FourierKAN [28,16,4]: params={model.n_params}, edges={model.n_edges}")
    print(f"Output shape: {y.shape}")
    print(f"M2 bounds: L0 mean={m2[0].mean():.4f}, L1 mean={m2[1].mean():.4f}")
