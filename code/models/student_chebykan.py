#!/usr/bin/env python3
"""
NeuroPLC — Student ChebyKAN (Chebyshev-Kolmogorov-Arnold Network)
===================================================================
Lightweight KAN using Chebyshev polynomial basis functions instead of B-splines.

Architecture:
    ChebyKAN([28, 16, 4], degree=5)
      → Input:  28-D features (z-score normalized)
      → Layer 0: 28 → 16  (ChebyKANLinear with Chebyshev polynomials)
      → Layer 1: 16 → 4   (ChebyKANLinear with Chebyshev polynomials)
      → Output:  4-class logits

Key differences from B-spline KAN:
    - Global basis (Chebyshev) vs local basis (B-spline)
    - M_2 bound via Markov's inequality vs segment-aware bounds
    - Polynomial NRA verification vs segment enumeration

Parameters: ~6,400 (comparable to B-spline KAN)
    Coefficients: degree+1 per (i,j) pair

Reference:
    Bozorgasl & Chen, "Wav-KAN: Wavelet Kolmogorov-Arnold Networks", arXiv 2024
    (ChebyKAN is a variant using Chebyshev instead of wavelets)

Usage:
    from models.student_chebykan import StudentChebyKAN, ChebyKANLinear
    model = StudentChebyKAN(layers_hidden=[28, 16, 4], degree=5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Chebyshev Polynomial Basis Functions
# ============================================================================

def chebyshev_polynomials(x: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Evaluate Chebyshev polynomials T_0, T_1, ..., T_degree at x.

    Uses the recurrence relation:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)

    Args:
        x: (...,) input values in [-1, 1] (via tanh preprocessing)
        degree: maximum polynomial degree N

    Returns:
        polys: (..., degree+1) — [T_0(x), T_1(x), ..., T_degree(x)]
    """
    # Initialize storage for all polynomials using clone-safe approach
    # polys[..., 0] = 1.0, polys[..., 1] = x
    polys_list = [torch.ones_like(x)]  # T_0(x) = 1
    if degree >= 1:
        polys_list.append(x.clone())   # T_1(x) = x

    # Recurrence: T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)
    for n in range(1, degree):
        T_next = 2.0 * x * polys_list[n] - polys_list[n-1]
        polys_list.append(T_next)

    polys = torch.stack(polys_list, dim=-1)  # (..., degree+1)
    return polys


# ============================================================================
# ChebyKAN Layer
# ============================================================================

class ChebyKANLinear(nn.Module):
    """
    Chebyshev-KAN linear layer: y_j = sum_{i,n} c_{j,i,n} * T_n(tanh(x_i))

    Key differences from B-spline KAN:
        - tanh(x) preprocesses input to [-1,1] (Chebyshev domain)
        - Chebyshev polynomials T_n(u) evaluated via recurrence
        - Global polynomial basis (all x contribute to all T_n)

    Args:
        in_features: input dimension
        out_features: output dimension
        degree: maximum Chebyshev polynomial degree N (default: 5)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Learnable Chebyshev coefficients: c_{j,i,n}
        # Shape: (out_features, in_features, degree+1)
        self.coeffs = nn.Parameter(
            torch.Tensor(out_features, in_features, degree + 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize coefficients using Xavier uniform.
        Scale by 1/sqrt(degree+1) for stable gradient flow.
        """
        nn.init.xavier_uniform_(self.coeffs)
        # Additional scaling for polynomial basis
        with torch.no_grad():
            self.coeffs *= (1.0 / math.sqrt(self.degree + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = sum_{i,n} c_{j,i,n} * T_n(tanh(x_i))

        Args:
            x: (batch, in_features) input

        Returns:
            y: (batch, out_features) output
        """
        batch_size = x.shape[0]

        # Step 1: Bound mapping u = tanh(x) to [-1, 1]
        u = torch.tanh(x)  # (batch, in_features)

        # Step 2: Evaluate Chebyshev polynomials T_n(u) for all u_i
        # polys: (batch, in_features, degree+1)
        polys = chebyshev_polynomials(u, self.degree)

        # Step 3: Linear combination with coefficients
        # coeffs: (out_features, in_features, degree+1)
        # polys:  (batch, in_features, degree+1)
        #
        # We want: y[b,j] = sum_{i,n} coeffs[j,i,n] * polys[b,i,n]
        #                 = sum_i (coeffs[j,i,:] @ polys[b,i,:])

        # Efficient implementation via einsum
        y = torch.einsum('oin,bin->bo', self.coeffs, polys)

        return y

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'degree={self.degree}')


# ============================================================================
# ChebyKAN Student Network
# ============================================================================

class StudentChebyKAN(nn.Module):
    """
    Two-layer Chebyshev-KAN for bearing fault classification.

    Architecture:
        Input (28) → ChebyKANLinear → ChebyKANLinear → Logits (4)

    Designed to match B-spline KAN's parameter count (~6K parameters)
    and provide an alternative SVNN-compliant architecture for comparison.

    Args:
        layers_hidden: list of layer sizes (default: [28, 16, 4])
        degree: Chebyshev polynomial degree (default: 5)
    """
    def __init__(
        self,
        layers_hidden: list = None,
        degree: int = 5,
    ):
        super().__init__()

        if layers_hidden is None:
            layers_hidden = [28, 16, 4]

        self.layers_hidden = layers_hidden
        self.degree = degree

        # Build ChebyKAN layers
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                ChebyKANLinear(in_dim, out_dim, degree=degree)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all ChebyKAN layers.

        Args:
            x: (batch, 28) input features (z-score normalized)

        Returns:
            logits: (batch, 4) class logits
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_m2_bounds(self) -> dict:
        """
        Compute analytical M_2 bounds via Markov's inequality.

        Returns dict with:
            - m2_analytical: N^2(N^2+5)/3 * sum(|c_n|) per layer
            - lipschitz_analytical: N^2 * sum(|c_n|) per layer

        Reference: Proposition 2 (ChebyKAN Is SVNN), Eq. (cheby_final_m2)
        """
        results = {}
        for layer_idx, layer in enumerate(self.layers):
            N = layer.degree

            # Compute sum of absolute coefficients per output neuron
            coeffs_abs_sum = layer.coeffs.abs().sum(dim=(1, 2))  # (out_features,)

            # Markov's inequality for M_2: N^2(N^2-1)/3 for polynomial
            # Plus composition with tanh: final bound is N^2(N^2+5)/3
            markov_factor = N**2 * (N**2 + 5) / 3.0
            m2_bound = markov_factor * coeffs_abs_sum

            # Lipschitz constant: N^2 * sum(|c_n|)
            lipschitz_bound = N**2 * coeffs_abs_sum

            results[f'layer_{layer_idx}'] = {
                'm2_analytical': m2_bound.mean().item(),
                'm2_max': m2_bound.max().item(),
                'lipschitz_analytical': lipschitz_bound.mean().item(),
                'lipschitz_max': lipschitz_bound.max().item(),
            }

        return results


# ============================================================================
# Factory function
# ============================================================================

def create_student_chebykan(
    in_features: int = 28,
    hidden_dim: int = 16,
    num_classes: int = 4,
    degree: int = 5,
) -> StudentChebyKAN:
    """
    Factory function for creating ChebyKAN student model.

    Args:
        in_features: input dimension (default: 28 for CWRU features)
        hidden_dim: hidden layer size (default: 16)
        num_classes: output classes (default: 4 for CWRU fault types)
        degree: Chebyshev polynomial degree (default: 5)

    Returns:
        StudentChebyKAN model
    """
    return StudentChebyKAN(
        layers_hidden=[in_features, hidden_dim, num_classes],
        degree=degree,
    )


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    # Test ChebyKAN layer
    print("Testing ChebyKANLinear...")
    layer = ChebyKANLinear(28, 16, degree=5)
    x = torch.randn(32, 28)
    y = layer(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in layer.parameters())}")

    # Test full ChebyKAN model
    print("\nTesting StudentChebyKAN...")
    model = StudentChebyKAN(layers_hidden=[28, 16, 4], degree=5)
    logits = model(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Total parameters: {model.count_parameters()}")

    # Test M_2 bounds
    print("\nComputing analytical M_2 bounds...")
    bounds = model.get_m2_bounds()
    for layer_name, layer_bounds in bounds.items():
        print(f"{layer_name}:")
        print(f"  M_2 (mean): {layer_bounds['m2_analytical']:.4f}")
        print(f"  M_2 (max):  {layer_bounds['m2_max']:.4f}")
        print(f"  Lipschitz (mean): {layer_bounds['lipschitz_analytical']:.4f}")
