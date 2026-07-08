#!/usr/bin/env python3
"""
NeuroPLC — Student KAN (Kolmogorov-Arnold Network)
=====================================================
Lightweight KAN with learnable B-spline activation functions.
Designed for knowledge distillation from Teacher CNN and eventual
compilation to IEC 61131-3 SCL via B-spline lookup tables.

Architecture:
    KAN([28, 16, 4], grid=8, spline_order=3)
      → Input:  28-D features (z-score normalized)
      → Layer 0: 28 → 16  (KANLinear with B-spline)
      → Layer 1: 16 → 4   (KANLinear with B-spline)
      → Output:  4-class logits

Parameters: 6,148 (grid_size=8, spline_order=3)
    B-spline: 4,928 + 704 = 5,632, Base: 448 + 64 = 512, Scales: 4
    KAN has more parameters than MLP but achieves PLC compatibility via LUT compilation

Key design choices (PLC-friendly):
    - grid_size=8:  small grid → fewer table entries
    - spline_order=3 (cubic):  smooth, well-behaved B-splines
    - x_range=[-3, 3]:  covers >99.7% of z-score normalized inputs
    - Residual SiLU base:  stable training even when splines are weak

Reference:
    Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv 2024
    Implementation adapted from pykan / efficient-kan

Usage:
    from models.student_kan import StudentKAN, KANLinear
    model = StudentKAN(layers_hidden=[28, 16, 4])
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# B-spline Basis Functions
# ============================================================================

def _bspline_basis(x: torch.Tensor, grid: torch.Tensor, k: int = 3
                   ) -> torch.Tensor:
    """
    Evaluate B-spline basis functions B_{i,k}(x) for all grid intervals.

    Args:
        x:    (...,) input values
        grid: (G,) knot vector (G = grid_size + 2*k + 1 typically)
        k:    spline order (3 = cubic)

    Returns:
        basis: (..., G + k - 1)  — B_{i,k}(x) for each basis index i
    """
    # Add trailing dims for broadcasting
    x = x.unsqueeze(-1)  # (..., 1)
    grid = grid.view(*([1] * (x.dim() - 1)), -1)  # (1, ..., G)

    # Base case: k=0 — B_{i,0}(x) = 1 if x in [grid[i], grid[i+1]), else 0
    # For numerical stability, we build up from k=0 using Cox-de Boor recursion

    # Efficient implementation via recursive formulation:
    # B_{i,k}(x) = (x-grid[i])/(grid[i+k]-grid[i]) * B_{i,k-1}(x)
    #            + (grid[i+k+1]-x)/(grid[i+k+1]-grid[i+1]) * B_{i+1,k-1}(x)

    G = grid.shape[-1]
    n_bases = G - 1

    # Initialize: B_{i,0}(x) is 1 on [grid[i], grid[i+1])
    # We represent this as left-closed, right-open intervals
    left = grid[..., :n_bases]      # (..., n_bases)
    right = grid[..., 1:]           # (..., n_bases)

    # For k=0: basis is 1 if grid[i] <= x < grid[i+1]
    # We use a smooth approximation for gradient flow
    bases = ((x >= left).float() * (x < right).float())  # (..., n_bases)

    # Recursion for k=1, 2, 3
    for order in range(1, k + 1):
        n = n_bases - order  # number of bases at this order

        # Left term: (x - grid[i]) / (grid[i+order] - grid[i])
        denom_left = grid[..., order:n+order] - grid[..., :n]
        term_left = torch.where(
            denom_left.abs() > 1e-10,
            (x - grid[..., :n]) / denom_left * bases[..., :n],
            torch.zeros_like(bases[..., :n]),
        )

        # Right term: (grid[i+order+1] - x) / (grid[i+order+1] - grid[i+1])
        denom_right = grid[..., order+1:n+order+1] - grid[..., 1:n+1]
        term_right = torch.where(
            denom_right.abs() > 1e-10,
            (grid[..., order+1:n+order+1] - x) / denom_right * bases[..., 1:n+1],
            torch.zeros_like(bases[..., 1:n+1]),
        )

        bases = term_left + term_right  # (..., n)

    return bases  # (..., G - k - 1)


# ============================================================================
# KANLinear Layer
# ============================================================================

class KANLinear(nn.Module):
    """
    A single Kolmogorov-Arnold layer.

    Each output neuron receives a learnable univariate function (B-spline + SiLU)
    applied to each input, summed:

        y_j = Σ_i [ w_b * SiLU(x_i) + w_s * Σ_c c_{j,i,c} * B_c(x_i) ]

    This replaces the standard Linear + Activation pattern.

    Args:
        in_features:   input dimension
        out_features:  output dimension
        grid_size:     number of B-spline grid intervals (default 8)
        spline_order:  B-spline order (default 3 = cubic)
        grid_eps:      grid extension margin (default 0.02)
        scale_base:    weight for SiLU base activation
        scale_spline:  weight for B-spline component
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_eps: float = 0.02,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_eps = grid_eps

        # Grid points (uniform by default, adaptive during training)
        # Range: [-1 - eps, 1 + eps] initially, expands if inputs exceed
        h = (grid_eps + 1.0) / grid_size
        grid = torch.linspace(-1.0 - grid_eps, 1.0 + grid_eps,
                              grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)  # (grid_size + 2k + 1,)

        # B-spline coefficients: (out_features, in_features, grid_size + spline_order)
        n_spline_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_spline_bases) * 0.1
        )

        # Base activation weight (SiLU residual): (out_features, in_features)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )

        # Scale factors
        self.scale_base = nn.Parameter(torch.tensor(scale_base))
        self.scale_spline = nn.Parameter(torch.tensor(scale_spline))

    def _extend_grid(self, x: torch.Tensor):
        """Extend grid if inputs fall outside current range (training only)."""
        if not self.training:
            return

        with torch.no_grad():
            x_min = x.min().item()
            x_max = x.max().item()
            grid_min = self.grid[self.spline_order].item()
            grid_max = self.grid[-(self.spline_order + 1)].item()

            if x_min < grid_min or x_max > grid_max:
                # Rebuild grid with extended range
                margin = max(abs(grid_min - x_min), abs(x_max - grid_max)) + self.grid_eps
                new_h = (max(x_max, -x_min) + margin) / (self.grid_size // 2)
                new_grid = torch.linspace(
                    min(x_min, -margin) - margin,
                    max(x_max, margin) + margin,
                    self.grid_size + 2 * self.spline_order + 1,
                    device=self.grid.device,
                    dtype=self.grid.dtype,
                )
                # Interpolate spline weights to new grid (simple: re-initialize)
                # In practice, for stable training, the grid should not need
                # frequent extension with z-score normalized inputs
                n_bases = self.grid_size + self.spline_order
                self.grid.copy_(new_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features)

        Returns:
            y: (..., out_features)
        """
        # Shape: (..., in_features)
        orig_shape = x.shape
        batch_dims = orig_shape[:-1]

        # ── Base activation (SiLU) ──
        base_out = F.silu(x)  # same shape as x

        # ── B-spline activation ──
        self._extend_grid(x)

        # Normalize x to [-1, 1] roughly (the grid is defined in [-1-eps, 1+eps])
        # For z-score normalized data, most values fall in [-3, 3]
        # We rescale by 1/3 to map into [-1, 1] where the grid lives
        x_scaled = x / 3.0  # [-3, 3] → [-1, 1]

        spline_basis = _bspline_basis(x_scaled, self.grid, self.spline_order)
        # spline_basis: (..., n_spline_bases)

        # Compute B-spline output: Σ_c c_{j,i,c} * B_c(x_i)
        # spline_weight: (out, in, n_bases)
        # spline_basis:  (..., in, n_bases)
        #
        # Reshape for matmul:
        #   spline_weight: (out, in, n_bases)
        #   spline_basis:  (..., 1, in, n_bases) → sum over n_bases
        spline_basis_expanded = spline_basis.unsqueeze(-2)  # (..., 1, n_bases)
        # Actually we need: for each (out, in), dot product with spline_basis over n_bases
        # spline_weight: (out, in, n_bases)
        # We want: y_j = Σ_i Σ_c w_{j,i,c} * B_c(x_i)
        # = Σ_i [ spline_basis_i · spline_weight[j,i,:] ]

        # More efficient:
        # Reshape to (..., in, 1, n_bases) * (1, out, in, n_bases) → sum over n_bases, then sum over in
        # Or: (..., in, n_bases) @ (out, in, n_bases)^T  doesn't work directly...

        # Simple approach: for each output j, ∀ input i:
        #   prod = spline_basis(..., in, n_bases) * spline_weight(out, in, n_bases) with broadcast
        #   sum over last dim, then sum over in

        # (..., 1, in, n_bases) * (out, in, 1, n_bases)  — no, wrong

        # Working:
        # spline_basis: (..., in, n_bases)
        # spline_weight: (out, in, n_bases)
        # We can do: (..., 1, in, n_bases) * (out, in, 1, n_bases)
        # That's not right either.
        #
        # Correct:
        # spline_basis: (..., in, n_bases) → (..., in, n_bases, 1)
        # spline_weight: (out, in, n_bases) → (1, in, n_bases, out)
        # Sum over n_bases: (..., in, out) then sum over in: (..., out)

        # Even simpler: for each (j,i), contract over n_bases:
        # spline_out[j,i] = Σ_c spline_weight[j,i,c] * spline_basis[..., i, c]
        # = einstein('... i c, j i c -> ... j i')
        spline_out = torch.einsum(
            '... i c, j i c -> ... j i',
            spline_basis, self.spline_weight
        )  # (..., out, in)

        # Sum over input dimension
        # base_out: (..., in) → weighted by base_weight: (..., out)
        base_weighted = torch.einsum(
            '... i, j i -> ... j', base_out, self.base_weight
        )  # (..., out)

        spline_sum = spline_out.sum(dim=-1)  # (..., out)

        # Combine
        y = (
            self.scale_base * base_weighted
            + self.scale_spline * spline_sum
        )

        return y

    def get_activation_function(self, input_idx: int, output_idx: int,
                                 x_range: tuple = (-3.0, 3.0),
                                 n_points: int = 200) -> tuple:
        """
        Evaluate the learned activation function φ_{j,i}(x) for visualization.

        Returns:
            (x_values, y_values): numpy arrays for plotting.
        """
        import numpy as np
        with torch.no_grad():
            xs = torch.linspace(x_range[0], x_range[1], n_points)
            xs_scaled = xs / 3.0

            # SiLU base
            base_y = F.silu(xs)
            # B-spline
            spline_basis = _bspline_basis(xs_scaled, self.grid, self.spline_order)
            spline_y = (spline_basis * self.spline_weight[output_idx, input_idx]).sum(-1)

            y = (
                self.scale_base * self.base_weight[output_idx, input_idx] * base_y
                + self.scale_spline * spline_y
            )
        return xs.numpy(), y.numpy()

    def get_adaptive_sample_points(self, n_points: int = 20,
                                    x_range: tuple = (-3.0, 3.0),
                                    n_estimate: int = 100) -> tuple:
        """
        Adaptive non-uniform sampling for B-spline lookup table.

        Uses curvature-aware discretization: more points where |φ''| is large.

        Returns:
            (x_samples, y_samples): numpy arrays of shape (n_points,)
        """
        import numpy as np
        with torch.no_grad():
            # High-res evaluation for curvature estimation
            xs = torch.linspace(x_range[0], x_range[1], n_estimate, dtype=torch.float64)
            xs_scaled = xs / 3.0

            base_y = F.silu(xs)
            spline_basis = _bspline_basis(xs_scaled, self.grid, self.spline_order)
            spline_y = (spline_basis * self.spline_weight.mean(0).mean(0)).sum(-1)

            y = (
                self.scale_base.float() * self.base_weight.mean() * base_y
                + self.scale_spline.float() * spline_y
            )

            # Compute curvature κ = |y''| / (1 + y'²)^(3/2)
            y_np = y.numpy()
            xs_np = xs.numpy()
            dy = np.gradient(y_np, xs_np)
            d2y = np.gradient(dy, xs_np)
            curvature = np.abs(d2y) / (1.0 + dy ** 2) ** 1.5 + 1e-10

            # Cumulative curvature
            cum_curve = np.cumsum(curvature)
            cum_curve /= cum_curve[-1]  # [0, 1]

            # Sample uniformly in cumulative curvature space
            target_cdf = np.linspace(0, 1, n_points)
            x_samples = np.interp(target_cdf, cum_curve, xs_np)

            # Evaluate at sampled points
            xs_sampled = torch.from_numpy(x_samples).float()
            xs_s_scaled = xs_sampled / 3.0
            base_y_s = F.silu(xs_sampled)
            spline_basis_s = _bspline_basis(xs_s_scaled, self.grid, self.spline_order)
            spline_y_s = (spline_basis_s * self.spline_weight.mean(0).mean(0)).sum(-1)
            y_samples = (
                self.scale_base * self.base_weight.mean() * base_y_s
                + self.scale_spline * spline_y_s
            ).numpy()

        return x_samples, y_samples

    def get_uniform_sample_points(self, n_points: int = 20,
                                   x_range: tuple = (-3.0, 3.0)) -> tuple:
        """Uniform sampling (baseline, not curvature-aware)."""
        import numpy as np
        with torch.no_grad():
            xs = torch.linspace(x_range[0], x_range[1], n_points)
            xs_scaled = xs / 3.0

            base_y = F.silu(xs)
            spline_basis = _bspline_basis(xs_scaled, self.grid, self.spline_order)
            spline_y_mean = (spline_basis * self.spline_weight.mean(0).mean(0)).sum(-1)

            y = (
                self.scale_base * self.base_weight.mean() * base_y
                + self.scale_spline * spline_y_mean
            )
        return xs.numpy(), y.numpy()


# ============================================================================
# StudentKAN
# ============================================================================

class StudentKAN(nn.Module):
    """
    Shallow KAN student for knowledge distillation.

    Args:
        layers_hidden: list of layer widths [input, hidden, ..., output]
        grid_size:     B-spline grid intervals per edge
        spline_order:  B-spline polynomial order
        grid_eps:      grid adaptivity margin
        scale_base:    SiLU residual weight
        scale_spline:  B-spline component weight
    """

    def __init__(
        self,
        layers_hidden: list = None,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_eps: float = 0.02,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ):
        super().__init__()
        if layers_hidden is None:
            layers_hidden = [28, 16, 4]

        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_classes = layers_hidden[-1]

        self.kan_layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.kan_layers.append(
                KANLinear(
                    in_features=layers_hidden[i],
                    out_features=layers_hidden[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_eps=grid_eps,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 28) — z-score normalized features
        Returns:
            logits: (batch_size, num_classes)
        """
        for layer in self.kan_layers:
            x = layer(x)
        return x

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Multi-line summary."""
        p = self.parameter_count
        lines = [f"StudentKAN — {p:,} parameters"]
        lines.append(f"  Architecture: {self.layers_hidden}")
        lines.append(f"  Grid: {self.grid_size}, Order: {self.spline_order}")
        lines.append(f"  Activations per edge: {self.grid_size + self.spline_order}")
        return "\n".join(lines)

    def get_all_activation_functions(self, x_range=(-3.0, 3.0), n_points=200):
        """
        Return all learned activation functions for visualization.

        Returns:
            dict: {layer_idx: {output_idx: {input_idx: (x, y)}}}
        """
        act_funcs = {}
        for l_idx, layer in enumerate(self.kan_layers):
            act_funcs[l_idx] = {}
            for j in range(layer.out_features):
                act_funcs[l_idx][j] = {}
                for i in range(layer.in_features):
                    x, y = layer.get_activation_function(i, j, x_range, n_points)
                    act_funcs[l_idx][j][i] = (x, y)
        return act_funcs
