#!/usr/bin/env python3
"""
NeuroPLC — Student MLP Baseline
=================================
Simple multi-layer perceptron for ablation comparison (E2: KAN vs MLP).

Architecture:
    Input (28-D features) → FC(32) → ReLU → Dropout(0.1)
                          → FC(16) → ReLU → Dropout(0.1)
                          → FC(4)  → softmax

Parameters: ~1,636 (28×32 + 32 + 32×16 + 16 + 16×4 + 4)

This is intentionally the SAME architecture as the old v1 Student,
preserved for E2 ablation to show KAN's parameter efficiency advantage:
    KAN ~300 params ≈ MLP ~1600 params in accuracy

Usage:
    from models.student_mlp import StudentMLP
    model = StudentMLP(input_dim=28, hidden_dims=[32, 16], num_classes=4)
"""

import torch
import torch.nn as nn


class StudentMLP(nn.Module):
    """
    Shallow MLP for PLC-deployable inference.

    Args:
        input_dim:    feature dimension (default 28)
        hidden_dims:  list of hidden layer widths
        num_classes:  output classes (default 4)
        dropout:      dropout rate
        activation:   activation function name
    """

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dims: list = None,
        num_classes: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Output layer (no activation — logits for CrossEntropyLoss)
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform init for ReLU layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) — z-score normalized features
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.net(x)

    @property
    def parameter_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def weight_matrices(self) -> list[dict]:
        """
        Extract all Linear layer weights and biases for compiler export.

        Returns:
            List of {"weight": Tensor, "bias": Tensor} in order.
        """
        matrices = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                matrices.append({
                    "weight": m.weight.data.clone(),
                    "bias": m.bias.data.clone(),
                })
        return matrices
