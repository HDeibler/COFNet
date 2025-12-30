"""
Classification Head for COFNet.

Takes box features and predicts class probabilities.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Simple MLP classification head.

    Takes features at each box location and predicts class logits.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes

        layers = []
        in_dim = feature_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, box_features: torch.Tensor) -> torch.Tensor:
        """
        Classify boxes.

        Args:
            box_features: [B, N, D] features at each box location

        Returns:
            [B, N, num_classes] class logits
        """
        return self.mlp(box_features)
