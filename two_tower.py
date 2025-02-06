import torch
import torch.nn as nn


class TowerOne(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class TowerTwo(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)
