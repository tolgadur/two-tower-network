import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TowerOne(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        rnn_layers: int = 1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0,
        )

        self.sequential = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        if lengths is None:
            _, (h_n, _) = self.rnn(x)
            x = h_n[-1]  # Take the last layer's hidden state
            return self.sequential(x)

        # Pack sequence
        x_packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through RNN and get final hidden state
        _, (h_n, _) = self.rnn(x_packed)
        x = h_n[-1]  # Take the last layer's hidden state

        return self.sequential(x)


class TowerTwo(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        rnn_layers: int = 1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0,
        )

        self.sequential = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        if lengths is None:
            _, (h_n, _) = self.rnn(x)
            x = h_n[-1]  # Take the last layer's hidden state
            return self.sequential(x)

        # Pack sequence
        x_packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through RNN and get final hidden state
        _, (h_n, _) = self.rnn(x_packed)
        x = h_n[-1]  # Take the last layer's hidden state

        return self.sequential(x)
