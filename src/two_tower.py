import torch


class TowerOne(torch.nn.Module):
    """Neural network tower for encoding queries.

    This tower processes pre-computed word embeddings through an RNN and FC layers
    to produce fixed-size encodings.

    Example:
        tower = TowerOne(hidden_dimension=258)
    """

    def __init__(
        self,
        hidden_dimension=258,
        dropout=0.1,
        embedding_dim=300,  # Input dimension from word2vec embeddings
    ):
        super().__init__()

        self.rnn = torch.nn.RNN(embedding_dim, hidden_dimension, batch_first=True)
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.LayerNorm(hidden_dimension),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension // 2),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Process pre-computed embeddings into fixed-size vectors.

        Args:
            x: Pre-computed embeddings tensor of shape
                (batch_size, seq_len, embedding_dim)
            lengths: Length of each sequence in the batch

        Returns:
            Encoded vectors of shape (batch_size, hidden_dimension // 2)
        """
        # RNN with packed sequence
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed_x)

        # FC layer
        hidden = h_n[-1]
        out = self.sequential(hidden)

        return out


class TowerTwo(torch.nn.Module):
    """Neural network tower for encoding documents.

    This tower processes pre-computed word embeddings through an RNN and FC layers
    to produce fixed-size encodings.

    Example:
        tower = TowerTwo(hidden_dimension=258)
    """

    def __init__(
        self,
        hidden_dimension=258,
        dropout=0.1,
        embedding_dim=300,  # Input dimension from word2vec embeddings
    ):
        super().__init__()

        self.rnn = torch.nn.RNN(embedding_dim, hidden_dimension, batch_first=True)
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.LayerNorm(hidden_dimension),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension // 2),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Process pre-computed embeddings into fixed-size vectors.

        Args:
            x: Pre-computed embeddings tensor of shape
                (batch_size, seq_len, embedding_dim)
            lengths: Length of each sequence in the batch

        Returns:
            Encoded vectors of shape (batch_size, hidden_dimension // 2)
        """
        # RNN with packed sequence
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed_x)

        # FC layer
        hidden = h_n[-1]
        out = self.sequential(hidden)

        return out
