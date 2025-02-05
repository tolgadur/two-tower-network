import torch


class TowerOne(torch.nn.Module):
    def __init__(
        self,
        embedding_matrix,
        vocab_size,
        hidden_dimension=258,
        embedding_dim=258,
        dropout=0.1,
    ):
        super().__init__()

        # embedding layer from word2vec with frozen weights
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False

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
        Inputs:
            x: torch.Tensor, shape (batch_size, seq_len). Represents query sentence
                indices in vocabulary.
            lengths: torch.Tensor, shape (batch_size,). Length of each query sentence.
        Outputs:
            out: torch.Tensor, shape (batch_size, hidden_dimension // 2).
        """

        # Embed the input
        x = self.embedding(x)  # shape (batch_size, seq_len, embedding_dim)

        # RNN with packed sequence
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )  # shape (batch_size, seq_len, embedding_dim)
        _, h_n = self.rnn(packed_x)  # shape (batch_size, hidden_dimension)

        # FC layer
        hidden = h_n[-1]
        out = self.sequential(hidden)  # shape (batch_size, hidden_dimension)

        return out


class TowerTwo(torch.nn.Module):
    def __init__(
        self,
        embedding_matrix,
        vocab_size,
        hidden_dimension=258,
        embedding_dim=258,
        dropout=0.1,
    ):
        super().__init__()

        # embedding layer from word2vec with frozen weights
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False

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
        Inputs:
            x: torch.Tensor, shape (batch_size, seq_len). Represents document
                indices in vocabulary.
            lengths: torch.Tensor, shape (batch_size,). Length of each document.
        Outputs:
            out: torch.Tensor, shape (batch_size, hidden_dimension // 2).
        """

        # Embed the input
        x = self.embedding(x)  # shape (batch_size, seq_len, embedding_dim)

        # RNN with packed sequence
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )  # shape (batch_size, seq_len, embedding_dim)
        _, h_n = self.rnn(packed_x)  # shape (batch_size, hidden_dimension)

        # FC layer
        hidden = h_n[-1]
        out = self.sequential(hidden)  # shape (batch_size, hidden_dimension)

        return out
