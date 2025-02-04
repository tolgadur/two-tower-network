import torch


class TowerOne(torch.nn.Module):
    def __init__(
        self,
        embedding_matrix,
        vocab_size,
        hidden_dimension=258,
        embedding_dim=258,
    ):
        super().__init__()

        # embedding layer from word2vec
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False

        # two parallel encoding layers (RNN based)
        self.model = torch.nn.Sequential(
            self.embedding,
            torch.nn.RNN(embedding_dim, hidden_dimension),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dimension, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TowerTwo(torch.nn.Module):
    def __init__(
        self,
        embedding_matrix,
        vocab_size,
        hidden_dimension=258,
        embedding_dim=258,
    ):
        super().__init__()

        # embedding layer from word2vec
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False

        # two parallel encoding layers (RNN based)
        self.model = torch.nn.Sequential(
            self.embedding,
            torch.nn.RNN(embedding_dim, hidden_dimension),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dimension, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
