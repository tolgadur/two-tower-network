import torch


class TwoTowerModel(torch.nn.Module):
    def __init__(
        self, embedding_matrix, hidden_dimension, vocab_size, embedding_dim=256
    ):
        super().__init__()

        # embedding layer from word2vec
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.copy_(embedding_matrix)
        self.embedding.requires_grad_ = False

        # two parallel encoding layers (RNN based)
        self.tower1 = torch.nn.Sequential(
            self.embedding,
            torch.nn.RNN(embedding_dim, hidden_dimension),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dimension, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1, 1),
        )

        self.tower2 = torch.nn.Sequential(
            self.embedding,
            torch.nn.RNN(embedding_dim, hidden_dimension),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dimension, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1, 1),
        )

    def forward(self, query_tensor, pos_tensors, neg_tensors):
        query_embedding = self.tower1(query_tensor)
        pos_embeddings = self.tower2(pos_tensors)
        neg_embeddings = self.tower2(neg_tensors)

        return query_embedding, pos_embeddings, neg_embeddings
