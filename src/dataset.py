import torch
import gensim.downloader as api
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


quick_test_queries = [
    "how to make coffee",
    "what is the capital of France",
    "best programming languages",
    "covid symptoms",
    "chocolate cake recipe",
]


class EmbeddingsBuilder:
    def __init__(self):
        self.word2vec = api.load("word2vec-google-news-300")
        self.embedding_dim = 300

    def text_to_embeddings(self, text: str, max_length: int = 30) -> torch.Tensor:
        words = text.lower().split()[:max_length]
        embeddings = []

        for word in words:
            if word in self.word2vec:
                emb = self.word2vec[word].copy()
                embeddings.append(torch.FloatTensor(emb))
            else:
                embeddings.append(torch.zeros(self.embedding_dim))

        if not embeddings:
            # Return a single zero vector as a sequence of length 1
            return torch.zeros(1, self.embedding_dim)

        # Stack embeddings to get sequence
        return torch.stack(embeddings)  # Shape: [seq_len, embedding_dim]

    def text_to_embeddings_batch(
        self, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = [self.text_to_embeddings(text) for text in texts]
        lengths = torch.tensor([len(emb) for emb in embeddings])
        padded = pad_sequence(embeddings, batch_first=True)

        return padded, lengths


class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        embeddings_builder: EmbeddingsBuilder = EmbeddingsBuilder(),
    ):
        self.data = data
        self.embeddings_builder = embeddings_builder

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.data["query"].iloc[index]
        positive = self.data["positive_passage"].iloc[index]
        negative = self.data["negative_passage"].iloc[index]

        query_embedding = self.embeddings_builder.text_to_embeddings(query)
        positive_embedding = self.embeddings_builder.text_to_embeddings(positive)
        negative_embedding = self.embeddings_builder.text_to_embeddings(negative)

        return (
            query_embedding,
            positive_embedding,
            negative_embedding,
        )

    def __len__(self):
        return len(self.data)
