import torch
import gensim.downloader as api
import pandas as pd

quick_test_queries = [
    "how to make coffee",
    "what is the capital of France",
    "best programming languages",
    "covid symptoms",
    "chocolate cake recipe",
]

dummy_triplets = pd.DataFrame(
    [
        (
            "what is ai",
            "artificial intelligence is the simulation of human intelligence",
            "the capital of france is paris",
        ),
        (
            "how to cook pasta",
            "cooking pasta involves boiling water and adding salt",
            "the stock market closed higher today",
        ),
        (
            "what is machine learning",
            "machine learning is a branch of artificial intelligence",
            "cats are cute",
        ),
        (
            "explain photosynthesis",
            "photosynthesis converts light energy into chemical energy",
            "computers process data",
        ),
    ],
    columns=["query", "positive_passage", "negative_passage"],
)


class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame = dummy_triplets):
        self.word2vec = api.load("word2vec-google-news-300")
        self.data = data
        self.embedding_dim = 300

    def _text_to_embeddings(self, text: str, max_length: int = 30) -> torch.Tensor:
        words = text.lower().split()[:max_length]
        embeddings = []

        for word in words:
            if word in self.word2vec:
                emb = self.word2vec[word].copy()
                embeddings.append(torch.FloatTensor(emb))
            else:
                embeddings.append(torch.zeros(self.embedding_dim))

        if not embeddings:
            return torch.zeros(self.embedding_dim)

        # Stack and mean pool
        stacked = torch.stack(embeddings)  # Shape: [num_words, embedding_dim]
        return torch.mean(stacked, dim=0)  # Shape: [embedding_dim]

    def _text_to_embeddings_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self._text_to_embeddings(text) for text in texts])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.data["query"].iloc[index]
        positive = self.data["positive_passage"].iloc[index]
        negative = self.data["negative_passage"].iloc[index]

        query_embedding = self._text_to_embeddings(query)
        positive_embedding = self._text_to_embeddings(positive)
        negative_embedding = self._text_to_embeddings(negative)

        return (
            query_embedding,
            positive_embedding,
            negative_embedding,
        )

    def __len__(self):
        return len(self.data)
