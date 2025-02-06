import torch
import gensim.downloader as api

docs = [
    "artificial intelligence is the simulation of human intelligence",
    "cooking pasta involves boiling water and adding salt",
    "the stock market closed higher today",
    "machine learning is a branch of artificial intelligence",
    "photosynthesis converts light energy into chemical energy",
    "computers process data",
    "the capital of france is paris",
    "cats are cute",
]

dummy_triplets = [
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
]


class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[tuple[str, str, str]] = dummy_triplets):
        self.word2vec = api.load("word2vec-google-news-300")
        self.data = data
        self.embedding_dim = 300

    def _text_to_embeddings(self, text: str, max_length: int = 30) -> torch.Tensor:
        words = text.lower().split()[:max_length]
        embeddings = []

        for word in words:
            if word in self.word2vec:
                emb = self.word2vec[word]
                embeddings.append(torch.FloatTensor(emb))
            else:
                embeddings.append(torch.zeros(self.embedding_dim))

        if not embeddings:
            return torch.zeros(self.embedding_dim)

        # Stack and mean pool
        stacked = torch.stack(embeddings)  # Shape: [num_words, embedding_dim]
        return torch.mean(stacked, dim=0)  # Shape: [embedding_dim]

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        query, positive, negative = self.data[index]
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
