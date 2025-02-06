import torch
import pandas as pd
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
    def __init__(self, data_path: str):
        """Initialize dataset with parquet file path.

        Args:
            data_path: Path to parquet file containing query, positive_passage,
                      and negative_passage columns
        """
        # self.data = pd.read_parquet(data_path)[:100]

        self.data = pd.DataFrame(
            dummy_triplets, columns=["query", "positive_passage", "negative_passage"]
        )
        # Load Google News embeddings
        self.word2vec = api.load("word2vec-google-news-300")
        self.embedding_dim = 300

    def __len__(self):
        return len(self.data)

    def _text_to_embeddings(
        self, text: str, max_length: int = 30
    ) -> tuple[torch.Tensor, int]:
        """Convert text to word embeddings.

        Args:
            text: Input text to convert to embeddings

        Returns:
            tuple of:
                - tensor of shape (seq_len, embedding_dim) containing embeddings
                - length of the sequence (number of tokens)
        """
        # Split and lowercase the text
        words = text.lower().split()[:max_length]

        # Get embeddings for each word
        embeddings = []
        for word in words:
            if word in self.word2vec:
                embeddings.append(torch.FloatTensor(self.word2vec[word]))
            else:
                # For unknown words, use zero vector
                embeddings.append(torch.zeros(self.embedding_dim))

        if not embeddings:  # If no words were found
            return torch.zeros(1, self.embedding_dim), 1

        # Stack embeddings into a single tensor
        embeddings = torch.stack(embeddings)
        return embeddings, len(words)

    def __getitem__(
        self, index
    ) -> tuple[
        tuple[torch.Tensor, int], tuple[torch.Tensor, int], tuple[torch.Tensor, int]
    ]:
        """Get item from dataset.

        Returns:
            tuple of:
                - (query_embeddings, query_length)
                - (positive_embeddings, positive_length)
                - (negative_embeddings, negative_length)
            where each embedding tensor has shape (seq_len, embedding_dim)
        """
        # Get raw text
        query = self.data["query"][index]
        positive = self.data["positive_passage"][index]
        negative = self.data["negative_passage"][index]

        # Convert to embeddings
        query_emb = self._text_to_embeddings(query)
        pos_emb = self._text_to_embeddings(positive)
        neg_emb = self._text_to_embeddings(negative)

        return query_emb, pos_emb, neg_emb
