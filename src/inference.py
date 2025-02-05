import torch
import pandas as pd
import faiss
import gensim.downloader as api
from two_tower import TowerOne, TowerTwo
from tqdm import tqdm
from config import DEVICE


class TwoTowerInference:
    def __init__(
        self,
        tower_one: TowerOne,
        tower_two: TowerTwo,
        embedding_dimension: int = 129,  # 258 // 2, matches our architecture
    ):
        """Initialize TwoTowerInference with models.

        Args:
            tower_one: Query tower model
            tower_two: Document tower model
            embedding_dimension: Dimension of final embeddings, defaults to 129 (258//2)
        """
        self.tower_one = tower_one
        self.tower_two = tower_two
        self.documents = []
        self.index = faiss.IndexFlatIP(embedding_dimension)

        # Load Google News embeddings
        self.word2vec = api.load("word2vec-google-news-300")
        self.embedding_dim = 300  # Google News vectors are 300d

    def kNN(self, query: str, k: int = 5) -> tuple[list[str], list[float]]:
        """Find k nearest neighbors for a query.

        Args:
            query: Query string
            k: Number of neighbors to return

        Returns:
            tuple of:
                - list of k most similar documents
                - list of similarity scores
        """
        if not self.documents:
            raise ValueError("No documents indexed")

        # Encode and normalize query
        query_encoding = self.encode_query(query)
        query_np = self._normalize_vector(query_encoding).cpu().numpy()

        # Search
        scores, indices = self.index.search(query_np, k)

        # Get documents
        documents = [self.documents[idx] for idx in indices[0]]
        scores = scores[0].tolist()

        return documents, scores

    def add_document_to_index(self, document: str):
        """Encode a single document and add it to the FAISS index.

        Args:
            document: Input document string
        """
        # Encode and normalize the document
        encoding = self.encode_document(document)
        normalized_encoding = self._normalize_vector(encoding)

        # Convert to numpy for FAISS
        encoding_np = normalized_encoding.cpu().numpy()

        # Add to documents list and index
        self.documents.append(document)
        self.index.add(encoding_np)

    def add_documents_from_file(
        self, filename: str = "data/unique_documents.parquet", batch_size: int = 512
    ):
        """Load documents from a file, encode them, and add them to the FAISS index.

        Args:
            filename: Path to the parquet file containing documents.
                Default is "data/unique_documents.parquet"
            batch_size: Number of documents to process at once.
        """
        # Load documents
        df = pd.read_parquet(filename)
        self.documents = df["document"].tolist()

        # Get document encodings
        all_encodings = []
        num_batches = (len(self.documents) + batch_size - 1) // batch_size

        # Process documents in batches
        for i in tqdm(
            range(0, len(self.documents), batch_size),
            total=num_batches,
            desc="Encoding documents",
        ):
            docs = self.documents[i : i + batch_size]
            encodings = self.encode_documents(docs)
            all_encodings.append(encodings)

        # Concatenate all batches and normalize
        all_encodings = torch.cat(all_encodings, dim=0)
        normalized_encodings = self._normalize_matrix(all_encodings)

        # Convert to numpy for FAISS
        encodings_np = normalized_encodings.cpu().numpy()

        # Reset the index and add the encodings
        self.index.reset()  # Clear any existing vectors
        self.index.add(encodings_np)  # Add all vectors at once

        print(f"Added {len(self.documents)} documents to FAISS index")

    def encode_query(self, query: str) -> torch.Tensor:
        """Encode a query string into a vector.

        Args:
            query: Query string to encode

        Returns:
            torch.Tensor: Query encoding of shape (hidden_dimension // 2,)
        """
        # Preprocess query
        words = query.lower().split()
        if not words:
            return torch.zeros(self.tower_one.sequential[-1].out_features).to(DEVICE)

        # Convert to embeddings
        embeddings = []
        for word in words:
            if word in self.word2vec:
                embeddings.append(
                    torch.tensor(self.word2vec[word], dtype=torch.float32)
                )
            else:
                embeddings.append(torch.zeros(self.embedding_dim))

        # Stack embeddings and add batch dimension
        embeddings = torch.stack(embeddings).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([len(words)], dtype=torch.int64)  # Keep lengths on CPU

        # Get encoding
        with torch.no_grad():
            encoding = self.tower_one(embeddings, lengths)

        return encoding.squeeze(0)

    def encode_document(self, document: str) -> torch.Tensor:
        """Encode a document string into a vector.

        Args:
            document: Document string to encode

        Returns:
            torch.Tensor: Document encoding of shape (hidden_dimension // 2,)
        """
        # Preprocess document
        words = document.lower().split()
        if not words:
            return torch.zeros(self.tower_two.sequential[-1].out_features).to(DEVICE)

        # Convert to embeddings
        embeddings = []
        for word in words:
            if word in self.word2vec:
                embeddings.append(
                    torch.tensor(self.word2vec[word], dtype=torch.float32)
                )
            else:
                embeddings.append(torch.zeros(self.embedding_dim))

        # Stack embeddings and add batch dimension
        embeddings = torch.stack(embeddings).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([len(words)], dtype=torch.int64)  # Keep lengths on CPU

        # Get encoding
        with torch.no_grad():
            encoding = self.tower_two(embeddings, lengths)

        return encoding.squeeze(0)

    def encode_documents(self, docs: list[str]) -> torch.Tensor:
        """Encode a batch of documents efficiently using padding and batch processing.

        Args:
            docs: List of documents to encode

        Returns:
            torch.Tensor: Document encodings matrix of shape (len(docs), hidden_dim//2)
        """
        from torch.nn.utils.rnn import pad_sequence

        # Preprocess all documents
        processed_docs = [doc.lower().split() for doc in docs]

        # Get lengths for packing
        lengths = torch.tensor([len(doc) for doc in processed_docs], dtype=torch.int64)

        # Convert words to embeddings
        batch_embeddings = []
        for words in processed_docs:
            doc_embeddings = []
            for word in words:
                if word in self.word2vec:
                    doc_embeddings.append(
                        torch.tensor(self.word2vec[word], dtype=torch.float32)
                    )
                else:
                    doc_embeddings.append(torch.zeros(self.embedding_dim))
            batch_embeddings.append(torch.stack(doc_embeddings))

        # Pad and stack into a single tensor (batch_size, max_len, embedding_dim)
        batch_embeddings = pad_sequence(batch_embeddings, batch_first=True).to(DEVICE)

        # Get encodings
        with torch.no_grad():
            encodings = self.tower_two(batch_embeddings, lengths)

        return encodings

    def _normalize_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize a vector to unit length.

        Args:
            vector: Input vector of shape (dim,) or (1, dim)

        Returns:
            Normalized vector of same shape
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        norm = torch.norm(vector, p=2, dim=1, keepdim=True)
        return vector / norm

    def _normalize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Normalize a matrix row-wise to unit length.

        Args:
            matrix: Input matrix of shape (n, dim) where each row is a vector

        Returns:
            Normalized matrix of same shape where each row has unit length
        """
        norm = torch.norm(matrix, p=2, dim=1, keepdim=True)
        return matrix / norm

    def save_index(self, path: str = "models/faiss-index.faiss"):
        """Save the FAISS index to disk.

        Args:
            path: Directory to save the index in
        """
        import os

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, path)
        print(f"Saved FAISS index to {path}")

    def load_index(
        self,
        path: str = "models/faiss-index.faiss",
        docs_path: str = "data/unique_documents.parquet",
    ):
        """Load a FAISS index and documents.

        Args:
            path: Directory containing the FAISS index
            docs_path: Path to the parquet file containing documents
        """
        import os

        # Load FAISS index
        if not os.path.exists(path):
            raise FileNotFoundError(f"No index found at {path}")
        self.index = faiss.read_index(path)

        # Load documents from parquet
        df = pd.read_parquet(docs_path)
        self.documents = df["document"].tolist()

        print(f"Loaded index with {len(self.documents)} documents")
