import torch
import pandas as pd
import faiss
from two_tower import TowerOne, TowerTwo
from tokenizer import Tokenizer
from config import DEVICE
from tqdm import tqdm


class TwoTowerInference:
    def __init__(
        self,
        tower_one: TowerOne,
        tower_two: TowerTwo,
        tokenizer: Tokenizer,
        embedding_dimension: int = 258,
    ):
        """
        Initialize TwoTowerInference with models and tokenizer.

        Args:
            tower_one: Trained TowerOne model
            tower_two: Trained TowerTwo model
            tokenizer: Tokenizer instance for text processing
            embedding_dimension: Dimension of the embeddings, defaults to 258
        """
        self.tower_one = tower_one
        self.tower_two = tower_two
        self.tokenizer = tokenizer
        self.documents = []
        self.document_encodings = None
        self.faiss_index = faiss.IndexFlatIP(embedding_dimension)

    def kNN(self, query: str, k: int = 5) -> tuple[list[str], list[float]]:
        """
        Find k nearest neighbors for a query string.

        Args:
            query: The query string to find similar documents for
            k: Number of nearest neighbors to return (default: 5)

        Returns:
            tuple[list[str], list[float]]: A tuple containing:
                - List of k most similar documents
                - List of corresponding similarity scores
        """
        # Encode and normalize query
        query_encoding = self.encode_query(query)
        normalized_query = self._normalize_vector(query_encoding)

        # Convert to numpy for FAISS
        query_np = normalized_query.cpu().numpy()

        # Search the index
        scores, indices = self.faiss_index.search(query_np, k)

        # Get the corresponding documents
        documents = [self.documents[idx] for idx in indices[0]]
        scores = scores[0].tolist()  # Convert scores to list

        return documents, scores

    def add_document_to_index(self, document: str):
        """
        Encode a single document and add it to the FAISS index.

        Args:
            document: Input document string
        """
        # Encode and normalize the document
        encoding = self.encode_document(document)
        normalized_encoding = self._normalize_vector(encoding)

        # Convert to numpy for FAISS
        encoding_np = normalized_encoding.cpu().numpy()

        # Add to index
        self.faiss_index.add(encoding_np)

    def add_documents_from_file(
        self, filename: str = "data/unique_documents.parquet", batch_size: int = 258
    ):
        """
        Load documents from a file, encode them, and add them to the FAISS index.

        Args:
            filename: Path to the parquet file containing documents.
                Default is "data/unique_documents.parquet"
            batch_size: Number of documents to process at once. Default is 258.
        """
        # Encode all documents
        encodings = self.encode_documents_by_filename(filename, batch_size)

        # Normalize the encodings
        normalized_encodings = self._normalize_matrix(encodings)

        # Convert to numpy for FAISS
        encodings_np = normalized_encodings.cpu().numpy()

        # Add to index
        self.faiss_index.add(encodings_np)

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query string using TowerOne.

        Args:
            query: Input query string

        Returns:
            torch.Tensor: Query embedding of shape (hidden_dimension,)
        """
        # Convert query to tensor
        query_tensor = self.tokenizer.text_to_tensor(query)

        # Add batch dimension and get length
        query_tensor = query_tensor.unsqueeze(0).to(DEVICE)  # shape: (1, seq_len)
        length = torch.tensor([len(query_tensor[0])], dtype=torch.long)

        # Get embedding
        with torch.no_grad():
            encoding = self.tower_one(query_tensor, length)

        return encoding.squeeze(0)  # shape: (hidden_dimension,)

    def encode_document(self, document: str) -> torch.Tensor:
        """
        Encode a document string using TowerTwo.

        Args:
            document: Input document string

        Returns:
            torch.Tensor: Document embedding of shape (hidden_dimension,)
        """
        # Convert document to tensor
        document_tensor = self.tokenizer.text_to_tensor(document)

        # Add batch dimension and get length
        document_tensor = document_tensor.unsqueeze(0).to(DEVICE)  # shape: (1, seq_len)
        length = torch.tensor([len(document_tensor[0])], dtype=torch.long)

        # Get embedding
        with torch.no_grad():
            encoding = self.tower_two(document_tensor, length)

        return encoding.squeeze(0)  # shape: (hidden_dimension,)

    def encode_documents_by_filename(
        self, filename: str = "data/unique_documents.parquet", batch_size: int = 258
    ) -> torch.Tensor:
        """
        Load documents from a parquet file and encode all of them at once.

        Args:
            filename: Path to the parquet file containing documents.
                Default is "data/unique_documents.parquet"
            batch_size: Number of documents to process at once. Default is 32.

        Returns:
            torch.Tensor: Document encodings matrix of shape
                (num_documents, hidden_dimension)
        """
        # Load documents from parquet file
        df = pd.read_parquet(filename)
        self.documents = df["document"].tolist()

        all_encodings = []
        num_batches = (len(self.documents) + batch_size - 1) // batch_size

        # Process documents in batches
        for i in tqdm(
            range(0, len(self.documents), batch_size),
            total=num_batches,
            desc="Encoding documents",
        ):
            batch_docs = self.documents[i : i + batch_size]

            # Convert batch documents to tensors
            document_tensors = []
            lengths = []

            for doc in batch_docs:
                doc_tensor = self.tokenizer.text_to_tensor(doc)
                document_tensors.append(doc_tensor)
                lengths.append(len(doc_tensor))

            # Pad sequences in this batch
            max_len = max(lengths)
            padded_tensors = []

            for doc_tensor in document_tensors:
                if len(doc_tensor) < max_len:
                    padding = torch.zeros(max_len - len(doc_tensor), dtype=torch.long)
                    doc_tensor = torch.cat([doc_tensor, padding])
                padded_tensors.append(doc_tensor)

            # Create batch tensor and move to correct device
            batch_tensor = torch.stack(padded_tensors).to(DEVICE)
            length_tensor = torch.tensor(lengths, dtype=torch.long)

            # Get embeddings for this batch
            with torch.no_grad():
                batch_encodings = self.tower_two(batch_tensor, length_tensor)
                all_encodings.append(batch_encodings)

        # Concatenate all batches
        self.document_encodings = torch.cat(all_encodings, dim=0)
        return self.document_encodings

    def _normalize_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Normalize a single vector to unit length for cosine similarity.

        Args:
            vector: Input vector of shape (dim,) or (1, dim)

        Returns:
            torch.Tensor: Normalized vector of shape (1, dim)
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        norm = torch.norm(vector, p=2, dim=1, keepdim=True)
        return vector / norm

    def _normalize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Normalize a matrix of vectors (each row is a vector) to unit length.

        Args:
            matrix: Input matrix of shape (n, dim)

        Returns:
            torch.Tensor: Normalized matrix of shape (n, dim)
        """
        norm = torch.norm(matrix, p=2, dim=1, keepdim=True)
        return matrix / norm
