import torch
from two_tower import TowerOne, TowerTwo
from tokenizer import Tokenizer
import pandas as pd
from tqdm import tqdm


def encode_query(query: str, tokenizer: Tokenizer, tower_one: TowerOne) -> torch.Tensor:
    """
    Encode a query string using TowerOne.

    Args:
        query: Input query string
        tower_one: Trained TowerOne model
        tokenizer: Optional Tokenizer instance. If None, will load from saved files.

    Returns:
        torch.Tensor: Query embedding of shape (hidden_dimension,)
    """

    # Convert query to tensor
    query_tensor = tokenizer.text_to_tensor(query)

    # Add batch dimension and get length
    query_tensor = query_tensor.unsqueeze(0)  # shape: (1, seq_len)
    length = torch.tensor([len(query_tensor[0])])  # shape: (1,)

    # Get embedding
    with torch.no_grad():
        embedding = tower_one(query_tensor, length)  # shape: (1, hidden_dimension)

    return embedding.squeeze(0)  # shape: (hidden_dimension,)


def encode_document(
    document: str, tokenizer: Tokenizer, tower_two: TowerTwo
) -> torch.Tensor:
    """
    Encode a document string using TowerTwo.

    Args:
        document: Input document string
        tower_two: Trained TowerTwo model
        tokenizer: Optional Tokenizer instance. If None, will load from saved files.

    Returns:
        torch.Tensor: Document embedding of shape (hidden_dimension,)
    """

    # Convert document to tensor
    document_tensor = tokenizer.text_to_tensor(document)

    # Add batch dimension and get length
    document_tensor = document_tensor.unsqueeze(0)  # shape: (1, seq_len)
    length = torch.tensor([len(document_tensor[0])])  # shape: (1,)

    # Get embedding
    with torch.no_grad():
        embedding = tower_two(document_tensor, length)  # shape: (1, hidden_dimension)

    return embedding.squeeze(0)  # shape: (hidden_dimension,)


def create_document_encodings(
    tokenizer: Tokenizer,
    tower_two: TowerTwo,
    documents_path: str = "data/unique_documents.parquet",
) -> tuple[dict[str, torch.Tensor], dict[torch.Tensor, str]]:
    """
    Load all unique documents and create bidirectional mappings between documents and their encodings.

    Args:
        tokenizer: Tokenizer instance for text processing
        tower_two: Trained TowerTwo model for document encoding
        documents_path: Path to the parquet file containing unique documents

    Returns:
        tuple containing:
            - dict mapping document strings to their encodings
            - dict mapping encodings to their document strings
    """
    print("Loading unique documents...")
    documents_df = pd.read_parquet(documents_path)

    doc_to_encoding = {}
    encoding_to_doc = {}

    print("Encoding documents...")
    for doc in tqdm(documents_df["document"]):
        encoding = encode_document(doc, tokenizer, tower_two)
        doc_to_encoding[doc] = encoding
        encoding_to_doc[encoding] = doc

    print(f"Created mappings for {len(doc_to_encoding)} documents")
    return doc_to_encoding, encoding_to_doc
