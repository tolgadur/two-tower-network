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
