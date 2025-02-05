import torch
from two_tower import TowerOne, TowerTwo
from dataset import TwoTowerDataset
from trainer import train
from inference import TwoTowerInference
from config import DEVICE


def load_dataset():
    """Load and return the training dataset."""
    print("Loading dataset...")
    dataset = TwoTowerDataset("data/train_triplets.parquet")
    print("First item from dataset:", dataset[0])
    return dataset


def train_two_tower():
    """Train the two-tower model with default parameters."""
    print("Starting two-tower model training...")
    train(epochs=5, save_model=True)


def test_encode_query():
    """Test query encoding with a sample query."""
    tower_one = load_tower_one()
    tower_two = load_tower_two()
    query = "What is the capital of France?"
    inference = TwoTowerInference(tower_one, tower_two)

    query_embedding = inference.encode_query(query)
    print(f"Query embedding: {query_embedding}")
    print(f"Shape of embedding: {query_embedding.shape}")


def test_encode_document():
    """Test document encoding with a sample document."""
    tower_one = load_tower_one()
    tower_two = load_tower_two()
    document = "Paris is the capital of France."
    inference = TwoTowerInference(tower_one, tower_two)

    document_embedding = inference.encode_document(document)
    print(f"Document embedding: {document_embedding}")
    print(f"Shape of embedding: {document_embedding.shape}")


def test_encode_documents():
    """Test batch document encoding from file."""
    tower_one = load_tower_one()
    tower_two = load_tower_two()
    inference = TwoTowerInference(tower_one, tower_two)

    inference.encode_documents()
    print(f"Document encodings: {inference.document_encodings}")
    print(f"Shape of document encodings: {inference.document_encodings.shape}")


def load_tower_one():
    """Load the query tower model."""
    tower_one = TowerOne().to(DEVICE)
    tower_one.load_state_dict(
        torch.load("models/tower_one.pt", weights_only=True, map_location=DEVICE)
    )
    tower_one.eval()
    return tower_one


def load_tower_two():
    """Load the document tower model."""
    tower_two = TowerTwo().to(DEVICE)
    tower_two.load_state_dict(
        torch.load("models/tower_two.pt", weights_only=True, map_location=DEVICE)
    )
    tower_two.eval()
    return tower_two
