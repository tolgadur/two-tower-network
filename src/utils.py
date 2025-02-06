import torch
from two_tower import TowerOne, TowerTwo
from dataset import TwoTowerDataset
from trainer import train
from inference import TwoTowerInference
from config import DEVICE
from random import sample
from datasets import load_dataset


def load_two_tower_dataset():
    """Load and return the training dataset."""
    print("Loading dataset...")
    dataset = TwoTowerDataset("data/train_triplets.parquet")
    print("First item from dataset:", dataset[0])
    return dataset


def train_two_tower():
    """Train the two-tower model with default parameters."""
    print("Starting two-tower model training...")
    train(epochs=15, batch_size=4, save_model=True)


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


def evaluate_model_on_test_queries(
    inference: TwoTowerInference,
    k: int = 5,
    num_examples: int = 1,
):
    """Evaluate the model on test queries and display results.

    Args:
        inference: Initialized TwoTowerInference instance with documents loaded
        test_path: Path to test triplets parquet file
        k: Number of neighbors to retrieve
        num_examples: Number of test examples to evaluate
    """
    # Load raw test data to get all positive passages
    print("Loading MS MARCO dataset...")
    ds = load_dataset("microsoft/ms_marco", "v1.1")
    test_ds = ds["test"]

    # Sample random test examples
    test_indices = (
        sample(range(len(test_ds)), num_examples)
        if num_examples < len(test_ds)
        else range(len(test_ds))
    )

    print(f"\nEvaluating {len(test_indices)} test queries...")
    print("-" * 80)

    total_hits = 0
    total_queries = 0
    best_ranks = []

    for idx in test_indices:
        query = test_ds[idx]["query"]
        positive_passages = set(test_ds[idx]["passages"]["passage_text"])

        # Get model predictions
        neighbors, scores = inference.kNN(query, k=k)

        # Print results
        print(f"\nQuery: {query}")
        print(f"\nNumber of Ground Truth Positives: {len(positive_passages)}")
        print("\nGround Truth Positive Passages:")
        for i, pos in enumerate(positive_passages, 1):
            print(f"\n{i}. {pos}")

        print(f"\nTop {k} Retrieved Documents:")
        for doc, score in zip(neighbors, scores):
            print(f"\nScore: {score:.4f}")
            print(doc)
        print("-" * 80)

        # Calculate metrics
        hits = [doc for doc in neighbors if doc in positive_passages]
        if hits:
            total_hits += 1
            # Find the best (lowest) rank among all positive passages
            best_rank = min(neighbors.index(doc) + 1 for doc in hits)
            best_ranks.append(best_rank)
            print(f"✓ Found {len(hits)} positive document(s). Best rank: {best_rank}")
        else:
            print("✗ No positive documents found in top k")
        total_queries += 1

    # Print summary metrics
    print("\nSummary Metrics:")
    print(f"Recall@{k}: {total_hits / total_queries:.2%}")
    if best_ranks:
        print(f"Mean Best Rank: {sum(best_ranks) / len(best_ranks):.2f}")
        print(f"Best Rank Distribution: {best_ranks}")
