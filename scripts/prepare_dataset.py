from datasets import load_dataset
import random
from tqdm import tqdm
import pandas as pd

random.seed(42)


def generate_training_triplets(dataset, savePath=""):
    """
    Generate training triplets from the MS MARCO dataset.
    Each triplet contains (query, positive passage, negative passage).
    For each query, creates all possible combinations of its positive passages
    with randomly sampled negative passages.

    Args:
        dataset: MS MARCO dataset split (train/validation/test)
        savePath: Optional path to save the triplets as parquet file

    Returns:
        DataFrame containing:
            - query: the question text
            - positive_passage: a relevant passage text
            - negative_passage: a non-relevant passage text
    """
    dataset_size = len(dataset)
    triplets = []

    print("Generating triplets...")
    for row in tqdm(dataset):
        # Get current passages to exclude from negative sampling
        current_passages = set(row["passages"]["passage_text"])
        query = row["query"]
        positive_passages = row["passages"]["passage_text"]

        # Sample passages from random rows
        candidate_passages = [
            passage
            for idx in random.sample(range(dataset_size), 20)
            for passage in dataset[idx]["passages"]["passage_text"]
            if passage not in current_passages
        ]
        negative_passages = random.sample(candidate_passages, 10)

        # Generate combinations for this query and extend the triplets list
        query_triplets = [
            (query, pos, neg) for pos in positive_passages for neg in negative_passages
        ]
        triplets.extend(query_triplets)

    if len(savePath):
        df = pd.DataFrame(
            triplets, columns=["query", "positive_passage", "negative_passage"]
        )
        df.to_parquet(savePath, index=False)
        print(f"Saved triplets to {savePath}")
    return pd.DataFrame(
        triplets, columns=["query", "positive_passage", "negative_passage"]
    )


def load_triplets(path):
    """
    Load triplets from a parquet file.

    Args:
        path: Path to the parquet file containing triplets

    Returns:
        DataFrame containing:
            - query: the question text
            - positive_passage: a relevant passage text
            - negative_passage: a non-relevant passage text
    """
    return pd.read_parquet(path)


def triplets_to_dataset():
    """
    Downloads the data and converts to triplets and saves them to parquet files.
    """
    print("Loading dataset...")
    ds = load_dataset("microsoft/ms_marco", "v1.1")

    # Generate triplets for each split
    print("\nProcessing train split...")
    train_triplets = generate_training_triplets(
        ds["train"], savePath="data/train_triplets.parquet"
    )
    print("\nProcessing validation split...")
    validation_triplets = generate_training_triplets(
        ds["validation"], savePath="data/validation_triplets.parquet"
    )
    print("\nProcessing test split...")
    test_triplets = generate_training_triplets(
        ds["test"], savePath="data/test_triplets.parquet"
    )

    # Print some statistics
    print(f"\nNumber of training triplets: {len(train_triplets)}")
    print(f"Number of validation triplets: {len(validation_triplets)}")
    print(f"Number of test triplets: {len(test_triplets)}")


triplets_to_dataset()
# triplets = load_triplets("data/train_triplets.parquet")
# print(triplets["query"][0])
