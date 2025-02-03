from datasets import load_dataset
import random
from tqdm import tqdm
import pandas as pd
import ast

random.seed(42)


def generate_training_triplets(dataset, savePath=""):
    """
    Generate training triplets from the MS MARCO dataset.
    Each triplet contains (query, positive passages, negative passages).
    All 10 passages for a query are positive, and 10 random passages
    from other queries are negative.

    Args:
        dataset: MS MARCO dataset split (train/validation/test)
        savePath: Optional path to save the triplets as CSV

    Returns:
        list of tuples containing:
            - query: the question text on the first position
            - positive_passages: list of relevant passage texts on the second position
            - negative_passages: list of non-relevant passage texts on the third position
    """
    dataset_size = len(dataset)
    triplets = []

    print("Generating triplets...")
    for row in tqdm(dataset):
        # Get current passages to exclude from negative sampling
        current_passages = set(row["passages"]["passage_text"])

        # Sample passages from random rows
        candidate_passages = [
            passage
            for idx in random.sample(range(dataset_size), 20)
            for passage in dataset[idx]["passages"]["passage_text"]
            if passage not in current_passages
        ]
        negative_passages = random.sample(candidate_passages, 10)

        triplet = (row["query"], row["passages"]["passage_text"], negative_passages)
        triplets.append(triplet)

    if len(savePath):
        df = pd.DataFrame(
            triplets, columns=["query", "positive_passages", "negative_passages"]
        )
        df.to_csv(savePath, index=False)
        print(f"Saved triplets to {savePath}")
    return triplets


def load_triplets(path):
    """
    Load triplets from a CSV file.

    Args:
        path: Path to the CSV file containing triplets

    Returns:
        list of tuples containing:
            - query: the question text on the first position
            - positive_passages: list of relevant passage texts on second position
            - negative_passages: list of non-relevant passage texts on third position
    """
    df = pd.read_csv(path)
    df["positive_passages"] = df["positive_passages"].apply(ast.literal_eval)
    df["negative_passages"] = df["negative_passages"].apply(ast.literal_eval)

    return df


def triplets_to_dataset():
    """
    Downloads the data and converts to triplets and saves them to csv files.
    """
    print("Loading dataset...")
    ds = load_dataset("microsoft/ms_marco", "v1.1")

    # Generate triplets for each split
    print("\nProcessing train split...")
    train_triplets = generate_training_triplets(
        ds["train"], savePath="data/train_triplets.csv"
    )
    print("\nProcessing validation split...")
    validation_triplets = generate_training_triplets(
        ds["validation"], savePath="data/validation_triplets.csv"
    )
    print("\nProcessing test split...")
    test_triplets = generate_training_triplets(
        ds["test"], savePath="data/test_triplets.csv"
    )

    # Print some statistics
    print(f"\nNumber of training triplets: {len(train_triplets)}")
    print(f"Number of validation triplets: {len(validation_triplets)}")
    print(f"Number of test triplets: {len(test_triplets)}")


# triplets_to_dataset()
triplets = load_triplets("data/train_triplets.csv")
print(triplets["query"][0])
