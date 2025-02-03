from datasets import load_dataset
import random
from tqdm import tqdm

random.seed(42)


def generate_training_triplets(dataset):
    """
    Generate training triplets from the MS MARCO dataset.
    Each triplet contains (query, positive passages, negative passages).
    All 10 passages for a query are positive, and 10 random passages
    from other queries are negative.

    Args:
        dataset: MS MARCO dataset split (train/validation/test)

    Returns:
        list of dictionaries containing:
            - query: the question text
            - positive_passages: list of relevant passage texts
            - negative_passages: list of non-relevant passage texts
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

        triplet = {
            "query": row["query"],
            "positive_passages": row["passages"]["passage_text"],
            "negative_passages": negative_passages,
        }
        triplets.append(triplet)

    return triplets


print("Loading dataset...")
ds = load_dataset("microsoft/ms_marco", "v1.1")

# Generate triplets for each split
print("\nProcessing train split...")
train_triplets = generate_training_triplets(ds["train"])
print("\nProcessing validation split...")
validation_triplets = generate_training_triplets(ds["validation"])
print("\nProcessing test split...")
test_triplets = generate_training_triplets(ds["test"])

# Print some statistics
print(f"\nNumber of training triplets: {len(train_triplets)}")
print(f"Number of validation triplets: {len(validation_triplets)}")
print(f"Number of test triplets: {len(test_triplets)}")
