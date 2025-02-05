from datasets import load_dataset
import random
from tqdm import tqdm
import pandas as pd

random.seed(42)


def generate_training_triplets(dataset, savePath=""):
    """
    Generate training triplets from the MS MARCO dataset.
    For each passage in a query's passage list, creates one row with:
    (query, positive passage, randomly sampled negative passage).
    The negative passage is guaranteed to not be in the original passage list.

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

        # For each positive passage in the current row
        for positive_passage in row["passages"]["passage_text"]:
            # Sample passages from random rows until we find one not in current_passages
            while True:
                random_row_idx = random.randint(0, dataset_size - 1)
                random_passage_list = dataset[random_row_idx]["passages"][
                    "passage_text"
                ]
                # Sample one random passage from the randomly selected row
                negative_passage = random.choice(random_passage_list)
                if negative_passage not in current_passages:
                    break

            # Add the triplet
            triplets.append((query, positive_passage, negative_passage))

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


def extract_unique_documents(save_path="data/unique_documents.parquet"):
    """
    Extracts all unique documents from the MS MARCO dataset and
    saves them to a parquet file.

    Args:
        save_path: Path where to save the unique documents parquet file

    Returns:
        DataFrame containing all unique documents
    """
    print("Loading dataset...")
    ds = load_dataset("microsoft/ms_marco", "v1.1")

    # Set to store unique documents
    unique_documents = set()

    # Process each split
    for split in ["train", "validation", "test"]:
        print(f"\nExtracting documents from {split} split...")
        for row in tqdm(ds[split]):
            unique_documents.update(row["passages"]["passage_text"])

    # Convert to DataFrame
    df = pd.DataFrame(list(unique_documents), columns=["document"])

    # Save to parquet
    df.to_parquet(save_path, index=False)
    print(f"\nSaved {len(df)} unique documents to {save_path}")
    return df


def load_unique_documents(path="data/unique_documents.parquet"):
    # Load and print first 10 documents
    documents_df = pd.read_parquet(path)
    print("\nFirst 10 documents:")
    for idx, doc in enumerate(documents_df["document"][:10]):
        print(f"\n{idx + 1}. {doc[:200]}...")  # Print first 200 chars of each doc
    print(documents_df["document"].nunique())


# extract_unique_documents()
# load_unique_documents("data/unique_documents.parquet")

triplets_to_dataset()
triplets = load_triplets("data/train_triplets.parquet")
print("Length of triplets:")
print(len(triplets))
# print(triplets["query"][0])
