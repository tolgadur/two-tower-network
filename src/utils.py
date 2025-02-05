import torch
from tokenizer import Tokenizer
from embeddings import SkipGramModel, EmbeddingTrainer
from dataset import TwoTowerDataset
from trainer import train
from two_tower import TowerOne, TowerTwo
from inference import TwoTowerInference
from config import DEVICE
import pandas as pd
from collections import Counter


def load_tokenizer_and_embeddings():
    tokenizer = Tokenizer(load_vocab=True)
    vocab_size = len(tokenizer.word2idx)
    model = SkipGramModel(vocab_size, 258)
    model.load_state_dict(torch.load("models/skipgram_model.pt", weights_only=True))
    model.eval()
    return tokenizer, model


def build_vocab():
    """Build and save vocabulary from both text8 and unique documents datasets."""
    tokenizer = Tokenizer(load_vocab=False)

    # Process text8 dataset
    text8_tokens = tokenizer.preprocess("data/text8", "data/final-text8")

    # Process unique documents
    df = pd.read_parquet("data/unique_documents.parquet")
    doc_tokens = tokenizer.process_documents(df["document"].tolist())

    # Combine tokens from both sources
    all_tokens = text8_tokens + doc_tokens

    # Build and save the combined vocabulary
    tokenizer.build_vocab(all_tokens)


def train_word2vec():
    tokenizer = Tokenizer(load_vocab=True)
    vocab_size = len(tokenizer.word2idx)
    print(f"Vocabulary size: {vocab_size}")

    # load tokens from file
    with open("data/final-text8", "r") as f:
        tokens = f.read().split()
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"Number of tokens loaded: {len(tokens)}")

    # convert tokens to indices
    token_indices = [
        tokenizer.word2idx[token] for token in tokens if token in tokenizer.word2idx
    ]
    print(f"Number of valid tokens after getting indices: {len(token_indices)}")

    # train model
    model = SkipGramModel(vocab_size, 258)
    trainer = EmbeddingTrainer(model, token_indices, batch_size=1024)
    trainer.train(vocab_size=vocab_size, save_model=True, num_epochs=5)


def embedding_test():
    tokenizer, model = load_tokenizer_and_embeddings()

    # test model
    test_word = "the"
    tensor = torch.LongTensor([tokenizer.word2idx[test_word]])
    test_embedding = model.compute_embedding(tensor)
    print(f"Embedding for {test_word}: {test_embedding}")
    print(f"Shape of embedding: {test_embedding.shape}")


def load_dataset():
    tokenizer, _ = load_tokenizer_and_embeddings()

    print("Loading dataset...")
    dataset = TwoTowerDataset("data/train_triplets.parquet", tokenizer)
    print(dataset[0])

    return dataset


def train_two_tower():
    tokenizer, model = load_tokenizer_and_embeddings()
    train(
        tokenizer=tokenizer,
        embedding_model=model,
        epochs=5,
        save_model=True,
    )


def test_encode_query():
    tokenizer, _ = load_tokenizer_and_embeddings()
    tower_one = load_tower_one()
    tower_two = load_tower_two()
    query = "What is the capital of France?"
    inference = TwoTowerInference(tower_one, tower_two, tokenizer)

    query_embedding = inference.encode_query(query, tokenizer, tower_one)
    print(f"Query embedding: {query_embedding}")
    print(f"Shape of embedding: {query_embedding.shape}")


def test_encode_document():
    tokenizer, _ = load_tokenizer_and_embeddings()
    tower_one = load_tower_one()
    tower_two = load_tower_two()
    document = "Paris is the capital of France."
    inference = TwoTowerInference(tower_one, tower_two, tokenizer)

    document_embedding = inference.encode_document(document)
    print(f"Document embedding: {document_embedding}")
    print(f"Shape of embedding: {document_embedding.shape}")


def test_encode_documents():
    tokenizer, _ = load_tokenizer_and_embeddings()
    tower_one = load_tower_one()
    tower_two = load_tower_two()
    inference = TwoTowerInference(tower_one, tower_two, tokenizer)

    inference.encode_documents_by_filename()
    print(f"Document encodings: {inference.document_encodings}")
    print(f"Shape of document encodings: {inference.document_encodings.shape}")


def load_tower_one():
    tokenizer, embedding_model = load_tokenizer_and_embeddings()

    tower_one = TowerOne(
        embedding_matrix=embedding_model.embedding.weight,
        vocab_size=len(tokenizer.word2idx),
        hidden_dimension=258,
        embedding_dim=258,
    )

    tower_one.load_state_dict(
        torch.load("models/tower_one.pt", weights_only=True, map_location=DEVICE)
    )
    tower_one.eval()
    tower_one.to(DEVICE)

    return tower_one


def load_tower_two():
    tokenizer, embedding_model = load_tokenizer_and_embeddings()

    tower_two = TowerTwo(
        embedding_matrix=embedding_model.embedding.weight,
        vocab_size=len(tokenizer.word2idx),
    )

    tower_two.load_state_dict(
        torch.load("models/tower_two.pt", weights_only=True, map_location=DEVICE)
    )
    tower_two.eval()
    tower_two.to(DEVICE)

    return tower_two


def evaluate_tokenizer_coverage():
    """
    Evaluates the tokenizer's coverage by analyzing the percentage of unknown tokens
    in the unique documents dataset.
    """

    # Load tokenizer
    tokenizer = Tokenizer(load_vocab=True)

    # Load unique documents
    df = pd.read_parquet("data/unique_documents.parquet")

    total_tokens = 0
    unknown_tokens = 0
    unknown_token_examples = Counter()

    # Process each document
    for doc in df["document"]:
        tokens = tokenizer.preprocess_text(doc)
        total_tokens += len(tokens)

        # Count unknown tokens
        for token in tokens:
            if token not in tokenizer.word2idx:
                unknown_tokens += 1
                unknown_token_examples[token] += 1

    # Calculate percentage
    unknown_percentage = (
        (unknown_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    )

    print(f"\nTokenizer Coverage Analysis:")
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Unknown tokens found: {unknown_tokens:,}")
    print(f"Percentage of unknown tokens: {unknown_percentage:.2f}%")

    # Print most common unknown tokens
    print("\nTop 10 most common unknown tokens:")
    for token, count in unknown_token_examples.most_common(10):
        print(f"'{token}': {count:,} occurrences")
