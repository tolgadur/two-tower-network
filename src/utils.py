import torch
from tokenizer import Tokenizer
from embeddings import SkipGramModel, EmbeddingTrainer
from dataset import TwoTowerDataset
from trainer import train
from two_tower import TowerOne, TowerTwo
import inference


def load_tokenizer_and_embeddings():
    tokenizer = Tokenizer(load_vocab=True)
    vocab_size = len(tokenizer.word2idx)
    model = SkipGramModel(vocab_size, 258)
    model.load_state_dict(torch.load("models/skipgram_model.pt", weights_only=True))
    model.eval()
    return tokenizer, model


def build_vocab():
    """Build and save vocabulary from both datasets."""
    tokenizer = Tokenizer(load_vocab=False)
    tokens = tokenizer.preprocess("data/text8", "data/final-text8")
    tokenizer.build_vocab(tokens)


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
    query = "What is the capital of France?"
    query_embedding = inference.encode_query(query, tokenizer, tower_one)
    print(f"Query embedding: {query_embedding}")
    print(f"Shape of embedding: {query_embedding.shape}")


def test_encode_document():
    tokenizer, _ = load_tokenizer_and_embeddings()
    tower_two = load_tower_two()
    document = "Paris is the capital of France."
    document_embedding = inference.encode_document(document, tokenizer, tower_two)
    print(f"Document embedding: {document_embedding}")
    print(f"Shape of embedding: {document_embedding.shape}")


def load_tower_one():
    tokenizer, embedding_model = load_tokenizer_and_embeddings()

    tower_one = TowerOne(
        embedding_matrix=embedding_model.embedding.weight,
        vocab_size=len(tokenizer.word2idx),
        hidden_dimension=258,
        embedding_dim=258,
    )

    tower_one.load_state_dict(torch.load("models/tower_one.pt", weights_only=True))
    tower_one.eval()

    return tower_one
