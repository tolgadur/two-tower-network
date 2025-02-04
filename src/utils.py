import torch
from tokenizer import Tokenizer
from embeddings import SkipGramModel, EmbeddingTrainer
from dataset import TwoTowerDataset
from trainer import train


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
    print(f"Tokens: {tokens[:10]}")
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


def load_dataset():
    tokenizer, model = load_tokenizer_and_embeddings()

    print("Loading dataset...")
    dataset = TwoTowerDataset("data/train_triplets.csv", tokenizer, model)
    print(dataset[0])

    return dataset


def train_two_tower():
    tokenizer, model = load_tokenizer_and_embeddings()
    train(
        tokenizer=tokenizer,
        embedding_model=model,
        epochs=5,
        save_model=True,
        model_path="models/two_tower_model.pt",
    )
