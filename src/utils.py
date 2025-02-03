from tokenizer import Tokenizer


def build_vocab():
    """Build and save vocabulary from both datasets."""
    tokenizer = Tokenizer(load_vocab=False)
    tokens = tokenizer.preprocess("data/text8", "data/final-text8")
    tokenizer.build_vocab(tokens)
