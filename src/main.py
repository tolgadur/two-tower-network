import dataset
from two_tower import TowerTwo, TowerOne
import utils
import gensim.downloader as api
import torch
from config import DEVICE

# At the top of the file, load word2vec once
word2vec = api.load("word2vec-google-news-300")


def encode_document(doc: str, tower_two: TowerTwo):
    words = doc.lower().split()

    embeddings = []
    for word in words:
        if word in word2vec:
            embeddings.append(torch.tensor(word2vec[word], dtype=torch.float32))
        else:
            embeddings.append(torch.zeros(300))

    embeddings = torch.stack(embeddings).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(words)])

    with torch.no_grad():
        encoding = tower_two(embeddings, lengths)

    return encoding.squeeze(0)


def encode_query(doc: str, tower_one: TowerOne):
    words = doc.lower().split()

    embeddings = []
    for word in words:
        if word in word2vec:
            embeddings.append(torch.tensor(word2vec[word], dtype=torch.float32))
        else:
            embeddings.append(torch.zeros(300))

    embeddings = torch.stack(embeddings).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(words)])

    with torch.no_grad():
        encoding = tower_one(embeddings, lengths)

    return encoding.squeeze(0)


def get_nearest_neighbour(
    query: str, docs: dict[str, torch.Tensor], tower_one: TowerOne
):
    query_encoding = encode_query(query, tower_one)

    # Stack all document encodings into a single tensor
    doc_texts = list(docs.keys())
    doc_encodings = torch.stack([docs[doc] for doc in doc_texts])

    # Compute similarities all at once
    similarities = torch.nn.functional.cosine_similarity(
        query_encoding.unsqueeze(0).to(DEVICE), doc_encodings.to(DEVICE)
    )

    # Get top k results (here k=1 since we want nearest neighbor)
    top_k = torch.topk(similarities, k=1)

    # Print all similarities
    for doc, sim in zip(doc_texts, similarities):
        print(f"Similarity to '{doc}': {sim:.4f}")

    print(f"Best similarity score: {top_k.values[0].item():.4f}")
    return doc_texts[top_k.indices[0].item()]


def main():
    utils.train_two_tower()

    # Load models
    tower_one = utils.load_tower_one()
    tower_two = utils.load_tower_two()

    # Initialize inference
    docs = {}
    for doc in dataset.docs:
        docs[doc] = encode_document(doc, tower_two)

    for q, _, _ in dataset.dummy_triplets:
        print(f"Query: {q}")
        nearest_neighbour = get_nearest_neighbour(q, docs, tower_one)
        print(f"Nearest neighbour: {nearest_neighbour}")
        print("--------------------------------")


if __name__ == "__main__":
    main()
