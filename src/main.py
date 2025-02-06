from trainer import train
from dataset import TwoTowerDataset, quick_test_queries
from two_tower import TowerOne, TowerTwo
import torch
from config import DEVICE

from inference import Inference


# import pandas as pd

# def print_dataset():
#     df = pd.read_parquet("data/train_triplets.parquet")
#     print(df.head())
#     print(df.tail())


# def embed_docs(tower_two: TowerTwo, dataset_: TwoTowerDataset):
#     encodings = []
#     for doc in docs:
#         embedding = dataset_._text_to_embeddings(doc).to(DEVICE)
#         encoding = tower_two(embedding)
#         encodings.append(encoding)

#     return torch.stack(encodings)


# def find_nearest_neighbors(
#     query: str,
#     tower_one: TowerOne,
#     doc_encodings: list[torch.Tensor],
#     dataset_: TwoTowerDataset,
# ):
#     embedding = dataset_._text_to_embeddings(query).to(DEVICE)
#     query_encoding = tower_one(embedding)

#     similarities = torch.nn.functional.cosine_similarity(query_encoding, doc_encodings)
#     top_k = torch.topk(similarities, k=1)

#     # get the top document
#     val, idx = top_k

#     return docs[idx], val


def main():
    # print_dataset()

    # print("Starting training on ", DEVICE)
    # tower_one, tower_two = train(epochs=15, batch_size=1024)
    # print("Training complete.")

    print("Loading models...")
    tower_one = TowerOne().to(DEVICE)
    tower_two = TowerTwo().to(DEVICE)
    tower_one.load_state_dict(
        torch.load("models/tower_one.pth", map_location=DEVICE, weights_only=True)
    )
    tower_two.load_state_dict(
        torch.load("models/tower_two.pth", map_location=DEVICE, weights_only=True)
    )

    print("Set models to evaluation mode...")
    tower_one.eval()
    tower_two.eval()

    print("Embedding docs...")
    dataset_ = TwoTowerDataset()
    inference = Inference(tower_one, tower_two, dataset_)
    # doc_encodings = embed_docs(tower_two, dataset_)
    inference.embed_docs()

    print("Finding nearest neighbors...")
    for query in quick_test_queries:
        docs, vals = inference.find_nearest_neighbors(query=query, k=3)
        print(f"\nQuery: {query}")
        print("Top 3 matches:")
        for doc, val in zip(docs, vals):
            print(f"Similarity: {val:.4f} | Doc: {doc}")


if __name__ == "__main__":
    main()
