from trainer import train
from dataset import EmbeddingsBuilder, quick_test_queries
from two_tower import TowerOne, TowerTwo
import torch
from config import DEVICE

from inference import Inference


def main():
    # print_dataset()

    print("Starting training on ", DEVICE)
    tower_one, tower_two = train(epochs=15, batch_size=1024)
    print("Training complete.")

    # print("Loading models...")
    # tower_one = TowerOne().to(DEVICE)
    # tower_two = TowerTwo().to(DEVICE)
    # tower_one.load_state_dict(
    #     torch.load("models/tower_one.pth", map_location=DEVICE, weights_only=True)
    # )
    # tower_two.load_state_dict(
    #     torch.load("models/tower_two.pth", map_location=DEVICE, weights_only=True)
    # )

    # print("Set models to evaluation mode...")
    # tower_one.eval()
    # tower_two.eval()

    # print("Embedding docs...")
    # inference = Inference(tower_one, tower_two, EmbeddingsBuilder())
    # inference.embed_docs(batch_size=4, save_index=False)

    # print("Finding nearest neighbors...")
    # for query in quick_test_queries:
    #     docs, vals = inference.find_nearest_neighbors(query=query, k=3)
    #     print(f"\nQuery: {query}")
    #     print("Top 3 matches:")
    #     for doc, val in zip(docs, vals):
    #         print(f"Similarity: {val:.4f} | Doc: {doc}")


if __name__ == "__main__":
    main()
