# from trainer import train
from dataset import EmbeddingsBuilder
from two_tower import TowerOne, TowerTwo
import torch
from config import DEVICE
from inference import Inference
import uvicorn
from api import create_endpoints
from utils import test_accuracy, sanity_test


def main():
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
    inference = Inference(tower_one, tower_two, EmbeddingsBuilder())
    inference.embed_docs()

    # Run accuracy test
    test_accuracy(inference, k=1)
    test_accuracy(inference, k=3)
    test_accuracy(inference, k=5)

    # Run sanity test
    sanity_test(inference)

    print("Setting up FastAPI application...")
    app = create_endpoints(inference)

    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, access_log=True)


if __name__ == "__main__":
    main()
