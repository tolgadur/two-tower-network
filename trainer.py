# from itertools import chain
import torch
from dataset import TwoTowerDataset
from two_tower import TowerOne, TowerTwo
from torch import nn
from torch.utils.data import DataLoader


def train(epochs: int = 10, batch_size: int = 128):
    dataset = TwoTowerDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    tower_one = TowerOne().to("mps")
    tower_two = TowerTwo().to("mps")

    criterion = nn.TripletMarginWithDistanceLoss(
        margin=0.3,
        distance_function=lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y),
    )
    optimizer = torch.optim.Adam(
        [
            {"params": tower_one.parameters(), "lr": 0.001},
            {"params": tower_two.parameters(), "lr": 0.001},
        ]
    )
    # optimizer = torch.optim.Adam(chain(tower_one.parameters(), tower_two.parameters), lr=0.001)

    for epoch in range(epochs):
        for query, positive, negative in dataloader:
            # Move data to device
            query = query.to("mps")
            positive = positive.to("mps")
            negative = negative.to("mps")

            query_embedding = tower_one(query)
            positive_embedding = tower_two(positive)
            negative_embedding = tower_two(negative)

            loss = criterion(query_embedding, positive_embedding, negative_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return tower_one, tower_two
