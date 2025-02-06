# from itertools import chain
import pandas as pd
import torch
import wandb
from dataset import TwoTowerDataset
from two_tower import TowerOne, TowerTwo
from torch import nn
from torch.utils.data import DataLoader
from config import DEVICE


def validate(
    tower_one: TowerOne,
    tower_two: TowerTwo,
    val_dataloader: DataLoader,
    criterion: nn.TripletMarginWithDistanceLoss,
):
    tower_one.eval()
    tower_two.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for query, positive, negative in val_dataloader:
            query = query.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            query_embedding = tower_one(query)
            positive_embedding = tower_two(positive)
            negative_embedding = tower_two(negative)

            loss = criterion(query_embedding, positive_embedding, negative_embedding)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def train(epochs: int = 10, batch_size: int = 128):
    print("Training...")
    training_data = pd.read_parquet(
        "data/train_triplets.parquet",
        columns=["query", "positive_passage", "negative_passage"],
    )[:1000]
    dataset = TwoTowerDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loading validation data
    validation_data = pd.read_parquet(
        "data/validation_triplets.parquet",
        columns=["query", "positive_passage", "negative_passage"],
    )[:1000]
    val_dataset = TwoTowerDataset(validation_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    tower_one = TowerOne().to(DEVICE)
    tower_two = TowerTwo().to(DEVICE)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    wandb.init(project="mlx6-two-tower", name="two-tower-model")
    for epoch in range(epochs):
        tower_one.train()
        tower_two.train()

        for query, positive, negative in dataloader:
            # Move data to device
            query = query.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            query_embedding = tower_one(query)
            positive_embedding = tower_two(positive)
            negative_embedding = tower_two(negative)

            loss = criterion(query_embedding, positive_embedding, negative_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})

        scheduler.step(loss)
        val_loss = validate(tower_one, tower_two, val_dataloader, criterion)
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
        print(
            {
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        torch.save(tower_one.state_dict(), f"models/tower_one_{epoch}.pth")
        torch.save(tower_two.state_dict(), f"models/tower_two_{epoch}.pth")

    print("Saving final model to wandb...")
    torch.save(tower_one.state_dict(), "models/tower_one.pth")
    torch.save(tower_two.state_dict(), "models/tower_two.pth")
    wandb.save("models/tower_one.pth")
    wandb.save("models/tower_two.pth")

    wandb.finish()

    return tower_one, tower_two
