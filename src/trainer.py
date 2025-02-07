# from itertools import chain
import pandas as pd
import torch
from tqdm import tqdm
import wandb
from dataset import TwoTowerDataset
from two_tower import TowerOne, TowerTwo
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import DEVICE


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    # Unpack batch using zip
    queries, positives, negatives = zip(*batch)

    # Get lengths before padding
    query_lengths = torch.tensor([len(x) for x in queries])
    positive_lengths = torch.tensor([len(x) for x in positives])
    negative_lengths = torch.tensor([len(x) for x in negatives])

    # Pad sequences
    queries_padded = pad_sequence(queries, batch_first=True)
    positives_padded = pad_sequence(positives, batch_first=True)
    negatives_padded = pad_sequence(negatives, batch_first=True)

    return (
        queries_padded,
        query_lengths,
        positives_padded,
        positive_lengths,
        negatives_padded,
        negative_lengths,
    )


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
        for batch in val_dataloader:
            (
                query,
                query_lengths,
                positive,
                positive_lengths,
                negative,
                negative_lengths,
            ) = batch
            query = query.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            query_embedding = tower_one(query, query_lengths)
            positive_embedding = tower_two(positive, positive_lengths)
            negative_embedding = tower_two(negative, negative_lengths)

            loss = criterion(query_embedding, positive_embedding, negative_embedding)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def train(epochs: int = 10, batch_size: int = 128):
    print("Training...")
    training_data = pd.read_parquet(
        "data/train_triplets.parquet",
        columns=["query", "positive_passage", "negative_passage"],
    )
    dataset = TwoTowerDataset(training_data)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    print("Training data length: ", len(training_data))

    # Loading validation data
    validation_data = pd.read_parquet(
        "data/validation_triplets.parquet",
        columns=["query", "positive_passage", "negative_passage"],
    )
    val_dataset = TwoTowerDataset(validation_data)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    print("Validation data length: ", len(validation_data))

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

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            (
                query,
                query_lengths,
                positive,
                positive_lengths,
                negative,
                negative_lengths,
            ) = batch
            query = query.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            query_embedding = tower_one(query, query_lengths)
            positive_embedding = tower_two(positive, positive_lengths)
            negative_embedding = tower_two(negative, negative_lengths)

            loss = criterion(query_embedding, positive_embedding, negative_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item()})

        # Get validation loss
        val_loss = validate(tower_one, tower_two, val_dataloader, criterion)

        epoch_metrics = {
            "epoch": epoch + 1,
            "val/epoch_loss": val_loss,
            "train/epoch_loss": loss.item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        wandb.log(epoch_metrics)
        print(epoch_metrics)

        scheduler.step(val_loss)
        torch.save(tower_one.state_dict(), f"models/tower_one_{epoch}.pth")
        torch.save(tower_two.state_dict(), f"models/tower_two_{epoch}.pth")

    print("Saving final model to wandb...")
    torch.save(tower_one.state_dict(), "models/tower_one.pth")
    torch.save(tower_two.state_dict(), "models/tower_two.pth")
    wandb.save("models/tower_one.pth")
    wandb.save("models/tower_two.pth")

    wandb.finish()

    return tower_one, tower_two
