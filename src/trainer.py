import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from dataset import TwoTowerDataset
from two_tower import TowerOne, TowerTwo
from tqdm import tqdm
import wandb
from config import DEVICE


def collate_fn(batch):
    """Custom collate function to handle variable length sequences of embeddings.

    Args:
        batch: List of tuples (query_emb, pos_emb, neg_emb) where each *_emb is
              itself a tuple of (embeddings, length)

    Returns:
        tuple of:
            - (padded_queries, query_lengths)
            - (padded_positives, positive_lengths)
            - (padded_negatives, negative_lengths)
        where each padded_* tensor has shape (batch_size, max_seq_len, embedding_dim)
    """
    # Unzip the batch into separate lists
    queries, positives, negatives = zip(*batch)

    # Further unzip each component into embeddings and lengths
    query_embs, query_lengths = zip(*queries)
    pos_embs, pos_lengths = zip(*positives)
    neg_embs, neg_lengths = zip(*negatives)

    # Convert lengths to tensors
    query_lengths = torch.tensor(query_lengths, dtype=torch.long)
    pos_lengths = torch.tensor(pos_lengths, dtype=torch.long)
    neg_lengths = torch.tensor(neg_lengths, dtype=torch.long)

    # Pad sequences
    padded_queries = rnn_utils.pad_sequence(query_embs, batch_first=True)
    padded_positives = rnn_utils.pad_sequence(pos_embs, batch_first=True)
    padded_negatives = rnn_utils.pad_sequence(neg_embs, batch_first=True)

    return (
        (padded_queries, query_lengths),
        (padded_positives, pos_lengths),
        (padded_negatives, neg_lengths),
    )


def triplet_loss(
    query_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
    margin: float = 0.7,
) -> torch.Tensor:
    """Compute triplet loss between query, positive, and negative embeddings."""
    # Compute cosine similarities
    pos_sim = torch.nn.functional.cosine_similarity(query_embedding, positive_embedding)
    neg_sim = torch.nn.functional.cosine_similarity(query_embedding, negative_embedding)

    loss = torch.clamp(pos_sim - neg_sim + margin, min=0)
    return loss.mean()


def train(epochs=10, batch_size=512, save_model=True):
    """Train the two-tower model.

    Args:
        epochs: Number of epochs to train
        save_model: Whether to save model checkpoints
    """
    print("Training on device:", DEVICE)

    # Initialize datasets and dataloaders
    train_dataset = TwoTowerDataset(data_path="data/train_triplets.parquet")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_dataset = TwoTowerDataset(data_path="data/validation_triplets.parquet")
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Initialize models
    tower_one = TowerOne().to(DEVICE)
    tower_two = TowerTwo().to(DEVICE)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        list(tower_one.parameters()) + list(tower_two.parameters()), lr=0.001
    )

    print("Starting training...")
    wandb.init(project="mlx6-two-tower", name="two-tower-model")
    for epoch in range(epochs):
        # Training
        tower_one.train()
        tower_two.train()

        for (qry, qry_lengths), (pos, pos_lengths), (neg, neg_lengths) in tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}"
        ):
            # Move data to device
            qry = qry.to(DEVICE)
            pos = pos.to(DEVICE)
            neg = neg.to(DEVICE)

            # Forward pass
            query_embedding = tower_one(qry, qry_lengths)
            pos_embedding = tower_two(pos, pos_lengths)
            neg_embedding = tower_two(neg, neg_lengths)

            # Compute loss
            loss = triplet_loss(query_embedding, pos_embedding, neg_embedding)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item()})

        # Validation
        val_loss = validate(tower_one, tower_two, val_dataloader)
        print(
            f"Epoch {epoch + 1} - "
            f"Train loss: {loss.item():.4f}, "
            f"Val loss: {val_loss:.4f}"
        )
        wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "epoch": epoch + 1})

    if save_model:
        torch.save(tower_one.state_dict(), "models/tower_one.pt")
        torch.save(tower_two.state_dict(), "models/tower_two.pt")
        # Save models to wandb
        wandb.save("models/tower_one.pt")
        wandb.save("models/tower_two.pt")

    wandb.finish()


def validate(
    tower_one: TowerOne, tower_two: TowerTwo, val_dataloader: DataLoader
) -> float:
    """Run validation and compute average loss.

    Args:
        tower_one: Query tower model
        tower_two: Document tower model
        val_dataloader: Validation data loader

    Returns:
        Average validation loss
    """
    tower_one.eval()
    tower_two.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for (
            (qry, qry_lengths),
            (pos, pos_lengths),
            (neg, neg_lengths),
        ) in val_dataloader:
            # Move data to device
            qry = qry.to(DEVICE)
            pos = pos.to(DEVICE)
            neg = neg.to(DEVICE)

            # Forward pass
            query_embedding = tower_one(qry, qry_lengths)
            pos_embedding = tower_two(pos, pos_lengths)
            neg_embedding = tower_two(neg, neg_lengths)

            # Compute loss
            loss = triplet_loss(query_embedding, pos_embedding, neg_embedding)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches
