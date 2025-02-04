import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from dataset import TwoTowerDataset
from tokenizer import Tokenizer
from embeddings import SkipGramModel
from two_tower import TowerOne, TowerTwo
from tqdm import tqdm
import wandb
from config import DEVICE


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Separate queries, positives, and negatives
    queries, positives, negatives = zip(*batch)

    # Pad query sequences
    lengths = torch.tensor([len(q) for q in queries], dtype=torch.int64, device="cpu")
    queries_padded = rnn_utils.pad_sequence(queries, batch_first=True, padding_value=0)
    query_out = (queries_padded, lengths)

    # Flatten and pad positive sequences
    lengths = torch.tensor([len(p) for p in positives], dtype=torch.int64, device="cpu")
    pos_padded = rnn_utils.pad_sequence(positives, batch_first=True, padding_value=0)
    positives_out = (pos_padded, lengths)

    # Flatten and pad negative sequences
    lengths = torch.tensor([len(n) for n in negatives], dtype=torch.int64, device="cpu")
    neg_padded = rnn_utils.pad_sequence(negatives, batch_first=True, padding_value=0)
    negatives_out = (neg_padded, lengths)

    return query_out, positives_out, negatives_out


def triplet_loss(
    query_embedding: torch.Tensor,
    pos_embedding: torch.Tensor,
    neg_embedding: torch.Tensor,
    margin=0.3,
):
    """
    Compute triplet loss for a batch of embeddings.
    Uses cosine similarity and returns mean loss over the batch.
    """
    pos_distance = 1 - torch.nn.functional.cosine_similarity(
        query_embedding, pos_embedding
    )
    neg_distance = 1 - torch.nn.functional.cosine_similarity(
        query_embedding, neg_embedding
    )
    # Use clamp to get max(0, x) for each element in the batch
    losses = torch.clamp(pos_distance - neg_distance + margin, min=0)
    # Return mean loss over the batch
    return losses.mean()


def train(
    tokenizer: Tokenizer,
    embedding_model: SkipGramModel,
    epochs=10,
    save_model=True,
    model_path="models/two_tower_model.pt",
):
    print("training on device: ", DEVICE)

    vocab_size = len(tokenizer.word2idx)

    dataset = TwoTowerDataset(
        data_path="data/train_triplets.parquet",
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset, batch_size=10240, shuffle=True, collate_fn=collate_fn
    )

    tower_one = TowerOne(
        embedding_matrix=embedding_model.embedding.weight,
        vocab_size=vocab_size,
        embedding_dim=258,
        hidden_dimension=258,
    ).to(DEVICE)

    tower_two = TowerTwo(
        embedding_matrix=embedding_model.embedding.weight,
        vocab_size=vocab_size,
        embedding_dim=258,
        hidden_dimension=258,
    ).to(DEVICE)

    # Single optimizer for both towers
    optimizer = torch.optim.Adam(
        list(tower_one.parameters()) + list(tower_two.parameters()), lr=0.001
    )

    print("Starting training...")
    wandb.init(project="mlx6-two-tower", name="two-tower-model")
    for epoch in range(epochs):
        for qry, pos, neg in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            qry, qry_lengths = qry
            pos, pos_lengths = pos
            neg, neg_lengths = neg

            qry = qry.to(DEVICE)
            pos = pos.to(DEVICE)
            neg = neg.to(DEVICE)

            query_embedding = tower_one(qry, qry_lengths)
            pos_embedding = tower_two(pos, pos_lengths)
            neg_embedding = tower_two(neg, neg_lengths)

            loss = triplet_loss(query_embedding, pos_embedding, neg_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

        print(f"Epoch {epoch} loss: {loss.item()}")

    if save_model:
        torch.save(tower_one.state_dict(), model_path)
