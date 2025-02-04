import torch
from torch.utils.data import DataLoader
from dataset import TwoTowerDataset
from tokenizer import Tokenizer
from embeddings import SkipGramModel
from two_tower import TwoTowerModel
from tqdm import tqdm
import wandb


def triplet_loss(
    query_embedding: torch.Tensor,
    pos_embedding: torch.Tensor,
    neg_embedding: torch.Tensor,
    margin=1.0,
):
    pos_similarity = torch.nn.functional.cosine_similarity(
        query_embedding, pos_embedding
    )
    neg_similarity = torch.nn.functional.cosine_similarity(
        query_embedding, neg_embedding
    )
    return max(0, pos_similarity - neg_similarity + margin)


def train(
    tokenizer: Tokenizer,
    embedding_model: SkipGramModel,
    epochs=10,
    save_model=True,
    model_path="models/two_tower_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(tokenizer.word2idx)

    dataset = TwoTowerDataset(
        data_path="data/train_triplets.csv",
        tokenizer=tokenizer,
        embedding_model=embedding_model,
    )
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = TwoTowerModel(
        embedding_matrix=embedding_model.embedding.weight,
        vocab_size=vocab_size,
        embedding_dim=258,
        hidden_dimension=258,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    wandb.init(project="mlx6-two-tower", name="two-tower-model")
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            query_embedding, pos_embedding, neg_embedding = model(batch)
            loss = triplet_loss(query_embedding, pos_embedding, neg_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

        print(f"Epoch {epoch} loss: {loss.item()}")

    if save_model:
        torch.save(model.state_dict(), model_path)
