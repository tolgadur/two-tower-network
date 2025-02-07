import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings, positive_embeddings, negative_embeddings):
        """
        Compute InfoNCE loss for two-tower network

        Args:
            query_embeddings: Tensor of shape [batch_size, embedding_dim]
            positive_embeddings: Tensor of shape [batch_size, embedding_dim]
            negative_embeddings: Tensor of shape [batch_size, embedding_dim]

        Returns:
            loss: InfoNCE loss value
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, dim=1)

        # Compute logits
        pos_logits = (
            torch.sum(query_embeddings * positive_embeddings, dim=1) / self.temperature
        )
        neg_logits = (
            torch.sum(query_embeddings * negative_embeddings, dim=1) / self.temperature
        )

        # Concatenate positive and negative logits
        logits = torch.stack([pos_logits, neg_logits], dim=1)

        # Labels are always 0 since positive is always the first element
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
