import torch
import pandas as pd
import ast
from tokenizer import Tokenizer
from embeddings import SkipGramModel


class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path: str, tokenizer: Tokenizer, embedding_model: SkipGramModel
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = pd.read_csv(data_path)
        self.data["positive_passages"] = self.data["positive_passages"].apply(
            ast.literal_eval
        )
        self.data["negative_passages"] = self.data["negative_passages"].apply(
            ast.literal_eval
        )
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model.to(self.device)
        self.embedding_model.eval()

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, index
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        # Process query
        query_tensor = self.tokenizer.text_to_tensor(self.data["query"][index])

        # Process positive passages
        pos_tensors = [
            self.tokenizer.text_to_tensor(p)
            for p in self.data["positive_passages"][index]
        ]

        # Process negative passages
        neg_tensors = [
            self.tokenizer.text_to_tensor(p)
            for p in self.data["negative_passages"][index]
        ]

        # Move all tensors to device at once
        query_tensor = query_tensor.to(self.device)
        pos_tensors = [t.to(self.device) for t in pos_tensors]
        neg_tensors = [t.to(self.device) for t in neg_tensors]

        # Compute embeddings
        query_embedding = self.embedding_model.compute_embedding(query_tensor)
        pos_embeddings = [
            self.embedding_model.compute_embedding(t) for t in pos_tensors
        ]
        neg_embeddings = [
            self.embedding_model.compute_embedding(t) for t in neg_tensors
        ]

        # stack list of tensors to a single tensor
        pos_tensor = torch.stack(pos_embeddings)
        neg_tensor = torch.stack(neg_embeddings)

        return query_embedding, pos_tensor, neg_tensor
