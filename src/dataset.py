import torch
import pandas as pd
from tokenizer import Tokenizer


class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer):
        self.data = pd.read_parquet(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, index
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        query_tensor = self.tokenizer.text_to_tensor(self.data["query"][index])
        pos_tensor = self.tokenizer.text_to_tensor(self.data["positive_passage"][index])
        neg_tensor = self.tokenizer.text_to_tensor(self.data["negative_passage"][index])

        return query_tensor, pos_tensor, neg_tensor
