import torch
from tqdm import tqdm
from two_tower import TowerOne, TowerTwo
import faiss
from dataset import TwoTowerDataset
from config import DEVICE
import pandas as pd
import os


class Inference:
    def __init__(
        self, tower_one: TowerOne, tower_two: TowerTwo, dataset: TwoTowerDataset
    ):
        self.tower_one = tower_one
        self.tower_two = tower_two

        # Put models in eval mode since we're only doing inference
        self.tower_one.eval()
        self.tower_two.eval()

        self.index = faiss.IndexFlatIP(256 // 2)
        self.dataset = dataset
        self.docs = None

    def save_index(self, path: str = "data/faiss.index"):
        """Save the FAISS index and document list to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        # Save documents alongside the index
        pd.Series(self.docs).to_parquet(path + ".docs.parquet")

    def load_index(self, path: str = "data/faiss.index") -> bool:
        """Load FAISS index and document list from disk. Returns True if successful."""
        if not os.path.exists(path):
            return False

        docs_path = path + ".docs.parquet"
        if not os.path.exists(docs_path):
            return False

        self.index = faiss.read_index(path)
        self.docs = pd.read_parquet(docs_path).values
        return True

    def embed_docs(
        self,
        filename: str = "data/unique_documents.parquet",
        batch_size: int = 512,
        save_index: bool = True,
    ):
        # Try to load existing index first
        if self.load_index():
            print("Loaded existing FAISS index")
            return

        print("Building new FAISS index...")
        df = pd.read_parquet(filename)
        self.docs = df["document"].values

        # Process and add to index in batches
        for i in tqdm(range(0, len(self.docs), batch_size)):
            batch = self.docs[i : i + batch_size]
            with torch.no_grad():
                embeddings = self.dataset._text_to_embeddings_batch(batch).to(DEVICE)
                encodings = self.tower_two(embeddings).cpu().numpy()
                faiss.normalize_L2(encodings)
                self.index.add(encodings)

        if save_index:
            self.save_index()

    def find_nearest_neighbors(self, query: str, k: int = 5):
        with torch.no_grad():
            query_embedding = self.dataset._text_to_embeddings(query).to(DEVICE)
            query_encoding = self.tower_one(query_embedding).cpu().numpy()

            # Reshape to 2D array (1, dim) as FAISS expects
            query_encoding = query_encoding.reshape(1, -1)

        faiss.normalize_L2(query_encoding)
        similarities, indices = self.index.search(query_encoding, k)

        return self.docs[indices.flatten()], similarities.flatten()
