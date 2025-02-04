import torch
import more_itertools
import tqdm
import wandb
from config import DEVICE


class SkipGramModel(torch.nn.Module):
    def __init__(self, voc, emb):
        """Initializes the CBOW word2Vec model

        Args:
            voc: number of tokens in the vocabularly
            emb: what dimension we want our embeddings to be in
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.linear = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.to(DEVICE)

    def forward(self, inpt, trgs, rand):
        """One forward pass

        Args:
            inpt: index of input token
            trgs: indices of target tokens
            rand: indices of negative samples

        Returns:
            _output of the loss function
        """
        emb = self.embedding(inpt)
        ctx = self.linear.weight[trgs]
        rnd = self.linear.weight[rand]

        out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
        rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()

        # compute loss
        out = self.sigmoid(out)
        rnd = self.sigmoid(rnd)
        pst = -out.log().mean()
        ngt = -(1 - rnd + 10 ** (-3)).log().mean()
        return pst + ngt

    def compute_embedding(self, word_tensor: torch.Tensor):
        """Compute embedding for a given word tensor."""
        with torch.no_grad():
            return self.embedding(word_tensor)


class EmbeddingTrainer:
    def __init__(self, model, tokens, batch_size=512, learning_rate=0.003):
        self.model = model
        self.tokens = tokens
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # initialize dataset
        windows = list(more_itertools.windowed(tokens, 5))  # window size is 3

        print(f"Number of windows before filtering: {len(windows)}")

        windows = [w for w in windows if None not in w]  # filter out incomplete windows

        print(f"Number of windows after filtering: {len(windows)}")
        if isinstance(model, SkipGramModel):
            self.inputs = [w[2] for w in windows]  # single center word
            self.targets = [[w[0], w[1], w[3], w[4]] for w in windows]
        else:
            self.inputs = [[w[0], w[2], w[3], w[4]] for w in windows]
            self.targets = [w[1] for w in windows]  # single center word

        print(f"Number of input samples: {len(self.inputs)}")
        print(f"Number of target samples: {len(self.targets)}")

        self.dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(self.inputs), torch.LongTensor(self.targets)
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )

    def train(
        self,
        vocab_size,
        num_epochs=10,
        save_model=False,
        model_path="models/skipgram_model.pt",
    ):
        """Train the model.

        Args:
            vocab_size (int): Size of vocabulary for negative sampling
            num_epochs (int, optional): Number of epochs to train.
            save_model (bool, optional): Whether to save model weights.
            model_path (str, optional): Where to save model weights.
        """
        print("Starting training...")
        wandb.init(project="mlx6-word2vec", name="skipgram-model")
        self.model.to(DEVICE)
        for epoch in range(num_epochs):
            prgs = tqdm.tqdm(self.dataloader, desc=f"Epoch {epoch + 1}", leave=False)
            for inpt, trgs in prgs:
                inpt, trgs = inpt.to(DEVICE), trgs.to(DEVICE)
                rand = torch.randint(0, vocab_size, (inpt.size(0), 2)).to(DEVICE)

                self.optimizer.zero_grad()
                loss: torch.Tensor = self.model(inpt, trgs, rand)
                loss.backward()
                self.optimizer.step()
                wandb.log({"loss": loss.item()})
            print("Loss for epoch", epoch + 1, ":", loss.item())

        if save_model:
            self._save_model(model_path)

    def _save_model(self, model_path):
        print("Saving...")
        torch.save(self.model.state_dict(), model_path)

        print("Uploading...")
        artifact = wandb.Artifact("model-weights-cbow", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        print("Done!")
        wandb.finish()
