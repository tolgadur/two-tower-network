import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
