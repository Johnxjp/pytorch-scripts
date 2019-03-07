import torch

from src.transforms import normalise


def cosine(x, y):
    x, y = normalise(x), normalise(y)
    return torch.matmul(x, y) + 1.0


def euclidean(x, y):
    return torch.sqrt((x - y) * (x - y))
