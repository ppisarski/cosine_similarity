"""

"""
import numpy as np
import torch


def cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    https://medium.com/@dhruvbird/all-pairs-cosine-similarity-in-pytorch-867e722c8572
    """
    x_row, x_col = x[None, :, :], x[:, None, :]
    x_row, x_col = x_row.expand(x.shape[0], x.shape[0], x.shape[1]), x_col.expand(x.shape[0], x.shape[0], x.shape[1])
    return torch.nn.functional.cosine_similarity(x_row, x_col, dim=-1)


def cosine_similarity_numpy(x: np.array) -> np.array:
    """
    https://medium.com/@dhruvbird/all-pairs-cosine-similarity-in-pytorch-867e722c8572
    """
    x = torch.Tensor(x)
    similarity = cosine_similarity(x)
    return similarity.numpy()


def main():
    x = torch.Tensor([[0, 0.1, 0.2], [1.1, 1.2, 1]])
    similarity = cosine_similarity(x)
    print(similarity)


if __name__ == "__main__":
    main()
