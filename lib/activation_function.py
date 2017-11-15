import torch


def squash(x, dim):
    square_x = torch.sum(x ** 2, dim=dim, keepdim=True)
    sqrt_x = torch.sqrt(square_x)
    output = x * square_x / (1 + square_x) / sqrt_x
    return output
