import torch


def softmax(x: torch.Tensor, index: int =-1) -> torch.Tensor:
    # do softmax over dimension index of x
    x_exp = torch.exp(x - torch.max(x, dim=index, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=index, keepdim=True)