import torch.nn as nn

from ..utils import pairwise


def create_decode(sizes, out_points, batch_norm=True):
    layers = []
    for in_size, out_size in pairwise(sizes):
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.PReLU(out_size))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_size))
    layers.append(nn.Linear(out_size, out_points*3))
    return nn.Sequential(*layers)
