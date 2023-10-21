import numpy as np
import torch
from torch import nn


def neighboorPadding(input, mask, width):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    padded_input = input * mask
    conv = nn.Conv2d(1, 1, (3,3), padding=1).to(device)
    conv.weight = torch.nn.Parameter(torch.ones_like(conv.weight))
    conv.bias = None
    eps = 1e-10

    for i in range(width):
        num = conv(padded_input)
        den = conv(mask)
        e = (1 - mask) * (num / (den + eps))
        padded_input = padded_input + e
        mask = torch.min(torch.ones_like(mask), mask + den)

    return padded_input
