import torch
import torch.nn as nn
import math
from typing import List


class Solution:
    def random_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        weights = torch.round(torch.randn(fan_out, fan_in), decimals=4)
        return weights.tolist()

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        stdev = math.sqrt(2/(fan_in + fan_out))
        weights = torch.round(torch.randn(fan_out, fan_in) * stdev, decimals=4)
        return weights.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        stdev = math.sqrt(2/fan_in)
        weights = torch.round(torch.randn(fan_out, fan_in) * stdev, decimals=4)
        return weights.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        stdevs = []
        weights = []
        for i in range(num_layers):
            fan_out = hidden_dim
            fan_in = fan_out
            if i == 0:
                fan_in = input_dim
            if init_type == "xavier":
                stdev = math.sqrt(2/(fan_in + fan_out))
            elif init_type == "kaiming":
                stdev = math.sqrt(2/(fan_in))
            else:
                stdev = 1.0
            W = torch.randn(fan_out, fan_in) * stdev
            weights.append(W)


        x = torch.randn(1, input_dim)
        h = x
        for W in weights:
            h = torch.relu(h @ W.T)
            stdevs.append(round(torch.std(h).item(), 2))
        return stdevs
