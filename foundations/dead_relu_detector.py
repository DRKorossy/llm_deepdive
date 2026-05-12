import torch
import torch.nn as nn
from typing import List


class Solution:
    @torch.no_grad()
    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        for module in model.children():
            x = module(x)
            if not isinstance(module, nn.ReLU):
                continue
            dead = (x == 0).all(dim=0).float().mean().item()
            dead = round(dead, 4)
            dead_fractions.append(dead)
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if any(dead_fraction > 0.5 for dead_fraction in dead_fractions):
            return "use_leaky_relu"
        if dead_fractions[0] > 0.3:
            return "reinitialize"
        strictly_increasing = all(x < y for x, y in zip(dead_fractions, dead_fractions[1:]))
        if strictly_increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"
        if max(dead_fractions) < 0.1:
            return "healthy"
        return "healthy"
