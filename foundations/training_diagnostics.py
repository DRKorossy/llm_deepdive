import torch
import torch.nn as nn
from typing import List, Dict
# The solutions at the time of me submitting this have a poor explaination,
# so I will leave some detailed comments for _compute_dead_fractions
# hoping it might be useful.

class Solution:

    @torch.no_grad()
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        for module in model.children():
            x = module(x)
            # ignore nonlinear layers
            if not isinstance(module, nn.Linear):
                continue
            mu = round(x.mean().item(), 4)
            sigma = round(x.std().item(), 4)
            # please see function definition below.
            dead_fraction = self._compute_dead_fractions(x)

            stat = {"mean": mu, "std": sigma, "dead_fraction": dead_fraction}
            stats.append(stat)
        return stats
    
    def _compute_dead_fractions(self, x: torch.Tensor) -> float:
        """
        Take x: the activation tensor, output fraction of dead neurons.
        """
        # compute a tensor, like x, with boolean values element-wise: 
        # take values True when the element is <= 0, False when > 0.
        # 
        #           [[ 0.2, -0.4,  2.4],    non-positive mask.   [[False, True, False],
        # example:   [-0.9, -0.2,  0.1],  --------------------->  [ True, True, False],
        #            [ 0.1,  0.0, -1.2]]                          [False, True,  True]]
        non_positive = x <= 0
        if x.dim() >= 2:
            # Check if along each column, all values are True. This collapses the 0-th dim.
            # 
            #          [[False, True, False],    check all True in col
            # example:  [ True, True, False],  --------------------------> [[False, True, False]]
            #           [False, True,  True]]
            dead_neurons = non_positive.all(dim=0) 
            # turn the booleans into something that we can compute the mean of - e.g. float32
            # 
            # e.g. [[False, True, False]] --> [[0.0, 1.0, 0.0]]
            dead_neurons_float = dead_neurons.float()
            # now simply compute the mean
            dead_fraction_tensor = dead_neurons_float.mean()
        else:
            # if x is < 2 dimensional, it's either a row vector or a scalar tensor.
            # so we couldn't collapse along dimension 0.
            # in this case we can simply compute the mean directly
            # 
            # example: [ 0.2, -0.4,  2.4] --> [False, True, False] --> [0, 1, 0] --> 1/3
            non_positive = x <= 0
            dead_fraction_tensor = non_positive.float().mean()
        dead_fraction = dead_fraction_tensor.item()
        return round(dead_fraction, 4)
        

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        # forward pass
        y_pred = model(x)
        loss_function = nn.MSELoss()
        loss = loss_function(y, y_pred)
        # backward - i.e. calculate gradients of loss wrt each parameter.
        loss.backward()
        stats = []
        for module in model.children():
            if not isinstance(module, nn.Linear):
                continue
            grad = module.weight.grad
            mu = round(grad.mean().item(), 4)
            sigma = round(grad.std().item(), 4)
            norm = round(torch.norm(grad).item(), 4)
            stat = {"mean": mu, "std": sigma, "norm": norm}
            stats.append(stat)
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for a, g in zip(activation_stats, gradient_stats):
            if a["dead_fraction"] > 0.5:
                return "dead_neurons"
            if g["norm"] > 1000:
                return "exploding_gradients"
            if g["norm"] < 1e-5:
                return "vanishing_gradients"
            if a["std"] < 0.1:
                return "vanishing_gradients"
            if a["std"] >10.0:
                return "exploding_gradients"
        
        return "healthy"