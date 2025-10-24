from collections import namedtuple

import torch
from torch import nn
from torch.nn import Module

# constants

Statistics = namedtuple('Statistics', [
    'mean',
    'variance',
    'gamma',
    'beta'
])


# reversible instance normalization
# proposed in https://openreview.net/forum?id=cGDAkQo1C0p

class RevIN(Module):
    def __init__(self, num_variates, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1))
        self.beta = nn.Parameter(torch.zeros(num_variates, 1))

    def forward(self, x, return_statistics=False):
        assert x.shape[1] == self.num_variates

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var_rsqrt = var.clamp(min=self.eps).rsqrt()
        instance_normalized = (x - mean) * var_rsqrt
        rescaled = instance_normalized * self.gamma + self.beta

        def reverse_fn(scaled_output):
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min=self.eps)
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            return unscaled_output * var.sqrt() + mean

        if not return_statistics:
            return rescaled, reverse_fn

        statistics = Statistics(mean, var, self.gamma, self.beta)

        return rescaled, reverse_fn, statistics


# sanity check

if __name__ == '__main__':
    rev_in = RevIN(512)

    x = torch.randn(2, 512, 1024)

    normalized, reverse_fn, statistics = rev_in(x, return_statistics=True)

    out = reverse_fn(normalized)

    assert torch.allclose(x, out)
