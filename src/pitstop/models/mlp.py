"""Implement the 15-to-64-to-32-to-1 ReLU MLP returning one logit per sample."""

import math

from torch import nn


class MLP(nn.Module):
    def __init__(self, architecture, prevalence):
        super().__init__()
        sizes = [architecture["input_size"], *architecture["hidden_sizes"]]
        layers = []
        for n_in, n_out in zip(sizes, sizes[1:], strict=False):
            layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(architecture["dropout"])]
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(sizes[-1], architecture["output_size"])
        # start at the train log-odds so early epochs do not waste steps learning the base rate
        nn.init.constant_(self.out.bias, math.log(prevalence / (1 - prevalence)))

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)
