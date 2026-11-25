"""Implement a causal temporal convolution network with a last-step logit head."""

import math

import torch.nn.functional as F
from torch import nn


class Conv(nn.Conv1d):
    """Convolution that pads only on the left, so step s never sees steps after s."""

    def forward(self, x):
        return super().forward(F.pad(x, ((self.kernel_size[0] - 1) * self.dilation[0], 0)))


class Block(nn.Module):
    """Two causal convolutions with ReLU and dropout, plus a residual connection."""

    def __init__(self, n_in, n_out, kernel_size, dilation, dropout, convolutions, residual):
        super().__init__()
        layers = []
        for i in range(convolutions):
            conv = Conv(n_in if i == 0 else n_out, n_out, kernel_size, dilation=dilation)
            layers += [conv, nn.ReLU(), nn.Dropout(dropout)]
        self.layers = nn.Sequential(*layers)
        self.skip = None
        if residual:
            self.skip = nn.Identity() if n_in == n_out else nn.Conv1d(n_in, n_out, 1)

    def forward(self, x):
        out = self.layers(x)
        return out if self.skip is None else out + self.skip(x)


class TCN(nn.Module):
    def __init__(self, architecture, prevalence):
        super().__init__()
        if not architecture["causal"]:
            raise ValueError("non-causal TCN needs approval; D6 fixes causal: true")
        k, dilations = architecture["kernel_size"], architecture["dilations"]
        convs = architecture["convolutions_per_block"]
        # receptive field 1 + convs * (k - 1) * sum(dilations) = 29 >= 15 for the default config
        assert 1 + convs * (k - 1) * sum(dilations) >= architecture["window"]
        channels = [architecture["input_size"]] + [architecture["channels"]] * len(dilations)
        self.blocks = nn.Sequential(
            *[
                Block(
                    channels[i],
                    channels[i + 1],
                    k,
                    d,
                    architecture["dropout"],
                    convs,
                    architecture["residual"],
                )
                for i, d in enumerate(dilations)
            ]
        )
        self.out = nn.Linear(architecture["channels"], architecture["output_size"])
        # start at the train log-odds so early epochs do not waste steps learning the base rate
        nn.init.constant_(self.out.bias, math.log(prevalence / (1 - prevalence)))

    def steps(self, x):
        return self.blocks(x.transpose(1, 2))  # [N, channels, window]; causal in the window axis

    def forward(self, x):
        return self.out(self.steps(x)[:, :, -1]).squeeze(-1)  # the last step is always a real lap
