"""Implement a two-layer GRU with an LSTM option and a last-step logit head."""

import math

from torch import nn

CELLS = {"gru": nn.GRU, "lstm": nn.LSTM}


class RNN(nn.Module):
    def __init__(self, architecture, prevalence):
        super().__init__()
        if architecture["bidirectional"]:
            raise ValueError("bidirectional needs approval; D6 fixes a one-directional model")
        self.rnn = CELLS[architecture["cell"]](
            architecture["input_size"],
            architecture["hidden_size"],
            architecture["num_layers"],
            batch_first=True,
            dropout=architecture["dropout"],
        )
        self.out = nn.Linear(architecture["hidden_size"], architecture["output_size"])
        # start at the train log-odds so early epochs do not waste steps learning the base rate
        nn.init.constant_(self.out.bias, math.log(prevalence / (1 - prevalence)))

    def forward(self, x):
        states, _ = self.rnn(x)  # [N, window, hidden]
        return self.out(states[:, -1]).squeeze(-1)  # the last step is always a real lap
