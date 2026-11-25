"""Build 15-lap windows ending at lap t, left-padded with zeros after preprocessing."""

import numpy as np
import pandas as pd

from pitstop.contracts import FEATURES, KEY, SEQ_CHANNELS, WINDOW


def build_windows(features, samples, width=WINDOW):
    """Return float32 [N, width, 16] windows t-width+1..t of each sample's race and driver.

    `features` holds preprocessed KEY + FEATURES for every real lap, so history laps
    that are not sample rows still supply context. A window stops at the first missing
    lap (consecutive laps only); the rest is zero features with is_pad = 1.
    """
    values = features[list(FEATURES)].to_numpy("float32")
    index = pd.MultiIndex.from_frame(features[list(KEY)])
    windows = np.zeros((len(samples), width, len(SEQ_CHANNELS)), dtype="float32")
    windows[:, :, -1] = 1  # is_pad
    real = np.ones(len(samples), dtype=bool)
    for back in range(width):
        wanted = samples[list(KEY)].assign(lap_number=samples["lap_number"] - back)
        rows = index.get_indexer(pd.MultiIndex.from_frame(wanted))
        real &= rows >= 0
        windows[real, width - 1 - back, :-1] = values[rows[real]]
        windows[real, width - 1 - back, -1] = 0
    return windows
