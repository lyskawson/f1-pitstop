"""Lap and asymmetric event metrics with race-level bootstrap uncertainty."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pitstop.contracts import KEY


def select_threshold(y, scores):
    """Return (threshold, F1) for validation labels and scores only."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (
        y.ndim != 1
        or y.shape != scores.shape
        or not len(y)
        or not np.isin(y, (0, 1)).all()
        or not ((scores >= 0) & (scores <= 1)).all()
    ):
        raise ValueError("Threshold selection needs nonempty binary labels and probabilities")

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    # Only a complete score tie can be admitted by an inclusive threshold.
    ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    f1 = 2 * y[order].cumsum()[ends] / (ends + 1 + y.sum())
    best = np.argmax(f1)
    # At 1 with zero alarms, F1 is zero for a positive group and undefined
    # otherwise. Neither can displace the best observed threshold.
    return float(ordered_scores[ends[best]]), float(f1[best])


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _classification(hits, correct, positives, alarms):
    recall = _ratio(hits, positives)
    precision = _ratio(correct, alarms)
    return {
        "recall": recall,
        "precision": precision,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else (0.0 if positives or alarms else None)
        ),
    }


class _RaceMetrics:
    """Cache event matches once, before races can appear more than once."""

    def __init__(self, frame, threshold):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be a probability in [0, 1]")
        self.y = frame["y"].to_numpy(dtype=bool)
        self.scores = frame["score"].to_numpy()
        self.alarms = self.scores >= threshold
        self.races = list(frame.groupby("race_id", sort=True, observed=True).indices.values())
        keys = pd.MultiIndex.from_frame(frame[list(KEY)])
        laps = frame["lap_number"].to_numpy()
        race_driver = [frame[name] for name in KEY[:2]]
        hit = correct = self.y & self.alarms
        self.events = {}
        for k in (1, 2):
            # Lookup each possible earlier alarm by its actual lap key.
            earlier = keys.get_indexer(pd.MultiIndex.from_arrays([*race_driver, laps - k]))
            targets = np.flatnonzero(self.y & (earlier >= 0))
            targets = targets[self.alarms[earlier[targets]]]
            hit, correct = hit.copy(), correct.copy()
            hit[targets] = True
            correct[earlier[targets]] = True
            self.events[str(k)] = hit, correct

    def values(self, rows, n_races):
        y, alarms = self.y[rows], self.alarms[rows]
        positives, n_alarms = y.sum(), alarms.sum()
        tp = (y & alarms).sum()
        values = {
            "prevalence": _ratio(positives, len(y)),
            "average_precision": (
                float(average_precision_score(y, self.scores[rows])) if positives else None
            ),
            **_classification(tp, tp, positives, n_alarms),
        }
        for k, (hit, correct) in self.events.items():
            hits, correct_alarms = hit[rows].sum(), correct[rows].sum()
            event = _classification(hits, correct_alarms, positives, n_alarms)
            event["false_alarms_per_race"] = _ratio(n_alarms - correct_alarms, n_races)
            values.update({f"events.{k}.{name}": value for name, value in event.items()})
        return values

    def draw(self, race_indices):
        return self.values(np.concatenate([self.races[i] for i in race_indices]), len(race_indices))


def _intervals(point, draws):
    defined = {name: [] for name in point}
    for draw in draws:
        for name, value in draw.items():
            if value is not None:
                defined[name].append(value)
    intervals, missing = {}, {}
    for name, values in defined.items():
        bounds = np.percentile(values, [2.5, 97.5]).tolist() if values else [None, None]
        intervals[name] = dict(zip(("lower", "upper"), bounds, strict=True))
        missing[name] = len(draws) - len(values)
    return intervals, missing


def _race_draws(n_races, bootstrap_samples, seed):
    for name, value, minimum in (("bootstrap_samples", bootstrap_samples, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    return np.random.default_rng(seed).integers(n_races, size=(bootstrap_samples, n_races))


def _undefined(values, intervals, n_rows):
    reasons = {}
    for name, value in values.items():
        metric = name.rsplit(".", 1)[-1]
        if value is None:
            reasons[name] = (
                "empty_group"
                if not n_rows
                else {
                    "average_precision": "no_positives",
                    "recall": "no_positives",
                    "precision": "no_alarms",
                    "f1": "no_positives_or_alarms",
                }[metric]
            )
        if intervals[name]["lower"] is None:
            reasons[f"confidence_intervals.{name}"] = (
                "empty_group" if not n_rows else "no_defined_bootstrap_replicates"
            )
    return reasons


def group_metrics(frame, threshold, *, bootstrap_samples=1000, seed=42):
    """Evaluate aligned prediction rows from one group; return metrics and undefined counts."""
    metrics = _RaceMetrics(frame, threshold)
    n_races = len(metrics.races)
    point = metrics.values(np.arange(len(frame)), n_races)
    races = _race_draws(n_races, bootstrap_samples, seed)
    draws = [metrics.draw(draw) for draw in races] if n_races else [point] * bootstrap_samples
    intervals, missing = _intervals(point, draws)
    events = {
        k: {
            name: point[f"events.{k}.{name}"]
            for name in ("recall", "precision", "f1", "false_alarms_per_race")
        }
        for k in ("1", "2")
    }
    return {
        "n_samples": len(frame),
        "n_positives": int(metrics.y.sum()),
        "n_races": n_races,
        "threshold": float(threshold),
        **{name: value for name, value in point.items() if not name.startswith("events.")},
        "events": events,
        "confidence_intervals": intervals,
        "undefined": _undefined(point, intervals, len(frame)),
    }, missing


def paired_bootstrap(
    left, right, left_threshold, right_threshold, *, bootstrap_samples=1000, seed=42
):
    """Compare left minus right on identical keys and labels from one split and group."""
    columns = [*KEY, "split", "y"]
    left = left.sort_values(list(KEY), ignore_index=True)
    right = right.sort_values(list(KEY), ignore_index=True)
    if not left[columns].equals(right[columns]) or left["split"].nunique() > 1:
        raise ValueError("Paired comparisons require identical keys and labels from one split")
    a, b = _RaceMetrics(left, left_threshold), _RaceMetrics(right, right_threshold)
    n_races = len(a.races)
    rows = np.arange(len(left))

    def difference(first, second):
        return {
            name: None if value is None or second[name] is None else value - second[name]
            for name, value in first.items()
        }

    point = difference(a.values(rows, n_races), b.values(rows, n_races))
    races = _race_draws(n_races, bootstrap_samples, seed)
    draws = (
        [difference(a.draw(draw), b.draw(draw)) for draw in races]
        if n_races
        else [point] * bootstrap_samples
    )
    intervals, missing = _intervals(point, draws)
    return {
        "differences": point,
        "confidence_intervals": intervals,
        "ties": {
            name: None if bounds["lower"] is None else bounds["lower"] <= 0 <= bounds["upper"]
            for name, bounds in intervals.items()
        },
        "undefined": _undefined(point, intervals, len(left)),
        "bootstrap": {
            "unit": "race",
            "samples": int(bootstrap_samples),
            "confidence_level": 0.95,
            "method": "percentile",
            "seed": int(seed),
            "undefined_replicates": missing,
            "undefined_handling": "omit_per_metric",
        },
    }
