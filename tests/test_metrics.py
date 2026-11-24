import pandas as pd
import pytest

from pitstop.contracts import KEY
from pitstop.metrics import group_metrics, paired_bootstrap, select_threshold

R1 = "2025_01_R"
R2 = "2025_02_R"


def frame(rows):
    return pd.DataFrame(rows, columns=[*KEY, "y", "score"]).assign(split="val")


@pytest.mark.parametrize(
    ("y", "scores", "expected"),
    [
        ([1, 0, 0, 1], [0.9, 0.8, 0.6, 0.3], (0.9, 2 / 3)),
        ([0, 1, 1], [1, 1, 0], (0, 4 / 5)),
        ([1, 0], [1, 0], (1, 1)),
        ([0, 1], [1, 1], (1, 2 / 3)),
        ([0, 0], [0, 0.5], (0.5, 0)),
    ],
)
def test_threshold_inclusive_ties_and_boundaries(y, scores, expected):
    assert select_threshold(y, scores) == pytest.approx(expected)


@pytest.mark.parametrize(("y", "scores"), [([], []), ([1], [float("nan")]), ([2], [0.5])])
def test_threshold_requires_labels_and_probabilities(y, scores):
    with pytest.raises(ValueError, match="nonempty binary labels and probabilities"):
        select_threshold(y, scores)


def test_asymmetric_windows_with_lap_gaps_and_independent_denominators():
    rows = frame(
        [
            (R1, "44", 1, 0, 0.9),  # Correct only for k=2.
            (R1, "44", 2, 0, 0.9),  # Early alarm.
            (R1, "44", 3, 1, 0.9),  # On time, also covers the next target.
            (R1, "44", 4, 1, 0.1),
            (R1, "44", 5, 0, 0.9),  # Late for lap 4 and too early for lap 8.
            (R1, "44", 8, 1, 0.1),
            (R1, "44", 11, 0, 0.9),  # Adjacent row, three actual laps away.
            (R1, "44", 14, 1, 0.1),
            (R1, "16", 7, 0, 0.9),  # Wrong driver for the lap 8 target.
            (R2, "44", 7, 0, 0.9),  # Wrong race for the lap 8 target.
        ]
    )
    result, _ = group_metrics(rows.iloc[::-1], 0.9, bootstrap_samples=6, seed=0)
    assert result["n_samples"] == 10
    assert result["n_races"] == 2
    assert result["n_positives"] == 4
    assert result["prevalence"] == 4 / 10
    assert result["average_precision"] == pytest.approx(47 / 140)
    assert result["recall"] == 1 / 4
    assert result["precision"] == 1 / 7
    assert result["f1"] == pytest.approx(2 / 11)
    assert result["events"]["1"] == pytest.approx(
        {
            "recall": 1 / 2,
            "precision": 2 / 7,
            "f1": 4 / 11,
            "false_alarms_per_race": 5 / 2,
        }
    )
    assert result["events"]["2"] == pytest.approx(
        {
            "recall": 1 / 2,
            "precision": 3 / 7,
            "f1": 6 / 13,
            "false_alarms_per_race": 2,
        }
    )


def test_bootstrap_counts_race_copies_and_pairs_the_same_draws():
    left = frame(
        [
            (R1, "44", 1, 0, 0.9),
            (R1, "44", 2, 0, 0.9),
            (R2, "44", 1, 0, 0.1),
            (R2, "44", 2, 1, 0.9),
        ]
    )
    # Seed 0 draws R2/R2, R2/R1, R1/R1, R1/R1, R1/R2, R2/R2.
    result, missing = group_metrics(left, 0.9, bootstrap_samples=6, seed=0)
    false_alarms = "events.1.false_alarms_per_race"
    assert result["confidence_intervals"][false_alarms] == {"lower": 0, "upper": 2}
    assert result["confidence_intervals"]["events.2.recall"] == {"lower": 1, "upper": 1}
    assert missing["average_precision"] == missing["events.2.recall"] == 2

    kwargs = {"bootstrap_samples": 6, "seed": 0}
    paired = paired_bootstrap(left, left.iloc[::-1], 0.9, 0.9, **kwargs)
    assert paired["differences"]["f1"] == 0
    assert paired["confidence_intervals"]["f1"] == {"lower": 0, "upper": 0}
    assert paired["ties"]["f1"] is True
    assert paired["bootstrap"]["undefined_replicates"]["average_precision"] == 2

    right = left.assign(score=[0, 0, 0, 0])
    paired = paired_bootstrap(left, right, 0.9, 0.9, **kwargs)
    assert paired["differences"][false_alarms] == 1
    assert paired["confidence_intervals"][false_alarms] == {"lower": 0, "upper": 2}
    assert paired["confidence_intervals"]["events.2.f1"] == {"lower": 0.5, "upper": 1}
    assert paired["ties"]["events.2.f1"] is False
    assert paired["bootstrap"]["undefined_replicates"]["events.2.f1"] == 2
    assert paired == paired_bootstrap(left.iloc[::-1], right, 0.9, 0.9, **kwargs)
    with pytest.raises(ValueError, match="identical keys and labels"):
        paired_bootstrap(left, right.iloc[:-1], 0.9, 0.9, **kwargs)


def test_empty_and_degenerate_metrics():
    rows = frame([(R1, "44", 1, 0, 0.1), (R2, "44", 1, 1, 0.1)])
    empty, missing = group_metrics(rows.iloc[:0], 1, bootstrap_samples=3)
    assert empty["n_samples"] == empty["n_positives"] == empty["n_races"] == 0
    assert empty["prevalence"] is None
    assert empty["events"]["1"]["false_alarms_per_race"] is None
    assert empty["undefined"]["events.1.false_alarms_per_race"] == "empty_group"
    assert empty["confidence_intervals"]["events.1.f1"] == {"lower": None, "upper": None}
    assert set(missing.values()) == {3}

    no_positives, _ = group_metrics(rows.iloc[:1], 0, bootstrap_samples=3)
    assert no_positives["average_precision"] is None
    assert no_positives["undefined"]["average_precision"] == "no_positives"
    assert no_positives["undefined"]["events.2.recall"] == "no_positives"
    assert no_positives["events"]["2"]["f1"] == 0
    assert no_positives["events"]["2"]["false_alarms_per_race"] == 1

    no_alarms, _ = group_metrics(rows.iloc[1:], 1, bootstrap_samples=3)
    assert no_alarms["average_precision"] == 1
    assert no_alarms["undefined"]["precision"] == "no_alarms"
    assert no_alarms["events"]["1"]["recall"] == no_alarms["events"]["1"]["f1"] == 0
    neither, _ = group_metrics(rows.iloc[:1], 1, bootstrap_samples=3)
    assert neither["undefined"]["f1"] == "no_positives_or_alarms"
