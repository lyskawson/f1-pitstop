"""D11 views: race, field heatmap, model comparison. Reads a precomputed demo pack only."""

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

KEY = ["race_id", "driver_number", "lap_number"]


@st.cache_data
def load_pack(pack_dir):
    """Return (predictions joined with sc/vsc flags, thresholds by run_id)."""
    pack = Path(pack_dir)
    predictions = pd.read_parquet(pack / "predictions.parquet")
    samples = pd.read_parquet(pack / "samples.parquet")[[*KEY, "sc", "vsc"]]
    thresholds = json.loads((pack / "thresholds.json").read_text())
    return predictions.merge(samples, on=KEY, validate="many_to_one"), thresholds


def filter_rows(df, race_id, driver, run_id):
    rows = df[(df.race_id == race_id) & (df.driver_number == driver) & (df.run_id == run_id)]
    return rows.sort_values("target_lap")


def make_chart(rows, threshold):
    base = alt.Chart(rows)
    # Status at lap t is shaded on the target lap, where the score is plotted.
    bands = (
        base.transform_filter("datum.sc || datum.vsc")
        .mark_rect(opacity=0.25, color="orange")
        .encode(x="target_lap:Q", x2="target_lap_end:Q")
        .transform_calculate(target_lap_end="datum.target_lap + 1")
    )
    line = base.mark_line().encode(
        x=alt.X("target_lap:Q", title="Target lap (t+1)"),
        y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
    )
    pits = base.transform_filter("datum.y == 1").mark_rule(color="red").encode(x="target_lap:Q")
    rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(
        color="green", strokeDash=[6, 4]
    )
    return (bands + line + pits + rule.encode(y="threshold:Q")).properties(height=350)


def make_heatmap(rows, threshold):
    base = alt.Chart(rows).encode(
        x=alt.X("target_lap:O", title="Target lap (t+1)"),
        y=alt.Y("driver_number:N", title="Driver"),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("score:Q", scale=alt.Scale(domain=[0, 1], scheme="blues"), title="Score")
    )
    # Alarm = score >= frozen threshold (dot); real tyre change = red outline.
    alarms = base.transform_filter(alt.datum.score >= threshold).mark_point(
        shape="circle", color="black", size=20
    )
    pits = base.transform_filter("datum.y == 1").mark_rect(
        fill=None, stroke="red", strokeWidth=2, strokeOpacity=1
    )
    return (cells + alarms + pits).properties(height=alt.Step(18))


def comparison_table(metrics):
    """One row per run, copied from the test/overall group of metrics.json."""
    rows = []
    for run_id, record in metrics.items():
        g = record["splits"]["test"]["overall"]
        ci = g["confidence_intervals"]["average_precision"]
        _, model, seed, *_ = run_id.split("_")
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "seed": seed,
                "ap": g["average_precision"],
                "ap_lower": ci["lower"],
                "ap_upper": ci["upper"],
                "prevalence": g["prevalence"],
                "f1": g["f1"],
                "precision": g["precision"],
                "recall": g["recall"],
                "event_f1_k1": g["events"]["1"]["f1"],
                "false_alarms_per_race_k1": g["events"]["1"]["false_alarms_per_race"],
            }
        )
    return pd.DataFrame(rows)


def make_ap_chart(table):
    points = alt.Chart(table).encode(x=alt.X("model:N", title="Model"))
    ap = points.mark_point(filled=True, size=60).encode(
        y=alt.Y("ap:Q", title="Average precision (test)")
    )
    bars = points.mark_errorbar().encode(y="ap_lower:Q", y2="ap_upper:Q")
    # Prevalence is a property of the test split, identical for all runs.
    ref = (
        alt.Chart(table.head(1)).mark_rule(color="red", strokeDash=[6, 4]).encode(y="prevalence:Q")
    )
    return (bars + ap + ref).properties(height=300)


def race_tab(df, race_id, run_id, threshold):
    race = df[df.race_id == race_id]
    driver = st.selectbox("Driver", sorted(race.driver_number.unique()))
    rows = filter_rows(df, race_id, driver, run_id)
    st.altair_chart(make_chart(rows, threshold), width="stretch")
    st.caption(
        "A score at lap t predicts a tyre change on lap t+1, so scores are plotted at "
        "target_lap. Red lines are real tyre changes, orange bands are SC/VSC laps "
        "(status at lap t), the dashed line is the frozen threshold."
    )


def heatmap_tab(df, race_id, run_id, threshold):
    rows = df[(df.race_id == race_id) & (df.run_id == run_id)]
    st.altair_chart(make_heatmap(rows, threshold), width="stretch")
    st.caption(
        "Colour is the score (fixed 0..1 scale) at target_lap. Black dots mark alarms "
        "(score >= frozen threshold); red outlines mark real tyre changes on that lap."
    )


def comparison_tab(pack_dir):
    try:
        table = comparison_table(json.loads((pack_dir / "metrics.json").read_text()))
    except FileNotFoundError:
        st.info("`metrics.json` not found in the demo pack, so there is no comparison.")
        return
    except (KeyError, ValueError) as err:
        st.error(f"Inconsistent `metrics.json`: {err!r}.")
        return
    st.dataframe(table.drop(columns="run_id"), hide_index=True)
    st.altair_chart(make_ap_chart(table), width="stretch")
    st.caption(
        "Full test-split metrics (all test races, not only the demo races) of the seed-42 "
        "run of each model; bars are 95% bootstrap CIs and the red line is the prevalence. "
        "Overlapping CIs mean no ranking. The mean over seeds is in reports/test-v1.md."
    )


def main():
    st.title("Pit stop dashboard")
    pack_dir = Path(os.environ.get("PITSTOP_DEMO_DIR", "demo"))
    try:
        df, thresholds = load_pack(pack_dir)
    except FileNotFoundError as err:
        st.info(
            f"Demo pack incomplete: `{err.filename}` not found. Set PITSTOP_DEMO_DIR or add demo/."
        )
        return
    except KeyError as err:
        st.error(f"Inconsistent demo pack: missing {err}.")
        return
    try:
        names = json.loads((pack_dir / "metadata.json").read_text())["event_names"]
    except (FileNotFoundError, KeyError, ValueError):
        names = {}  # fall back to race_id
    race_id = st.selectbox(
        "Race", sorted(df.race_id.unique()), format_func=lambda r: names.get(r, r)
    )
    run_id = st.selectbox("Model (run_id)", sorted(df[df.race_id == race_id].run_id.unique()))
    race_t, heat_t, comp_t = st.tabs(["Race view", "Field heatmap", "Model comparison"])
    with comp_t:
        comparison_tab(pack_dir)
    if run_id not in thresholds:
        for tab in (race_t, heat_t):
            tab.error(f"Inconsistent demo pack: no threshold for run_id {run_id}.")
        return
    threshold = thresholds[run_id]["threshold"]
    with race_t:
        race_tab(df, race_id, run_id, threshold)
    with heat_t:
        heatmap_tab(df, race_id, run_id, threshold)


main()
