"""D11 race view: reads a precomputed demo pack, never trains or imports pitstop."""

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


def main():
    st.title("Pit stop race view")
    pack_dir = Path(os.environ.get("PITSTOP_DEMO_DIR", "demo"))
    try:
        df, thresholds = load_pack(pack_dir)
    except FileNotFoundError:
        st.info(f"Demo pack not found in `{pack_dir}`. Set PITSTOP_DEMO_DIR or add demo/.")
        return
    race_id = st.selectbox("Race", sorted(df.race_id.unique()))
    race = df[df.race_id == race_id]
    driver = st.selectbox("Driver", sorted(race.driver_number.unique()))
    run_id = st.selectbox("Model (run_id)", sorted(race.run_id.unique()))
    rows = filter_rows(df, race_id, driver, run_id)
    threshold = thresholds[run_id]["threshold"]
    st.altair_chart(make_chart(rows, threshold), use_container_width=True)
    st.caption(
        "A score at lap t predicts a tyre change on lap t+1, so scores are plotted at "
        "target_lap. Red lines are real tyre changes, orange bands are SC/VSC laps "
        "(status at lap t), the dashed line is the frozen threshold."
    )


main()
