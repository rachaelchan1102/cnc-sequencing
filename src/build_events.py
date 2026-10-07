"""Step 2: reconstruct the event table from the raw 1 Hz CNC data.

Input : data/data.csv   (one row = one second of machine state)
Output: results/events.csv  (one row = one changeover OR one production run)

Why this works (verified against the authors' Data_irregularities log):
  - Label_08 has 12 values. Odd = changeover, even = production. The pair
    (k, k+1) is one order: changeover k followed by the production run k+1.
  - The changeover label encodes the transition (see TRANSITIONS).
  - Each changeover matrix runs the orders in the order EXPECTED_ORDER.

Durations are counted in rows (= seconds, since sampling is 1 Hz), never by
subtracting timestamps across recording gaps.
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "data.csv"
OUT = REPO / "results" / "events.csv"

GAP_S = 60  # a jump between consecutive timestamps above this = recording gap
TRANSITIONS = {5: ("A", "B"), 11: ("B", "C"), 7: ("C", "B"),
               1: ("B", "A"), 9: ("A", "C"), 3: ("C", "A")}
EXPECTED_ORDER = [5, 11, 7, 1, 9, 3]  # B, C, B, A, C, A


def load(path=DATA):
    df = pd.read_csv(path, index_col=0, low_memory=False).reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_events(df):
    step = df["time"].diff().dt.total_seconds()
    gap = step.fillna(1) > GAP_S
    # A new segment starts whenever Label_08 changes OR a recording gap occurs.
    seg_id = ((df["Label_08"] != df["Label_08"].shift()) | gap).cumsum()

    rows = []
    for _, g in df.groupby(seg_id):
        l8 = int(g["Label_08"].iloc[0])
        is_changeover = l8 % 2 == 1
        rows.append({
            "label_08": l8,
            "event": "changeover" if is_changeover else "production",
            "start_row": int(g.index[0]),
            "end_row": int(g.index[-1]),
            "start_time": g["time"].iloc[0],
            "end_time": g["time"].iloc[-1],
            "duration_s": len(g),                      # rows == seconds at 1 Hz
            "irregular_s": int((g["Label_07"] == 2).sum()),
            "label_03_values": sorted(g["Label_03"].unique().tolist()),
            "gap_before_s": float(step.loc[g.index[0]]) if step.loc[g.index[0]] > GAP_S else 0.0,
        })
    ev = pd.DataFrame(rows)

    # Matrix = counts of A->B changeovers (label 5) so far.
    ev["matrix"] = (ev["label_08"] == 5).cumsum()
    ev["order_in_matrix"] = ev.groupby("matrix").cumcount() // 2 + 1

    # Products: changeover label gives (prev, new); production runs product 'new'.
    def prev_new(l8):
        k = l8 if l8 % 2 == 1 else l8 - 1
        return TRANSITIONS[k]
    ev["prev_product"] = ev["label_08"].map(lambda l: prev_new(l)[0] if l % 2 == 1 else prev_new(l)[1])
    ev["new_product"] = ev["label_08"].map(lambda l: prev_new(l)[1])
    ev["transition"] = ev.apply(
        lambda r: f"{r.prev_product}->{r.new_product}" if r.event == "changeover" else "", axis=1)
    ev["regular_s"] = ev["duration_s"] - ev["irregular_s"]
    ev.insert(0, "event_id", range(1, len(ev) + 1))
    return ev


def validate(df, ev):
    """Fail loudly if any assumption we verified earlier is violated."""
    assert len(ev) == 60, f"expected 60 events, got {len(ev)}"
    assert ev["duration_s"].sum() == len(df), "durations must cover every row exactly once"
    assert (ev["event"].iloc[::2] == "changeover").all() and (ev["event"].iloc[1::2] == "production").all()
    assert (ev["label_08"].iloc[1::2].values == ev["label_08"].iloc[::2].values + 1).all(), "production must follow its changeover"
    for m, g in ev[ev.event == "changeover"].groupby("matrix"):
        assert g["label_08"].tolist() == EXPECTED_ORDER, f"matrix {m} order differs: {g['label_08'].tolist()}"
    assert ev["matrix"].nunique() == 5
    # Inside a segment, timestamps must be contiguous: end - start + 1 == rows.
    span = (ev["end_time"] - ev["start_time"]).dt.total_seconds() + 1
    assert (span == ev["duration_s"]).all(), "a segment contains a recording gap"
    # Label_03 must be constant in a segment and equal to 0 (changeover) / 1 (production).
    want = ev["event"].map({"changeover": [0.0], "production": [1.0]})
    assert (ev["label_03_values"].apply(tuple) == want.apply(tuple)).all(), "Label_03 disagrees with Label_08 parity"


def main():
    df = load()
    ev = build_events(df)
    validate(df, ev)
    OUT.parent.mkdir(exist_ok=True)
    ev.drop(columns=["label_03_values"]).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}: {len(ev)} events, all validation checks passed")
    return ev


if __name__ == "__main__":
    main()
