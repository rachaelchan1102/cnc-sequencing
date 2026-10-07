"""Independent cross-check of results/events.csv.

build_events.py segments on Label_08. This script ignores Label_08 and instead
segments on Label_03 (0 = changeover, 1 = production) plus recording gaps, then
compares event counts, durations and irregular seconds to events.csv.
Agreement of two different derivations is our evidence the table is right.
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GAP_S = 60

df = pd.read_csv(REPO / "data" / "data.csv", index_col=0, low_memory=False).reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"])
ev = pd.read_csv(REPO / "results" / "events.csv", parse_dates=["start_time", "end_time"])

gap = df["time"].diff().dt.total_seconds().fillna(1) > GAP_S
seg = ((df["Label_03"] != df["Label_03"].shift()) | gap).cumsum()
alt = df.groupby(seg).agg(l3=("Label_03", "first"), n=("time", "size"),
                          irr=("Label_07", lambda s: int((s == 2).sum())),
                          start=("time", "first")).reset_index(drop=True)

print("segments via Label_03:", len(alt), "| events in events.csv:", len(ev))
# Label_03 can merge adjacent same-type segments only if two changeovers/productions touch; not expected.
assert len(alt) == len(ev) == 60
same_dur = (alt["n"].values == ev["duration_s"].values).all()
same_irr = (alt["irr"].values == ev["irregular_s"].values).all()
same_start = (alt["start"].values == ev["start_time"].values).all()
same_type = (alt["l3"].map({0.0: "changeover", 1.0: "production"}).values == ev["event"].values).all()
print("event type identical:      ", same_type)
print("durations identical:       ", same_dur)
print("irregular secs identical:  ", same_irr)
print("start timestamps identical:", same_start)
assert same_type and same_dur and same_irr and same_start
print("CROSS-CHECK PASSED")
