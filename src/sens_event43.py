"""Sensitivity: what if changeover event 43 (B->A, 532 s, recorded right after a 1.7-day gap) is dropped? B->A then has n = 2
(671 s and 611 s, mean 641 s instead of 605 s). Re-runs the baseline comparison with and without it. Output: results/sens_event43.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc, simulate as sim  # noqa: E401
proc, chg = sc.load_empirical()
chg2 = dict(chg)
ev = pd.read_csv(sc.RES / "events.csv")
keep = ev[(ev.event == "changeover") & (ev.transition == "B->A") & ev.matrix.isin(sc.PRIMARY_MATRICES) & (ev.event_id != 43)]
chg2["B->A"] = keep.duration_s.to_numpy(float)
print("B->A observations used:", chg["B->A"], "->", chg2["B->A"])
rows = []
for label, c in (("with event 43 (used)", chg), ("without event 43", chg2)):
    lam = sc.load_levels(sc.MIXES["balanced"], proc, c)["medium"]
    for strat, opts in (("fcfs", {}), ("edd", {}), ("grouped", {}), ("opt", dict(opt_lam=30, opt_window=30, opt_solver="dp"))):
        r = []
        for rep in range(30):
            seed = 1000 + rep
            jobs = sc.generate_jobs("balanced", lam, 4, seed, emp=(proc, c))
            r.append(sim.simulate(jobs, strat, c, seed, **opts))
        d = pd.DataFrame(r)
        rows.append({"version": label, "strategy": strat, "jobs_per_h": round(lam, 2), "changeover_min_per_h": d.changeover_s_per_h.mean() / 60,
                     "avg_lateness_h": d.avg_tardiness_s.mean() / 3600, "pct_late": 100 * d.pct_late.mean()})
out = pd.DataFrame(rows).round(2)
out.to_csv(sc.RES / "sens_event43.csv", index=False)
print(out.to_string(index=False))
