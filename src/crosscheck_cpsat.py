"""Step 6: does the slow general OR-Tools planner (CP-SAT) behave like the fast DP planner INSIDE the simulation?
Same 3 job streams (medium load), window 8, lam = 10. Takes a few minutes. Output: results/crosscheck_cpsat.csv"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc, simulate as sim  # noqa: E401
emp = sc.load_empirical(); proc, chg = emp
lam = sc.load_levels(sc.MIXES["balanced"], proc, chg)["medium"]
rows = []
for solver in ("dp", "cpsat"):
    df = sim.run_reps("balanced", lam, 4, "opt", n_rep=3, emp=emp, opt_lam=10, opt_window=8, opt_solver=solver, opt_time_s=0.5)
    rows.append({"solver": solver, **{m: df[m].mean() for m in ("changeover_s_per_h", "avg_tardiness_s", "pct_late", "unfinished_jobs")}})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(sc.RES / "crosscheck_cpsat.csv", index=False)
