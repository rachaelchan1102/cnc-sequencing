"""Tests for src/simulate.py. Run:  python3 src/check_simulate.py   (from the repo root)

Part A: tiny hand-solvable cases with fixed changeover times (answers worked out on paper).
Part B: statistical / sanity checks on the real scenarios.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc  # noqa: E402
import simulate as sim  # noqa: E402


def J(rows):
    return pd.DataFrame(rows, columns=["arrival_s", "product", "proc_s", "due_s"]).assign(
        job_id=lambda d: np.arange(1, len(d) + 1))


# deterministic changeovers: A->B = 100 s, B->A = 50 s (others unused)
CHG = {t: np.array([1.0]) for t in ["A->C", "C->A", "B->C", "C->B"]}
CHG.update({"A->B": np.array([100.0]), "B->A": np.array([50.0])})
KW = dict(horizon_h=0.1, warmup_h=0.0, chg_scale=0.0, return_detail=True)  # horizon = 360 s, no warm-up

# ---- Part A -------------------------------------------------------------------------------------------
# j1: A, arrives 0, runs 10 | j2: B, arrives 5, runs 20 | j3: A, arrives 6, runs 10
jobs = J([(0, "A", 10, 100), (5, "B", 20, 120), (6, "A", 10, 150)])
m, d = sim.simulate(jobs, "fcfs", CHG, 1, **KW)
# FCFS: j1 0-10; j2 changeover A->B 10-110 then 110-130; j3 changeover B->A 130-180 then 180-190
assert np.allclose(d["finish"], [10, 130, 190]), d["finish"]
assert abs(m["changeover_s_per_h"] - 150 / (360 / 3600)) < 1e-6 and m["n_changeovers"] == 2
assert abs(m["avg_tardiness_s"] - (0 + 10 + 40) / 3) < 1e-9
print("A1. FCFS matches hand calculation (finish 10, 130, 190; changeovers 150 s; avg tardiness 16.67): OK")

m, d = sim.simulate(jobs, "grouped", CHG, 1, **KW)
# grouped: after j1 (A) the waiting A job j3 goes next: 10-20; then j2: changeover 20-120, run 120-140
assert np.allclose(d["finish"], [10, 140, 20]), d["finish"]
assert abs(m["changeover_s_per_h"] - 100 / (360 / 3600)) < 1e-6 and m["n_changeovers"] == 1
assert abs(m["avg_tardiness_s"] - (0 + 20 + 0) / 3) < 1e-9
print("A2. grouped matches hand calculation (finish 10, 140, 20; changeovers 100 s; avg tardiness 6.67): OK")

jobs_edd = J([(0, "A", 10, 1000), (5, "B", 20, 500), (6, "A", 10, 50)])
_, d = sim.simulate(jobs_edd, "edd", CHG, 1, **KW)
assert np.allclose(d["finish"], [10, 140, 20]), d["finish"]  # j3 (due 50) beats j2 (due 500)
print("A3. EDD picks the earliest due date (j3 before j2): OK")

jobs_cap = J([(0, "A", 10, 1000), (1, "A", 10, 1000), (2, "A", 10, 1000), (3, "B", 10, 1000)])
_, d_nocap = sim.simulate(jobs_cap, "grouped", CHG, 1, **KW)
_, d_cap = sim.simulate(jobs_cap, "grouped", CHG, 1, max_batch=2, **KW)
assert np.allclose(d_nocap["finish"], [10, 20, 30, 140]), d_nocap["finish"]
assert np.allclose(d_cap["finish"], [10, 20, 190, 130]), d_cap["finish"]
print("A4. batch cap of 2 forces a switch to B before the third A (finish 10, 20, 190, 130 vs uncapped 10, 20, 30, 140): OK")

late = J([(355, "A", 100, 300)])  # cannot finish before the 360 s horizon
m, _ = sim.simulate(late, "fcfs", CHG, 1, **KW)
assert m["unfinished_jobs"] == 1 and abs(m["avg_tardiness_s"] - 60) < 1e-9 and abs(m["avg_flow_s"] - 5) < 1e-9
print("A5. a job unfinished at the end is counted (tardiness lower bound 360-300 = 60 s), not dropped: OK")

# ---- Part B -------------------------------------------------------------------------------------------
emp = sc.load_empirical()
proc, chg = emp
mix = sc.MIXES["balanced"]
lv = sc.load_levels(mix, proc, chg)

# B1. common random numbers: every strategy sees the identical jobs
r = {s: sim.run_reps("balanced", lv["medium"], 4, s, n_rep=10, emp=emp) for s in ("fcfs", "edd", "grouped")}
assert all((r[s]["jobs_counted"].values == r["fcfs"]["jobs_counted"].values).all() for s in ("fcfs", "edd", "grouped"))
print("B1. all three strategies face the same jobs in each replication: OK")

# B2. an unsorted line changes over when the next product differs: probability 2/3 for a balanced mix
fr = []
for s in range(100):
    jobs_b = sc.generate_jobs("balanced", sc.load_levels(mix, proc, chg)["light"], 4, 7000 + s, emp=emp)
    _, dd = sim.simulate(jobs_b, "fcfs", chg, 7000 + s, return_detail=True)
    if dd["jobs_done"] > 20:
        fr.append(dd["co_events"] / (dd["jobs_done"] - 1))
assert abs(np.mean(fr) - 2 / 3) < 0.03, np.mean(fr)
print(f"B2. FCFS changes over on {np.mean(fr):.3f} of jobs (theory for a balanced mix: 0.667): OK")

# B3. at very low load nothing waits, so nobody is meaningfully late and the machine is mostly idle
low = pd.concat([sim.run_reps("balanced", 0.2, 8, s, n_rep=30, emp=emp) for s in ("fcfs", "edd", "grouped")])
assert low.avg_tardiness_s.mean() < 60 and low.utilization.max() < 0.35, (low.avg_tardiness_s.mean(), low.utilization.max())
print(f"B3. very low load: mean tardiness {low.avg_tardiness_s.mean():.1f} s, max utilization {low.utilization.max():.2f}: OK")

# B4. Little's law in a stable run: average jobs in system = throughput x average time in system
g = sim.run_reps("balanced", lv["light"], 4, "grouped", n_rep=60, emp=emp)
little = (g.throughput_per_h * g.avg_flow_s / 3600).mean()
assert abs(little - g.wip.mean()) / g.wip.mean() < 0.12, (little, g.wip.mean())
print(f"B4. Little's law (grouped, light load): throughput x flow = {little:.2f} vs measured WIP {g.wip.mean():.2f}: OK")

# B5. direction: sorting by product removes changeovers
assert r["grouped"].changeover_s_per_h.mean() < r["fcfs"].changeover_s_per_h.mean()
print(f"B5. at medium load grouped changeover {r['grouped'].changeover_s_per_h.mean():.0f} s/h < FCFS {r['fcfs'].changeover_s_per_h.mean():.0f} s/h: OK")

# B6. utilization can never exceed 100%
assert all((r[s].utilization <= 1 + 1e-9).all() for s in ("fcfs", "edd", "grouped"))
print("B6. utilization never exceeds 100%: OK")

# B7. (Step 6) the "opt" strategy plugged into the simulator: every job still gets handled, utilization <= 100%,
#     and with a very large changeover weight it should switch about as little as Grouped does.
o = sim.run_reps("balanced", lv["medium"], 4, "opt", n_rep=10, emp=emp, opt_lam=300, opt_window=30, opt_solver="dp")
assert (o.jobs_counted.values == r["fcfs"]["jobs_counted"].values).all() and (o.utilization <= 1 + 1e-9).all()
assert o.changeover_s_per_h.mean() < 1.25 * r["grouped"].changeover_s_per_h.mean()
print(f"B7. opt (lam=300): changeover {o.changeover_s_per_h.mean():.0f} s/h vs grouped {r['grouped'].changeover_s_per_h.mean():.0f}, "
      f"FCFS {r['fcfs'].changeover_s_per_h.mean():.0f}: OK")
print("\nALL CHECKS PASSED")
