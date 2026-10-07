"""Tests for src/scenarios.py. Run:  python3 src/check_scenarios.py   (from the repo root)

Checks the generator against known answers (rate, mix, reproducibility, common random numbers, due-date rules)
and MEASURES why the plan's original due-date rule (TWK: due = arrival + k * own processing time) was replaced.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc  # noqa: E402

emp = sc.load_empirical()
proc, chg = emp
mix = sc.MIXES["balanced"]
lam = sc.load_levels(mix, proc, chg)["medium"]

# 1. Reproducible: same inputs -> identical jobs
a = sc.generate_jobs("balanced", lam, 4, seed=7, emp=emp)
b = sc.generate_jobs("balanced", lam, 4, seed=7, emp=emp)
assert a.equals(b), "same seed must give identical jobs"
print("1. reproducible with same seed: OK")

# 2. Common random numbers: changing k must not change arrivals, products, processing times
c = sc.generate_jobs("balanced", lam, 8, seed=7, emp=emp)
assert (a[["arrival_s", "product", "proc_s"]].equals(c[["arrival_s", "product", "proc_s"]]))
_w = sc.load_window(mix, proc, chg)
unit = _w["pbar_s"] + _w["sbar_fcfs_s"]  # slack unit = average run time + average unsorted-line changeover
assert np.allclose(c.due_s - a.due_s, (8 - 4) * unit)
print("2. k changes only the due dates (common random numbers): OK")

# 3. Statistics over many seeds: arrival rate, interarrival shape, mix proportions
counts, inter, prods = [], [], []
for s in range(300):
    j = sc.generate_jobs("balanced", lam, 4, seed=1000 + s, emp=emp)
    counts.append(len(j) / sc.HORIZON_H)
    inter.extend(np.diff(j.arrival_s.to_numpy()))
    prods.extend(j["product"])
rate = np.mean(counts)
assert abs(rate - lam) / lam < 0.02, (rate, lam)
inter = np.array(inter)
cv = inter.std() / inter.mean()
assert abs(inter.mean() - 3600 / lam) / (3600 / lam) < 0.02 and abs(cv - 1) < 0.05, (inter.mean(), cv)
share = pd.Series(prods).value_counts(normalize=True)
assert all(abs(share[p] - 1 / 3) < 0.02 for p in sc.PRODUCTS), share
print(f"3. rate {rate:.2f}/h vs target {lam:.2f}; interarrival CV {cv:.2f} (exponential = 1); "
      f"mix A/B/C = {share['A']:.3f}/{share['B']:.3f}/{share['C']:.3f}: OK")

# 4. Processing times come only from the observed runs of that product
for p in sc.PRODUCTS:
    assert set(a[a["product"] == p].proc_s) <= set(proc[p]), p
print("4. processing times are real observed values only: OK")

# 5. Every other mix hits its target proportions
for name, m in sc.MIXES.items():
    la = sc.load_levels(m, proc, chg)["medium"]
    pr = pd.concat([sc.generate_jobs(name, la, 4, seed=500 + s, emp=emp)["product"] for s in range(100)]).value_counts(normalize=True)
    assert all(abs(pr[p] - m[i]) < 0.02 for i, p in enumerate(sc.PRODUCTS)), (name, pr)
print("5. all four mixes match their target proportions: OK")

# 6. SLK rule: no job is impossible; allowance always >= own processing time
assert (a.due_s - a.arrival_s >= a.proc_s + 2 * unit - 1e-6).all()
print("6. SLK rule: every job has its own run time plus at least 2 job-cycles (run time + average changeover) of slack: OK")

# 7. Load windows are ordered and consistent
for name, m in sc.MIXES.items():
    w, lv = sc.load_window(m, proc, chg), sc.load_levels(m, proc, chg)
    assert w["lam_lo"] < lv["light"] < lv["medium"] < lv["heavy"] < w["lam_hi"]
    assert abs(w["lam_hi"] - 3600 / w["pbar_s"]) < 1e-9
print("7. load windows: lo < light < medium < heavy < hi for every mix: OK")

# 8. MEASURE the problem with the plan's original TWK rule: how many jobs have an allowance (k * own p)
#    too small to cover even the cheapest possible changeover INTO their product plus their own run?
S = sc.mean_changeover_matrix(chg)
min_into = {p: min(S[i, j] for i in range(3) if i != j) for j, p in enumerate(sc.PRODUCTS)}
print("\n8. TWK rule check (balanced mix): share of jobs that can be on time ONLY IF the machine is already set up for their product")
print("   (allowance k*p < p + the cheapest average changeover into their product, so any changeover at all makes them late; cheapest-into = "
      + ", ".join(f"{p}: {v:.0f} s" for p, v in min_into.items()) + ")")
big = pd.concat([sc.generate_jobs("balanced", lam, 2, seed=2000 + s, emp=emp) for s in range(50)])
rows = []
for k in sc.K_VALUES:
    imp = (k * big.proc_s) < (big.proc_s + big["product"].map(min_into))
    rows.append({"k": k, "all jobs": f"{imp.mean():.0%}", **{f"product {p}": f"{imp[big['product'] == p].mean():.0%}" for p in sc.PRODUCTS}})
print(pd.DataFrame(rows).to_string(index=False))
print("\nALL CHECKS PASSED")
