"""Step 4: ASSUMED demand scenarios (arrivals, product mix, due dates).

REAL (from the CNC data, via results/events.csv): processing times per product, changeover times per transition.
ASSUMED (labeled, swept, never hidden): arrival rate, product mix, due-date slack k, horizon, load placement.

A "job" here = ONE production run (one workpiece). The dataset does not state a quantity, so this is an assumption.

Random numbers: every scenario is a deterministic function of (mix, lam, k, seed). Arrivals, products and
processing times depend ONLY on (mix, lam, seed), not on k, so different k (and different strategies) see the
exact same jobs (common random numbers).
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"

PRODUCTS = ["A", "B", "C"]
MIXES = {  # probabilities of (A, B, C)
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "A-heavy": (0.6, 0.2, 0.2),
    "B-heavy": (0.2, 0.6, 0.2),
    "C-heavy": (0.2, 0.2, 0.6),
}
LOAD_FRACTIONS = {"light": 0.15, "medium": 0.50, "heavy": 0.75}  # position inside the load window
K_VALUES = [1, 2, 3, 4, 6, 8, 12]  # slack in job-cycles (run time + average changeover, ~20 min); baseline 4
HORIZON_H = 40.0
PRIMARY_MATRICES = [3, 4, 5]  # same choice as step 3 (learning-curve effect excluded)


def load_empirical():
    """Observed processing times per product and changeover times per transition (seconds)."""
    ev = pd.read_csv(RES / "events.csv")
    proc = {p: ev[(ev.event == "production") & (ev.new_product == p)].duration_s.to_numpy(float) for p in PRODUCTS}
    co = ev[(ev.event == "changeover") & ev.matrix.isin(PRIMARY_MATRICES)]
    chg = {t: g.duration_s.to_numpy(float) for t, g in co.groupby("transition")}
    return proc, chg


def mean_changeover_matrix(chg):
    """3x3 array S[i, j] = mean changeover time from product i to product j (0 on the diagonal)."""
    S = np.zeros((3, 3))
    for i, a in enumerate(PRODUCTS):
        for j, b in enumerate(PRODUCTS):
            if i != j:
                S[i, j] = chg[f"{a}->{b}"].mean()
    return S


def load_window(mix, proc, chg):
    """Plan formula: 1/(p+s) < lambda < 1/p, in jobs per HOUR.

    p = mix-weighted mean processing time. s = mean changeover when each job's product is drawn independently
    from the mix, i.e. what an unsorted (FCFS) line pays: sum_ij w_i w_j S_ij (no changeover if same product)."""
    w = np.array(mix)
    pbar = float(sum(w[i] * proc[p].mean() for i, p in enumerate(PRODUCTS)))
    S = mean_changeover_matrix(chg)
    sbar = float(w @ S @ w)
    return {"pbar_s": pbar, "sbar_fcfs_s": sbar, "lam_lo": 3600 / (pbar + sbar), "lam_hi": 3600 / pbar}


def load_levels(mix, proc, chg):
    win = load_window(mix, proc, chg)
    return {name: win["lam_lo"] + f * (win["lam_hi"] - win["lam_lo"]) for name, f in LOAD_FRACTIONS.items()}


def generate_jobs(mix_name, lam_per_h, k, seed, horizon_h=HORIZON_H, due_rule="slk", emp=None, slack_unit="cycle"):
    """Return a DataFrame of jobs: job_id, arrival_s, product, proc_s, due_s.

    due_rule 'slk' (default): due = arrival + proc + k * UNIT   (same slack for every job)
        slack_unit 'cycle' (default since 2026-10-07): UNIT = pbar_mix + sbar_mix = average run time PLUS the average changeover an
                           unsorted line pays per job (balanced mix: about 1,200 s = 20 min). k = "how many job-cycles of waiting
                           the customer allows".
        slack_unit 'run'  (original Steps 4-7): UNIT = pbar_mix only (about 520 s = 9 min). Kept so old results can be reproduced.
    due_rule 'twk'          : due = arrival + k * proc               (plan's original; see docs for why not default)
    """
    proc, chg = emp if emp is not None else load_empirical()
    mix = MIXES[mix_name]
    win = load_window(mix, proc, chg)
    unit = win["pbar_s"] + (win["sbar_fcfs_s"] if slack_unit == "cycle" else 0.0)
    assert slack_unit in ("cycle", "run"), slack_unit
    rng = np.random.default_rng(seed)
    horizon_s = horizon_h * 3600.0
    n_draw = int(lam_per_h * horizon_h * 2 + 100)
    arrivals = np.cumsum(rng.exponential(3600.0 / lam_per_h, size=n_draw))
    assert arrivals[-1] > horizon_s, "drew too few arrivals to cover the horizon"
    arrivals = arrivals[arrivals < horizon_s]
    n = len(arrivals)
    prod_idx = rng.choice(3, size=n, p=mix)
    proc_s = np.array([proc[PRODUCTS[i]][rng.integers(len(proc[PRODUCTS[i]]))] for i in prod_idx])
    if due_rule == "slk":
        due = arrivals + proc_s + k * unit
    elif due_rule == "twk":
        due = arrivals + k * proc_s
    else:
        raise ValueError(due_rule)
    return pd.DataFrame({"job_id": np.arange(1, n + 1), "arrival_s": arrivals,
                         "product": [PRODUCTS[i] for i in prod_idx], "proc_s": proc_s, "due_s": due})


def main():
    emp = load_empirical()
    proc, chg = emp
    rows = []
    for name, mix in MIXES.items():
        win = load_window(mix, proc, chg)
        lv = load_levels(mix, proc, chg)
        rows.append({"mix": name, "p_A": mix[0], "p_B": mix[1], "p_C": mix[2],
                     **{k: round(v, 1) if k.endswith("_s") else round(v, 2) for k, v in win.items()},
                     **{f"lam_{n}": round(v, 2) for n, v in lv.items()},
                     "rho_at_heavy_if_no_changeovers": round(lv["heavy"] * win["pbar_s"] / 3600, 2)})
    wdf = pd.DataFrame(rows)
    wdf.to_csv(RES / "load_windows.csv", index=False)
    ex = generate_jobs("balanced", load_levels(MIXES["balanced"], proc, chg)["medium"], k=4, seed=1, emp=emp)
    ex.round(1).to_csv(RES / "example_jobs.csv", index=False)
    pd.set_option("display.width", 220)
    print("LOAD WINDOWS (jobs/hour)"); print(wdf.to_string(index=False))
    print(f"\nexample_jobs.csv: {len(ex)} jobs (balanced mix, medium load, k=4, seed=1); first 6:")
    print(ex.head(6).round(0).to_string(index=False))


if __name__ == "__main__":
    main()
