"""Step 5b: one-at-a-time screening in SimPy -> tornado chart.

Baseline: balanced mix, MEDIUM load, k = 4 job-cycles (~80 min of slack), real changeover variability (100%), grouped with no batch cap.
Each factor is changed alone; everything else stays at baseline. Mix changes keep the same RELATIVE load
(medium position inside that mix's own window) so a mix change is not secretly also an overload.
All strategies see identical jobs in each replication (common random numbers), so the differences
"grouped minus FCFS" are paired and far less noisy than comparing two separate averages.

Outputs: results/screening_results.csv (all metrics, every condition x strategy),
         results/screening_diff.csv    (paired grouped-minus-FCFS differences with 95% CI),
         results/fig_tornado.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc  # noqa: E402
import simulate as sim  # noqa: E402

RES = sc.RES
N_REP = 30
BASE = {"mix": "balanced", "level": "medium", "k": 4, "chg_scale": 1.0, "max_batch": None}
FACTORS = {
    "Load": [("below window (0.7x)", {"level": "below"}), ("light", {"level": "light"}), ("heavy", {"level": "heavy"})],
    "Due-date slack k": [("k = 1", {"k": 1}), ("k = 8", {"k": 8})],
    "Product mix": [("A-heavy", {"mix": "A-heavy"}), ("B-heavy", {"mix": "B-heavy"}), ("C-heavy", {"mix": "C-heavy"})],
    "Changeover variability": [("0%", {"chg_scale": 0.0}), ("200%", {"chg_scale": 2.0})],
    "Batch cap (grouped only)": [("cap 2", {"max_batch": 2}), ("cap 5", {"max_batch": 5}), ("cap 10", {"max_batch": 10})],
}
METRICS = ["changeover_s_per_h", "avg_tardiness_s", "pct_late", "avg_flow_s", "utilization", "wip", "unfinished_jobs"]
SURFACE, INK, INK2, GRID, BLUE, BLUE_D = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6", "#184f95"


def run_condition(emp, spec):
    proc, chg = emp
    mix = sc.MIXES[spec["mix"]]
    if spec["level"] == "below":  # extra setting BELOW the plan's window: 70% of the rate FCFS can sustain
        lam = 0.7 * sc.load_window(mix, proc, chg)["lam_lo"]
    else:
        lam = sc.load_levels(mix, proc, chg)[spec["level"]]
    out = {}
    for s in ("fcfs", "edd", "grouped"):  # the "opt" strategy is Step 6, run in tradeoff.py
        opts = {"chg_scale": spec["chg_scale"]}
        if s == "grouped":
            opts["max_batch"] = spec["max_batch"]
        out[s] = sim.run_reps(spec["mix"], lam, spec["k"], s, n_rep=N_REP, emp=emp, **opts)
    return lam, out


def paired(out, metric):
    d = (out["grouped"][metric] - out["fcfs"][metric]).to_numpy()
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


def main():
    emp = sim.sc.load_empirical()
    conds = [("Baseline", "baseline", dict(BASE))]
    for f, settings in FACTORS.items():
        for label, ov in settings:
            conds.append((f, label, {**BASE, **ov}))
    rows, diffs = [], []
    for f, label, spec in conds:
        lam, out = run_condition(emp, spec)
        for s in ("fcfs", "edd", "grouped"):  # the "opt" strategy is Step 6, run in tradeoff.py
            summ = sim.summarize(out[s])
            rows.append({"factor": f, "setting": label, "strategy": s, "lambda_per_h": round(lam, 2),
                         **{m: summ[m] for m in METRICS}, **{m + "_ci95": summ[m + "_ci95"] for m in METRICS}})
        dt, dtc = paired(out, "avg_tardiness_s")
        dc, dcc = paired(out, "changeover_s_per_h")
        diffs.append({"factor": f, "setting": label, "lambda_per_h": round(lam, 2),
                      "d_tardiness_s": dt, "d_tardiness_ci95": dtc, "d_changeover_s_per_h": dc, "d_changeover_ci95": dcc})
        print(f"done: {f} / {label}", flush=True)
    res, dif = pd.DataFrame(rows), pd.DataFrame(diffs)
    res.round(2).to_csv(RES / "screening_results.csv", index=False)
    dif.round(2).to_csv(RES / "screening_diff.csv", index=False)
    tornado(dif)
    pd.set_option("display.width", 220)
    print("\nBASELINE, all strategies:")
    print(res[res.factor == "Baseline"][["strategy", "lambda_per_h"] + METRICS].round(1).to_string(index=False))
    print("\nPAIRED DIFFERENCE grouped - FCFS (negative = grouped better):")
    print(dif.round(1).to_string(index=False))


ROW_TITLES = {  # plain-language name for each input
    "Load": "How busy the machine is\n(new jobs per hour)",
    "Batch cap (grouped only)": "Limit on same-product jobs\nin a row (Grouped only)",
    "Product mix": "Which products\narrive most",
    "Due-date slack k": "How tight the deadlines are",
    "Changeover variability": "How much changeover\ntimes vary",
}
SETTING_TEXT = {  # plain-language name for each setting (lam = jobs per hour is filled in for load)
    "below window (0.7x)": "very quiet ({lam:.1f}/h)", "light": "light ({lam:.1f}/h)", "heavy": "very busy ({lam:.1f}/h)",
    "cap 2": "max 2 in a row", "cap 5": "max 5 in a row", "cap 10": "max 10 in a row",
    "A-heavy": "mostly A", "B-heavy": "mostly B", "C-heavy": "mostly C",
    "k = 1": "tight deadlines (~20 min)", "k = 8": "loose deadlines (~2.7 h)",
    "0%": "none", "200%": "double the real",
}
BASELINE_TEXT = ("Black line = starting scenario: even mix of A, B, C; medium load (4.9 jobs/h); deadlines allow ~80 min of waiting (k = 4); "
                 "no limit on same-product jobs in a row; real changeover times.")


def tornado(dif):
    """Tornado chart: how much BETTER Grouped is than FCFS (positive = Grouped better), and what changes that.
    One bar per input; the bar spans the lowest to highest result across that input's settings (baseline included).
    Both panels share one row order."""
    base = dif[dif.factor == "Baseline"].iloc[0]
    panels = [("d_tardiness_s", 3600.0, "How much less late are jobs under Grouped?\n(average lateness saved per job, in hours)"),
              ("d_changeover_s_per_h", 60.0, "How much less time is spent switching products?\n(changeover minutes saved per hour)")]

    def spans_for(col, div):
        out = {}
        b = -base[col] / div  # flip sign: positive = Grouped better
        for f in FACTORS:
            d = dif[dif.factor == f]
            vals = -d[col].to_numpy() / div
            labs = [SETTING_TEXT[s].format(lam=l) for s, l in zip(d.setting, d.lambda_per_h)]
            lo_i, hi_i = int(np.argmin(vals)), int(np.argmax(vals))
            out[f] = dict(lo=min(vals.min(), b), hi=max(vals.max(), b), b=b, lo_lab=labs[lo_i], hi_lab=labs[hi_i],
                          lo_val=vals.min(), hi_val=vals.max())
        return out

    first = spans_for(*panels[0][:2])
    order = sorted(FACTORS, key=lambda f: first[f]["hi"] - first[f]["lo"])  # longest bar on top, same in both panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), facecolor=SURFACE)
    for ax, (col, div, title) in zip(axes, panels):
        sp = spans_for(col, div)
        b = sp[order[0]]["b"]
        allv = [v for f in FACTORS for v in (sp[f]["lo"], sp[f]["hi"])] + [0.0]
        rng = max(allv) - min(allv)
        thr = 0.02 * rng
        ax.set_facecolor(SURFACE)
        for y, f in enumerate(order):
            s = sp[f]
            if s["hi"] - s["lo"] < thr:
                ax.plot([b], [y], marker="o", markersize=7, color=BLUE, zorder=5)
                ax.text(b + 0.035 * rng, y, "makes no real difference", ha="left", va="center", fontsize=9, color=INK2)
                continue
            ax.barh(y, s["hi"] - s["lo"], left=s["lo"], height=0.5, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
            if b - s["lo_val"] > thr:
                ax.text(s["lo"] - 0.015 * rng, y, s["lo_lab"], ha="right", va="center", fontsize=9, color=INK2)
            if s["hi_val"] - b > thr:
                ax.text(s["hi"] + 0.015 * rng, y, s["hi_lab"], ha="left", va="center", fontsize=9, color=INK2)
        ax.axvline(b, color=INK, linewidth=1.2, zorder=4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([ROW_TITLES[f] for f in order], color=INK, fontsize=10)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10, fontweight="bold")
        ax.tick_params(colors=INK2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(min(allv) - 0.45 * rng, max(allv) + 0.45 * rng)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("bigger = Grouped does better", fontsize=9, color=INK2, labelpad=6)
    fig.suptitle("Grouped vs. First-come-first-served: how much better is Grouped, and what changes that?", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.062, BASELINE_TEXT, fontsize=8.8, color=INK2, ha="left")
    fig.text(0.01, 0.036, "Each bar changes ONE thing and re-runs 30 simulated 40-hour days (both strategies get the same jobs). "
             "Lateness = how long after its promised time a job finishes; jobs on time count as 0.", fontsize=8.8, color=INK2, ha="left")
    fig.text(0.01, 0.010, "First-come-first-served cannot keep up above about 3 jobs/hour, so its lateness keeps growing and depends on the 40-hour length of the simulation.",
             fontsize=8.8, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(RES / "fig_tornado.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
