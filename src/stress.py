"""Step 7: three stress tests of the Grouped result (asked for by Rachael after Step 6).

 A. BREAK-EVEN: shrink every changeover (to 0%..100% of its real average length) with demand held fixed.
    Question: how short must changeovers get before grouping stops being worth it?
 B. TIGHT DUE DATES: slack k from 0.5 (very tight) to 8 (loose).
    Question: does Grouped's lateness advantage survive when promises are tight?
 C. STARVATION: lateness of each PRODUCT separately (worst job in each run, averaged over runs).
    Question: does Grouped look good on average while one product waits for hours?

Strategies in every test: FCFS, EDD, Grouped, Grouped with a cap of 5 in a row, and the OR-Tools-style planner (DP version, dial 30,
window 30). 30 replications on identical jobs (common random numbers).
Outputs: results/stress_breakeven.csv, stress_tight.csv, stress_starvation.csv, fig_breakeven.png, fig_tight_due.png, fig_starvation.png
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
N_REP, SEED0 = 30, 1000
DIAL, WINDOW = 30, 30
STRATS = {  # label -> (strategy, extra options)
    "First come, first served": ("fcfs", {}),
    "Earliest due date first": ("edd", {}),
    "Grouped": ("grouped", {}),
    "Grouped, max 5 in a row": ("grouped", {"max_batch": 5}),
    "OR-Tools planner": ("opt", {"opt_lam": DIAL, "opt_window": WINDOW, "opt_solver": "dp"}),
}
MULTS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
KS = [1, 2, 4, 8, 16]  # slack in job-cycles (~20 min each)
METRICS = ["changeover_s_per_h", "avg_tardiness_s", "max_tardiness_s", "pct_late", "unfinished_jobs", "utilization"]


def reps(mix, lam, k, label, extra, emp, **over):
    strat, opts = STRATS[label]
    return sim.run_reps(mix, lam, k, strat, n_rep=N_REP, seed0=SEED0, emp=emp, **{**opts, **extra, **over})


def ci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def breakeven(emp):
    proc, chg = emp
    rows = []
    for level in ("medium", "heavy"):
        lam = sc.load_levels(sc.MIXES["balanced"], proc, chg)[level]
        for mult in MULTS:
            res = {lab: reps("balanced", lam, 4, lab, {}, emp, chg_mult=mult) for lab in STRATS}
            for lab, df in res.items():
                rows.append({"level": level, "jobs_per_h": round(lam, 2), "chg_pct_of_real": int(mult * 100), "strategy": lab,
                             **{m: df[m].mean() for m in METRICS}, **{m + "_ci95": ci(df[m]) for m in METRICS}})
            d = (res["Grouped"].avg_tardiness_s - res["Earliest due date first"].avg_tardiness_s).to_numpy()
            rows.append({"level": level, "jobs_per_h": round(lam, 2), "chg_pct_of_real": int(mult * 100),
                         "strategy": "PAIRED Grouped minus EDD (lateness s)", "avg_tardiness_s": d.mean(), "avg_tardiness_s_ci95": ci(d)})
            print("breakeven", level, mult, round(d.mean()), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "stress_breakeven.csv", index=False)
    return out


def tight(emp):
    """Tight-due-date test at two loads: 'below' (0.7 x the rate FCFS can sustain: EDD/FCFS are stable, so the test is
    purely about due dates) and 'medium' (EDD/FCFS are overloaded there, so they lose partly for capacity reasons)."""
    proc, chg = emp
    rows = []
    mix = sc.MIXES["balanced"]
    lams = {"below": 0.7 * sc.load_window(mix, proc, chg)["lam_lo"], "medium": sc.load_levels(mix, proc, chg)["medium"]}
    for level, lam in lams.items():
        for k in KS:
            res = {lab: reps("balanced", lam, k, lab, {}, emp) for lab in STRATS}
            for lab, df in res.items():
                rows.append({"level": level, "jobs_per_h": round(lam, 2), "k": k, "strategy": lab,
                             **{m: df[m].mean() for m in METRICS}, **{m + "_ci95": ci(df[m]) for m in METRICS}})
            d = (res["Grouped"].avg_tardiness_s - res["Earliest due date first"].avg_tardiness_s).to_numpy()
            rows.append({"level": level, "k": k, "strategy": "PAIRED Grouped minus EDD (lateness s)",
                         "avg_tardiness_s": d.mean(), "avg_tardiness_s_ci95": ci(d)})
            print("tight", level, k, round(d.mean()), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "stress_tight.csv", index=False)
    return out


def starvation(emp):
    proc, chg = emp
    rows = []
    for mix in ("balanced", "A-heavy"):
        lam = sc.load_levels(sc.MIXES[mix], proc, chg)["medium"]
        for lab, (strat, opts) in STRATS.items():
            per = {p: {"max": [], "mean": [], "n": []} for p in "ABC"}
            for r in range(N_REP):
                seed = SEED0 + r
                jobs = sc.generate_jobs(mix, lam, 4, seed, emp=emp)
                _, d = sim.simulate(jobs, strat, chg, seed, return_detail=True, **opts)
                for p in "ABC":
                    m = d["counted"] & (d["prod"] == p)
                    per[p]["n"].append(int(m.sum()))
                    per[p]["max"].append(d["tard"][m].max() if m.any() else np.nan)
                    per[p]["mean"].append(d["tard"][m].mean() if m.any() else np.nan)
            for p in "ABC":
                rows.append({"mix": mix, "jobs_per_h": round(lam, 2), "strategy": lab, "product": p,
                             "jobs_per_run": np.mean(per[p]["n"]),
                             "worst_job_lateness_s": np.nanmean(per[p]["max"]), "worst_job_lateness_s_ci95": ci(np.array(per[p]["max"])),
                             "avg_lateness_s": np.nanmean(per[p]["mean"]), "avg_lateness_s_ci95": ci(np.array(per[p]["mean"]))})
            print("starv", mix, lab, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "stress_starvation.csv", index=False)
    return out


INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
COL = {"Earliest due date first": "#8a8985", "Grouped": "#2a78d6", "OR-Tools planner": "#2a9d6f", "Grouped, max 5 in a row": "#e08a1e"}
LEVEL_NAME = {"below": "Below the load window", "medium": "Medium load", "heavy": "Heavy load"}


def _style(ax, grid="y"):
    ax.set_facecolor(SURF)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, length=0)


def crossover(x, d):
    """first place the paired difference (Grouped minus EDD) crosses from positive to negative, by straight-line interpolation."""
    for i in range(len(x) - 1):
        if d[i] > 0 >= d[i + 1]:
            return x[i] + (x[i + 1] - x[i]) * d[i] / (d[i] - d[i + 1])
    return None


def plot_breakeven(b=None):
    b = b if b is not None else pd.read_csv(RES / "stress_breakeven.csv")
    labs = ["Earliest due date first", "Grouped", "OR-Tools planner"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURF)
    for ax, lvl in zip(axes, ("medium", "heavy")):
        _style(ax)
        d = b[b.level == lvl]
        for lab in labs:
            r = d[d.strategy == lab].sort_values("chg_pct_of_real")
            ax.plot(r.chg_pct_of_real, r.avg_tardiness_s / 3600, "-o", color=COL[lab], lw=2, ms=5, label=lab, zorder=3)
        pr = d[d.strategy.str.startswith("PAIRED")].sort_values("chg_pct_of_real")
        x0 = crossover(pr.chg_pct_of_real.to_numpy(float), pr.avg_tardiness_s.to_numpy(float))
        if x0 is not None:
            ax.axvline(x0, color=INK2, lw=1, ls=":")
            ax.annotate(f"break-even about {x0:.0f}%\nof real changeover length", (x0, ax.get_ylim()[1] * 0.55), xytext=(8, 0),
                        textcoords="offset points", fontsize=10, color=INK)
        ax.set_xlabel("Changeover length (% of the real measured length)", color=INK2)
        ax.set_ylabel("Average lateness (hours per job)", color=INK2)
        ax.set_title(f"{LEVEL_NAME[lvl]} ({d.jobs_per_h.iloc[0]:.1f} new jobs/hour)", fontsize=12.5, color=INK, loc="left", pad=10)
        ax.set_xlim(-2, 102); ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    fig.suptitle("How short must changeovers get before staying on one product stops paying off?", fontsize=14, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Demand is held fixed while changeovers shrink. Lower is better. Left of the dotted line, simply doing the earliest due date first wins; "
             "right of it, Grouped wins.\n30 simulated runs per point. Assumed demand; real run and changeover times (scaled).",
             fontsize=8.8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(RES / "fig_breakeven.png", dpi=150, facecolor=SURF); plt.close(fig)


def plot_tight(t=None):
    t = t if t is not None else pd.read_csv(RES / "stress_tight.csv")
    labs = ["Earliest due date first", "Grouped", "OR-Tools planner"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), facecolor=SURF, sharex=True)
    for ri, lvl in enumerate(("below", "medium")):
        d = t[t.level == lvl]
        for ci_, (col, div, ylab) in enumerate([("pct_late", 0.01, "Jobs finishing late (%)"), ("avg_tardiness_s", 3600, "Average lateness (hours per job)")]):
            ax = axes[ri][ci_]; _style(ax)
            for lab in labs:
                r = d[d.strategy == lab].sort_values("k")
                ax.plot(range(len(r)), r[col] / div, "-o", color=COL[lab], lw=2, ms=5, label=lab, zorder=3)
            ax.set_ylabel(ylab, color=INK2); ax.set_ylim(0, None)
            if ci_ == 0:
                ax.set_title(f"{LEVEL_NAME[lvl]} ({d.jobs_per_h.iloc[0]:.1f} new jobs/hour)", fontsize=12, color=INK, loc="left", pad=8)
    for ax in axes[1]:
        ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"{k:g}" for k in KS], color=INK)
        ax.set_xlabel("Due-date slack (left = very tight promises, right = loose)", color=INK2)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3, frameon=False, fontsize=10.5)
    fig.suptitle("Does Grouped still help when due dates are tight?", fontsize=14, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             "Slack = how many job-cycles (average run time + average changeover, about 20 min each) a customer allows beyond the job's own run time;\n"
             "1 = about 20 min, 16 = about 5 hours. Lower is better in every panel. 30 simulated runs per point.\n"
             "At medium load, earliest-due-date-first is overloaded, so part of its gap to Grouped is capacity, not due dates; the top row is the cleaner test.",
             fontsize=8.6, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(RES / "fig_tight_due.png", dpi=150, facecolor=SURF); plt.close(fig)


def plot_starvation(sv=None):
    sv = sv if sv is not None else pd.read_csv(RES / "stress_starvation.csv")
    labs = ["Earliest due date first", "Grouped", "Grouped, max 5 in a row", "OR-Tools planner"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURF, sharey=True)
    w = 0.2
    for ax, mix in zip(axes, ("balanced", "A-heavy")):
        _style(ax)
        d = sv[sv.mix == mix]
        for si, lab in enumerate(labs):
            for pi, p in enumerate("ABC"):
                r = d[(d.strategy == lab) & (d["product"] == p)].iloc[0]
                x = pi + (si - 1.5) * (w + 0.02); v = r.worst_job_lateness_s / 3600; c = r.worst_job_lateness_s_ci95 / 3600
                ax.bar(x, v, w, color=COL[lab], label=lab if pi == 0 else None, zorder=3)
                ax.errorbar(x, v, yerr=c, color=INK2, lw=1, capsize=2, zorder=4)
                ax.text(x, v + c + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        names = {"balanced": "Even mix (A, B, C equally common)", "A-heavy": "A-heavy mix (60% A, 20% B, 20% C)"}
        ax.set_title(f"{names[mix]}, medium load", fontsize=12, color=INK, loc="left", pad=10)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"Product {p}" for p in "ABC"], color=INK, fontsize=10.5)
    axes[0].set_ylabel("Worst job of that product, hours late\n(average over runs)", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Does staying on one product leave any product waiting for hours?", fontsize=14, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Lower is better. Each bar = the single latest job of that product in a 40-hour run, averaged over 30 runs; thin lines = 95% confidence intervals. "
             "Unfinished jobs count as late up to the end of the run.", fontsize=8.8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(RES / "fig_starvation.png", dpi=150, facecolor=SURF); plt.close(fig)


def plot_all():
    plot_breakeven(); plot_tight(); plot_starvation()


if __name__ == "__main__":
    which = sys.argv[1:] or ["breakeven", "tight", "starvation"]
    emp = sc.load_empirical()
    for w in which:
        {"breakeven": breakeven, "tight": tight, "starvation": starvation}[w](emp)
    plot_all()
