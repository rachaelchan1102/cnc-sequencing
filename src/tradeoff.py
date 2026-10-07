"""Step 6: the changeover-vs-lateness tradeoff curve.

For each load level (balanced mix, k = 4) we run, on IDENTICAL jobs (common random numbers, 30 replications):
  * the three simple rules: FCFS, EDD, Grouped
  * the OR-Tools-style planner ("opt") at several values of lam, where
        lam = how many seconds of lateness we are willing to accept to save ONE second of changeover
    lam = 0 ignores changeovers; large lam groups products as hard as possible.
The planner here is the fast exact-per-product DP (optimize.plan_order_dp) so that it can see the whole queue
(window 30). The CP-SAT model (optimize.plan_order) is the general reference; src/check_optimize.py compares them.

Outputs: results/tradeoff_results.csv, results/fig_tradeoff.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc  # noqa: E402
import simulate as sim  # noqa: E402

RES = sc.RES
LAMS = [0, 0.3, 1, 3, 10, 30, 100, 300]
LEVELS = ["light", "medium", "heavy"]
WINDOW = 30
N_REP = 30
K = 4
MIX = "balanced"
METRICS = ["changeover_s_per_h", "avg_tardiness_s", "max_tardiness_s", "pct_late", "unfinished_jobs", "utilization"]


def main():
    emp = sc.load_empirical()
    proc, chg = emp
    rows, reps = [], []
    for level in LEVELS:
        lam_rate = sc.load_levels(sc.MIXES[MIX], proc, chg)[level]
        plans = [("fcfs", None, {}), ("edd", None, {}), ("grouped", None, {})]
        plans += [("opt", lam, dict(opt_lam=lam, opt_window=WINDOW, opt_solver="dp")) for lam in LAMS]
        for s, lam, opts in plans:
            df = sim.run_reps(MIX, lam_rate, K, s, n_rep=N_REP, emp=emp, **opts)
            summ = sim.summarize(df)
            reps.append(df.assign(level=level, strategy=s, lam=lam if lam is not None else np.nan))
            rows.append({"level": level, "jobs_per_h": round(lam_rate, 2), "strategy": s,
                         "lam": lam if lam is not None else np.nan,
                         **{m: summ[m] for m in METRICS}, **{m + "_ci95": summ[m + "_ci95"] for m in METRICS}})
            print(level, s, lam, round(summ["changeover_s_per_h"]), round(summ["avg_tardiness_s"]), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "tradeoff_results.csv", index=False)
    pd.concat(reps).to_csv(RES / "tradeoff_reps.csv", index=False)  # every replication, for paired comparisons
    return res


if __name__ == "__main__":
    main()


def paired_vs_grouped(reps, level, lam, metric):
    """mean and 95% half-width of (opt at lam) minus (grouped), replication by replication (same jobs)."""
    from scipy import stats
    a = reps[(reps.level == level) & (reps.strategy == "opt") & (reps.lam == lam)].sort_values("rep")[metric].to_numpy()
    b = reps[(reps.level == level) & (reps.strategy == "grouped")].sort_values("rep")[metric].to_numpy()
    d = a - b
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


def plot(res=None):
    """One panel per load level. x = time lost to changeovers, y = average lateness. Lower-left is best."""
    res = res if res is not None else pd.read_csv(RES / "tradeoff_results.csv")
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
    BLUE, RED, GREEN, ORANGE = "#2a78d6", "#d03b3b", "#2a9d6f", "#e08a1e"
    names = {"light": "Light load", "medium": "Medium load", "heavy": "Heavy load"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURF)
    for ax, lvl in zip(axes, LEVELS):
        d = res[res.level == lvl]
        ax.set_facecolor(SURF)
        o = d[d.strategy == "opt"].sort_values("lam")
        x, y = o.changeover_s_per_h / 60, o.avg_tardiness_s / 3600
        ax.plot(x, y, "-o", color=BLUE, ms=5, lw=1.8, zorder=3, label="OR-Tools-style planner (dial setting)")
        prev = None
        for _, r in o.iterrows():  # label each dial setting, skip ones that land on the same spot
            pt = (round(r.changeover_s_per_h / 60, 1), round(r.avg_tardiness_s / 3600, 2))
            if pt == prev or r.lam not in (0, 1, 10, 100):
                continue
            prev = pt
            ax.annotate(f"{r.lam:g}", (r.changeover_s_per_h / 60, r.avg_tardiness_s / 3600), textcoords="offset points",
                        xytext=(5, 6), fontsize=8.5, color=BLUE)
        for strat, col, lab in (("fcfs", RED, "First come, first served"), ("edd", ORANGE, "Earliest due date first"),
                                ("grouped", GREEN, "Grouped (stay on a product)")):
            r = d[d.strategy == strat].iloc[0]
            ax.errorbar(r.changeover_s_per_h / 60, r.avg_tardiness_s / 3600,
                        xerr=r.changeover_s_per_h_ci95 / 60, yerr=r.avg_tardiness_s_ci95 / 3600, fmt="s", color=col,
                        ms=9, capsize=3, zorder=4, label=lab)
        ax.set_title(f"{names[lvl]} ({d.jobs_per_h.iloc[0]:.1f} new jobs per hour)", fontsize=12, color=INK, loc="left")
        ax.set_xlabel("Time lost to changeovers (minutes per hour)", color=INK2)
        ax.set_ylabel("Average lateness per job (hours)", color=INK2)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.tick_params(colors=INK2)
    axes[1].legend(loc="center left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, 0.55))
    fig.suptitle("Switching less vs finishing on time: where each way of choosing the next job lands (lower-left is best)",
                 fontsize=13.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             "Numbers on the blue line = the dial: how many seconds of lateness the planner will accept to save 1 second of switching "
             "(0 = ignore switching; 300 = switching matters most).\n"
             "Lateness = how long after its promised time a job finishes; jobs on time count as 0. 30 simulated runs per point on identical jobs; "
             "whiskers = 95% confidence intervals. Assumed demand, real run and changeover times.",
             fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(RES / "fig_tradeoff.png", dpi=150, facecolor=SURF)


# ---------------------------------------------------------------------------------------------------------------
# Simpler figures (replace the crowded scatter above). Two figures:
#   fig_tradeoff.png : three ways of choosing the next job, side by side, at each load (bars)
#   fig_dial.png     : what turning the planner's dial does (medium and heavy load)
# ---------------------------------------------------------------------------------------------------------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
GRAY, BLUE, GREEN = "#8a8985", "#2a78d6", "#2a9d6f"
LOAD_NAME = {"light": "Light load", "medium": "Medium load", "heavy": "Heavy load"}
PLANNER_DIAL = 30


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, length=0)


def plot_rules(res=None):
    res = res if res is not None else pd.read_csv(RES / "tradeoff_results.csv")
    series = [("First come, first served", GRAY, lambda d: d[d.strategy == "fcfs"].iloc[0]),
              ("Grouped (stay on a product)", BLUE, lambda d: d[d.strategy == "grouped"].iloc[0]),
              (f"OR-Tools planner (dial {PLANNER_DIAL})", GREEN,
               lambda d: d[(d.strategy == "opt") & (d.lam == PLANNER_DIAL)].iloc[0])]
    panels = [("changeover_s_per_h", 60, "How much of each hour is lost to switching products?", "minutes lost per hour", "{:.0f}"),
              ("avg_tardiness_s", 3600, "How late is the average job?", "hours late per job (on-time jobs count 0)", "{:.1f}")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURF)
    w = 0.26
    for ax, (col, div, title, ylab, fmt) in zip(axes, panels):
        _style(ax)
        for si, (name, color, getter) in enumerate(series):
            for li, lvl in enumerate(LEVELS):
                r = getter(res[res.level == lvl])
                x = li + (si - 1) * (w + 0.03)
                v, ci = r[col] / div, r[col + "_ci95"] / div
                ax.bar(x, v, w, color=color, label=name if li == 0 else None, zorder=3)
                ax.errorbar(x, v, yerr=ci, color=INK2, lw=1, capsize=2.5, zorder=4)
                ax.text(x, v + ci + ax.get_ylim()[1] * 0.01, fmt.format(v), ha="center", va="bottom",
                        fontsize=9.5, color=INK, zorder=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{LOAD_NAME[l]}\n{res[res.level == l].jobs_per_h.iloc[0]:.1f} jobs/h" for l in LEVELS],
                           color=INK, fontsize=10.5)
        ax.set_ylabel(ylab, color=INK2)
        ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3, frameon=False, fontsize=10.5)
    fig.suptitle("Staying on one product cuts switching time and lateness a lot; the OR-Tools planner adds only a little",
                 fontsize=14, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             "Shorter bars are better. Average of 30 simulated 40-hour runs on identical jobs; thin lines = 95% confidence intervals. "
             "Demand is assumed; run and changeover times are real.\n"
             "First come, first served falls further behind the longer it runs, so read its lateness as 'much worse', not an exact number.\n"
             "Earliest-due-date-first looks almost the same as first come, first served and is left out.",
             fontsize=8.8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.17, 1, 0.93))
    fig.savefig(RES / "fig_tradeoff.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def plot_dial(res=None):
    res = res if res is not None else pd.read_csv(RES / "tradeoff_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), facecolor=SURF, sharex=True)
    cols = [("changeover_s_per_h", 60, "Switching time\n(minutes lost per hour)"),
            ("avg_tardiness_s", 3600, "Average lateness\n(hours per job)")]
    for ri, lvl in enumerate(("medium", "heavy")):
        d = res[res.level == lvl]
        o = d[d.strategy == "opt"].sort_values("lam")
        g = d[d.strategy == "grouped"].iloc[0]
        for ci, (col, div, ylab) in enumerate(cols):
            ax = axes[ri][ci]
            _style(ax)
            xs = range(len(o))
            ax.plot(xs, o[col] / div, "-o", color=GREEN, lw=2, ms=6, zorder=3)
            ax.axhline(g[col] / div, color=BLUE, lw=1.8, ls="--", zorder=2)
            if ri == 0 and ci == 0:
                ax.plot([], [], "-o", color=GREEN, lw=2, label="OR-Tools planner")
                ax.plot([], [], "--", color=BLUE, lw=1.8, label="Grouped (for comparison)")
            ax.set_ylim(0, None)
            ax.set_ylabel(ylab, color=INK2)
            if ci == 0:
                ax.set_title(f"{LOAD_NAME[lvl]} ({d.jobs_per_h.iloc[0]:.1f} new jobs/hour)", fontsize=12, color=INK, loc="left", pad=8)
    for ax in axes[1]:
        ax.set_xticks(range(len(LAMS))); ax.set_xticklabels([f"{l:g}" for l in LAMS], color=INK)
        ax.set_xlabel("The dial (higher = switching matters more)", color=INK2)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False, fontsize=10.5)
    fig.suptitle("What turning the planner's dial does to switching time and lateness",
                 fontsize=14, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "The dial = how many seconds of lateness the planner will accept to save 1 second of switching. Dial 0: it ignores switching. "
             "High dial: it behaves like Grouped.\nLower is better in every panel. 30 simulated runs per point.", fontsize=8.8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(RES / "fig_dial.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def plot(res=None):
    plot_rules(res)
    plot_dial(res)
