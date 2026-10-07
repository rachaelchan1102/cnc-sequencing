"""Step 3: quantify processing times and changeover times from results/events.csv.

Outputs (all in results/):
  processing_stats.csv        stats per product (all runs, and runs without irregular seconds)
  changeover_stats.csv        stats per transition under several matrix/irregular-second choices
  changeover_matrix_mean.csv  3x3 matrix (from-product x to-product) of the primary mean changeover time
  fig_processing_times.png    processing-time distribution per product
  fig_changeover_heatmap.png  changeover mean per transition (primary choice)

Decisions (made by Rachael, see docs/PROJECT_LOG.md):
  PRIMARY  = matrices 3-5 only (authors: matrix 1, probably 2, show a learning-curve effect),
             irregular seconds KEPT IN (real plants have delays).
  Sensitivity variants reported beside it: irregular seconds removed, matrices 2-5, all matrices.

Honest caveat: with n = 3 observations per transition a bootstrap has only 10 distinct resamples
(multisets), so its interval is essentially the range of the data. We still report it, plus the min/max.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
SEED, B = 20261006, 10_000
PRODUCTS = ["A", "B", "C"]
TRANSITIONS = ["A->B", "A->C", "B->A", "B->C", "C->A", "C->B"]

VARIANTS = {  # name -> (matrices kept, duration column)
    "m3-5 (primary)":                   ([3, 4, 5],       "duration_s"),
    "m3-5, irregular seconds removed":  ([3, 4, 5],       "regular_s"),
    "m2-5":                             ([2, 3, 4, 5],    "duration_s"),
    "m1-5 (all)":                       ([1, 2, 3, 4, 5], "duration_s"),
}

# Palette (reference blue ramp from the dataviz skill; light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE = {"100": "#cde2fb", "250": "#86b6ef", "450": "#2a78d6", "600": "#184f95", "700": "#0d366b"}


def summarize(x, rng):
    x = np.asarray(x, dtype=float)
    boots = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(x), "mean": x.mean(), "median": float(np.median(x)),
            "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
            "min": x.min(), "max": x.max(), "boot95_mean_lo": lo, "boot95_mean_hi": hi}


def processing_stats(ev, rng):
    pr = ev[ev.event == "production"]
    rows = []
    for variant, d in [("all runs", pr), ("runs without irregular seconds", pr[pr.irregular_s == 0])]:
        for p in PRODUCTS:
            rows.append({"product": p, "variant": variant, **summarize(d[d.new_product == p].duration_s, rng)})
    return pd.DataFrame(rows)


def changeover_stats(ev, rng):
    co = ev[ev.event == "changeover"]
    rows = []
    for name, (mats, col) in VARIANTS.items():
        d = co[co.matrix.isin(mats)]
        for t in TRANSITIONS:
            rows.append({"transition": t, "variant": name, **summarize(d[d.transition == t][col], rng)})
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_processing(ev):
    pr = ev[ev.event == "production"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), facecolor=SURFACE)
    for ax, p in zip(axes, PRODUCTS):
        d = pr[pr.new_product == p]
        style_axes(ax)
        ax.boxplot(d.duration_s, positions=[0], widths=0.45, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=BLUE["100"], edgecolor=BLUE["450"], linewidth=1.5),
                   medianprops=dict(color=BLUE["600"], linewidth=2),
                   whiskerprops=dict(color=BLUE["450"], linewidth=1.5), capprops=dict(color=BLUE["450"], linewidth=1.5))
        rng = np.random.default_rng(1)
        x = rng.uniform(-0.12, 0.12, len(d))
        reg = d.irregular_s.values == 0
        ax.scatter(x[reg], d.duration_s.values[reg], s=34, color=BLUE["450"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.scatter(x[~reg], d.duration_s.values[~reg], s=44, facecolor=SURFACE, edgecolor=BLUE["600"], linewidth=1.8, zorder=3)
        ax.set_xticks([0]); ax.set_xticklabels([f"Product {p}"], color=INK, fontsize=11)
        ax.set_xlim(-0.6, 0.6)
        lo, hi = d.duration_s.min(), d.duration_s.max()
        pad = (hi - lo) * 0.15 or 5
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"median {d.duration_s.median():.0f} s  (n={len(d)})", fontsize=10, color=INK2, pad=8)
        if p == "A":
            ax.set_ylabel("Production run time (s)", color=INK2)
    fig.suptitle("Processing time per product, all 5 matrices", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.015, "Each panel has its own y-scale (A is ~15x shorter than C).  Filled dot = regular run;  open ring = run containing irregular seconds.",
             fontsize=8.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(RES / "fig_processing_times.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_heatmap(stats):
    s = stats[stats.variant == "m3-5 (primary)"].set_index("transition")
    cmap = LinearSegmentedColormap.from_list("blue", [BLUE["100"], BLUE["250"], BLUE["450"], BLUE["600"], BLUE["700"]])
    vmin, vmax = s["mean"].min(), s["mean"].max()
    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, frm in enumerate(PRODUCTS):
        for j, to in enumerate(PRODUCTS):
            t = f"{frm}->{to}"
            if frm == to:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0efec", edgecolor=SURFACE, linewidth=3))
                ax.text(j, i, "—", ha="center", va="center", color=INK2, fontsize=14)
                continue
            r = s.loc[t]
            frac = (r["mean"] - vmin) / (vmax - vmin)
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=cmap(frac), edgecolor=SURFACE, linewidth=3))
            tc = "white" if frac > 0.45 else INK
            ax.text(j, i - 0.12, f"{r['mean']:.0f} s", ha="center", va="center", color=tc, fontsize=17, fontweight="bold")
            ax.text(j, i + 0.2, f"range {r['min']:.0f}–{r['max']:.0f}  (n={int(r['n'])})", ha="center", va="center", color=tc, fontsize=8.5)
    ax.set_xlim(-.5, 2.5); ax.set_ylim(2.5, -.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"to {p}" for p in PRODUCTS], color=INK, fontsize=11)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"from {p}" for p in PRODUCTS], color=INK, fontsize=11)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.suptitle("Mean changeover time by transition", x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.02, 0.90, "Matrices 3–5 only, irregular seconds included. Darker = longer.", fontsize=9.5, color=INK2, ha="left")
    fig.text(0.02, 0.02, "Only 3 observations per cell: treat differences smaller than the ranges shown as unproven.\nBootstrap intervals and sensitivity variants: results/changeover_stats.csv", fontsize=8.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(RES / "fig_changeover_heatmap.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main():
    ev = pd.read_csv(RES / "events.csv", parse_dates=["start_time", "end_time"])
    rng = np.random.default_rng(SEED)
    ps, cs = processing_stats(ev, rng), changeover_stats(ev, rng)
    ps.round(1).to_csv(RES / "processing_stats.csv", index=False)
    cs.round(1).to_csv(RES / "changeover_stats.csv", index=False)
    prim = cs[cs.variant == "m3-5 (primary)"].set_index("transition")
    mat = pd.DataFrame(index=[f"from {p}" for p in PRODUCTS], columns=[f"to {p}" for p in PRODUCTS], dtype=float)
    for t in TRANSITIONS:
        f, to = t.split("->")
        mat.loc[f"from {f}", f"to {to}"] = prim.loc[t, "mean"]
    mat.round(1).to_csv(RES / "changeover_matrix_mean.csv")
    fig_processing(ev)
    fig_heatmap(cs)
    pd.set_option("display.width", 220)
    print("PROCESSING (s)"); print(ps.round(1).to_string(index=False))
    print("\nCHANGEOVER (s), mean by variant:")
    print(cs.pivot(index="transition", columns="variant", values="mean").round(0)[list(VARIANTS)].to_string())
    print("\nPRIMARY with spread:"); print(cs[cs.variant == "m3-5 (primary)"].round(0).drop(columns="variant").to_string(index=False))
    print("\nMEAN MATRIX (primary):"); print(mat.round(0).to_string())


if __name__ == "__main__":
    main()
