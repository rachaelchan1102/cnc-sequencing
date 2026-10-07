"""Step 5: SimPy single-machine model.

One CNC machine, one waiting line. Jobs come from scenarios.generate_jobs (assumed demand, real run times).
When the next job's product differs from the previous one, the machine first does a changeover whose duration is
drawn from the REAL observed changeovers for that transition (matrices 3-5).

Strategies = rules for "which waiting job next":
  fcfs    : oldest arrival first
  edd     : earliest due date first
  grouped : stay on the current product while any of it is waiting (earliest due within it); when none is
            waiting, take the earliest-due job. Optional max_batch caps consecutive same-product jobs.

Measurement
  * First WARMUP_H hours are discarded. Job metrics use jobs that ARRIVE after warm-up; resource metrics
    (changeover time, utilization, WIP) use the time window [warm-up, horizon].
  * Jobs unfinished at the end are NOT dropped: their tardiness/flow are lower bounds measured to the end time.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import simpy
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc  # noqa: E402

WARMUP_H = 4.0
STRATEGIES = ("fcfs", "edd", "grouped", "opt")


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def simulate(jobs, strategy, chg, seed, horizon_h=sc.HORIZON_H, warmup_h=WARMUP_H,
             chg_scale=1.0, max_batch=None, return_detail=False, opt_lam=1.0, opt_window=8,
             opt_time_s=0.5, opt_solver="cpsat", chg_mult=1.0):
    assert strategy in STRATEGIES, strategy
    n = len(jobs)
    arr = jobs.arrival_s.to_numpy(float)
    prod = jobs["product"].to_numpy()
    proc = jobs.proc_s.to_numpy(float)
    due = jobs.due_s.to_numpy(float)
    T_end, T_w = horizon_h * 3600.0, warmup_h * 3600.0

    # one independent random stream per transition (so changeover draws are comparable across strategies)
    streams = {t: np.random.default_rng([seed, 77, i]) for i, t in enumerate(sorted(chg))}

    env = simpy.Environment()
    waiting, st = [], {"wake": None}
    finish = np.full(n, np.nan)
    co_iv, pr_iv = [], []  # (start, planned_end) of changeovers / processing

    if strategy == "opt":
        import optimize as op  # OR-Tools planner (Step 6); imported only when needed
        chg_mean = {k_: chg_mult * v for k_, v in op.chg_dict(sc.mean_changeover_matrix(chg)).items()}  # planner knows the (scaled) average
        n_plan = {"solves": 0, "not_optimal": 0}

    def pick(cur, consec):
        if strategy == "opt":
            # the opt_window waiting jobs with earliest due dates are sequenced; only the first is run
            cand = sorted(waiting, key=lambda j: (due[j], j))[:opt_window]
            args = (env.now, cur, [prod[j] for j in cand], [proc[j] for j in cand], [due[j] for j in cand],
                    chg_mean, opt_lam)
            if opt_solver == "dp":
                order, info = op.plan_order_dp(*args)
            else:
                order, info = op.plan_order(*args, time_limit_s=opt_time_s)
            n_plan["solves"] += 1
            n_plan["not_optimal"] += info["status"] not in ("OPTIMAL", "TRIVIAL")
            return cand[order[0]]
        if strategy == "fcfs":
            return min(waiting, key=lambda j: (arr[j], j))
        if strategy == "edd":
            return min(waiting, key=lambda j: (due[j], j))
        if cur is None:
            pool = waiting
        else:
            same = [j for j in waiting if prod[j] == cur]
            other = [j for j in waiting if prod[j] != cur]
            if same and (max_batch is None or consec < max_batch or not other):
                pool = same
            elif other:
                pool = other
            else:
                pool = same
        return min(pool, key=lambda j: (due[j], j))

    def arrivals():
        for i in range(n):
            yield env.timeout(max(0.0, arr[i] - env.now))
            waiting.append(i)
            w = st["wake"]
            if w is not None and not w.triggered:
                w.succeed()

    def machine():
        cur, consec = None, 0
        while True:
            if not waiting:
                st["wake"] = env.event()
                yield st["wake"]
                st["wake"] = None
                continue
            i = pick(cur, consec)
            waiting.remove(i)
            if cur is not None and prod[i] != cur:
                key = f"{cur}->{prod[i]}"
                obs = chg[key]
                x = streams[key].choice(obs)
                cs = chg_mult * max(0.0, obs.mean() + chg_scale * (x - obs.mean()))  # chg_mult shrinks/stretches the MEAN level
                co_iv.append((env.now, env.now + cs))
                yield env.timeout(cs)
                consec = 1
            else:
                consec += 1
            pr_iv.append((env.now, env.now + proc[i]))
            yield env.timeout(proc[i])
            finish[i] = env.now
            cur = prod[i]

    env.process(arrivals())
    env.process(machine())
    env.run(until=T_end)

    W = T_end - T_w
    end_eff = np.where(np.isnan(finish), T_end, finish)
    tard = np.maximum(0.0, end_eff - due)
    flow = end_eff - arr
    counted = arr >= T_w
    co_time = sum(_overlap(a, b, T_w, T_end) for a, b in co_iv)
    pr_time = sum(_overlap(a, b, T_w, T_end) for a, b in pr_iv)
    wip = sum(_overlap(arr[i], end_eff[i], T_w, T_end) for i in range(n)) / W
    done_in_window = np.sum((~np.isnan(finish)) & (finish >= T_w))
    m = {
        "changeover_s_per_h": co_time / (W / 3600.0),
        "n_changeovers": sum(1 for a, _ in co_iv if T_w <= a < T_end),
        "avg_tardiness_s": float(tard[counted].mean()) if counted.any() else np.nan,
        "max_tardiness_s": float(tard[counted].max()) if counted.any() else np.nan,  # worst single job
        "pct_late": float((tard[counted] > 0).mean()) if counted.any() else np.nan,
        "avg_flow_s": float(flow[counted].mean()) if counted.any() else np.nan,
        "utilization": (co_time + pr_time) / W,
        "wip": wip,
        "throughput_per_h": done_in_window / (W / 3600.0),
        "unfinished_jobs": int(np.sum(np.isnan(finish) & counted)),
        "jobs_counted": int(counted.sum()),
    }
    if return_detail:
        return m, {"finish": finish.copy(), "co_total_s": sum(b - a for a, b in co_iv),
                   "co_events": len(co_iv), "jobs_done": int((~np.isnan(finish)).sum()),
                   "plan": n_plan if strategy == "opt" else None,
                   "tard": tard, "flow": flow, "counted": counted, "prod": prod}
    return m


def run_reps(mix, lam, k, strategy, n_rep=30, seed0=1000, emp=None, **opts):
    emp = emp or sc.load_empirical()
    _, chg = emp
    rows = []
    for r in range(n_rep):
        seed = seed0 + r
        jobs = sc.generate_jobs(mix, lam, k, seed, emp=emp)
        rows.append({"rep": r, "seed": seed, **simulate(jobs, strategy, chg, seed, **opts)})
    return pd.DataFrame(rows)


def summarize(df):
    """mean and 95% confidence half-width (t distribution) per metric."""
    out = {}
    n = len(df)
    tq = stats.t.ppf(0.975, n - 1)
    for c in df.columns:
        if c in ("rep", "seed"):
            continue
        out[c] = df[c].mean()
        out[c + "_ci95"] = tq * df[c].std(ddof=1) / np.sqrt(n)
    return out
