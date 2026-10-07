"""Step 6: OR-Tools sequencing of the jobs that are waiting right now.

The idea ("rolling window"): each time the machine becomes free, look at the jobs waiting in line (at most
`window` of them, the ones with the earliest due dates), find the ORDER that minimises

        total lateness of those jobs  +  lam * total changeover seconds

and then run only the FIRST job of that order. When the machine is free again the plan is rebuilt with
whatever has arrived in the meantime.

lam is the dial:  "one second of changeover counts as lam seconds of lateness".
    lam = 0   -> only lateness matters (changeovers are free as far as the solver knows)
    lam large -> only changeovers matter (the solver groups products as hard as it can)

The planner is given the AVERAGE changeover time for each product pair (it cannot know the random draw that
will actually happen); each job's run time is known when the job arrives.

Model: CP-SAT, one Boolean per "job i directly before job j" arc, an AddCircuit constraint so the arcs form
one path through all jobs, and a finish-time variable per job that must be at least (previous finish +
changeover + own run time).
"""
from ortools.sat.python import cp_model

SCALE = 100


def chg_dict(matrix, products="ABC"):
    """3x3 mean changeover matrix (rows = from, cols = to, order A,B,C) -> {'A->B': seconds, ...}"""
    return {f"{a}->{b}": float(matrix[i][j]) for i, a in enumerate(products)
            for j, b in enumerate(products) if a != b}  # objective integers: lateness weight 100, changeover weight round(100*lam)


def plan_order(now, cur, prods, procs, dues, chg_mean, lam, time_limit_s=0.5, workers=1):
    """Return (order, info). order = indices 0..n-1 of the inputs in the planned run order.

    now      : current time in seconds
    cur      : product the machine is currently set up for (None if it has never run)
    prods    : product of each waiting job ('A'/'B'/'C')
    procs    : run time (s) of each waiting job
    dues     : due time (s) of each waiting job
    chg_mean : {'A->B': seconds, ...}
    """
    n = len(prods)
    if n == 1:
        return [0], {"status": "TRIVIAL", "obj": None}

    def co(p, q):
        return 0 if (p is None or p == q) else int(round(chg_mean[f"{p}->{q}"]))

    horizon = int(now + sum(procs) + n * 2000 + 1)
    m = cp_model.CpModel()
    C = [m.NewIntVar(0, horizon, f"C{j}") for j in range(n)]
    T = [m.NewIntVar(0, horizon, f"T{j}") for j in range(n)]
    arcs, chg_terms = [], []
    w_chg = int(round(SCALE * lam))
    for j in range(n):
        m.Add(T[j] >= C[j] - int(round(dues[j])))
        lit = m.NewBoolVar(f"first{j}")             # job j is run first
        arcs.append((0, j + 1, lit))
        c0 = co(cur, prods[j])
        m.Add(C[j] >= int(now) + c0 + int(round(procs[j]))).OnlyEnforceIf(lit)
        chg_terms.append(w_chg * c0 * lit)
        arcs.append((j + 1, 0, m.NewBoolVar(f"last{j}")))  # job j is run last
        for i in range(n):
            if i == j:
                continue
            lit = m.NewBoolVar(f"a{i}_{j}")          # job i immediately before job j
            arcs.append((i + 1, j + 1, lit))
            cij = co(prods[i], prods[j])
            m.Add(C[j] >= C[i] + cij + int(round(procs[j]))).OnlyEnforceIf(lit)
            chg_terms.append(w_chg * cij * lit)
    m.AddCircuit(arcs)
    m.Minimize(SCALE * sum(T) + sum(chg_terms))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit_s
    s.parameters.num_workers = workers
    s.parameters.random_seed = 1
    status = s.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return list(range(n)), {"status": "NOSOL", "obj": None}
    nxt = {a: b for a, b, lit in arcs if s.Value(lit)}
    order, node = [], nxt[0]
    while node != 0:
        order.append(node - 1)
        node = nxt[node]
    return order, {"status": s.StatusName(status), "obj": s.ObjectiveValue() / SCALE}


def plan_order_dp(now, cur, prods, procs, dues, chg_mean, lam, max_labels=40):
    """Fast planner for a queue that holds only a few PRODUCTS (here 3).

    Restriction (the only difference from plan_order): jobs of the SAME product are run in due-date order.
    With that fixed, a plan is fully described by "how many A, B, C jobs have been done so far, and which
    product was last". We walk through those states and keep, for each, the best (finish time, cost) pairs
    ("labels"; a label is dropped if another is both earlier and cheaper). Cost = lateness + lam * changeover.
    Returns (order, info) in the same format as plan_order, order = indices into the inputs.
    """
    n = len(prods)
    if n == 1:
        return [0], {"status": "TRIVIAL", "obj": None}
    names = sorted(set(prods))
    lists = {p: sorted([j for j in range(n) if prods[j] == p], key=lambda j: (dues[j], j)) for p in names}
    k = len(names)
    sizes = [len(lists[p]) for p in names]

    def co(p, q):
        return 0.0 if (p is None or p == q) else chg_mean[f"{p}->{q}"]

    # state key: (counts tuple, last product index or -1); value: list of (time, cost, first_job)
    start = ((0,) * k, -1)
    labels = {start: [(float(now), 0.0, None)]}
    frontier = [start]
    for _ in range(n):
        nxt = {}
        for key in frontier:
            counts, last = key
            lastp = cur if last < 0 else names[last]
            for pi in range(k):
                if counts[pi] >= sizes[pi]:
                    continue
                j = lists[names[pi]][counts[pi]]
                c = co(lastp, names[pi])
                newc = counts[:pi] + (counts[pi] + 1,) + counts[pi + 1:]
                for (t, cost, first) in labels[key]:
                    fin = t + c + procs[j]
                    cost2 = cost + max(0.0, fin - dues[j]) + lam * c
                    nxt.setdefault((newc, pi), []).append((fin, cost2, j if first is None else first))
        labels = {}
        for key, lab in nxt.items():
            lab.sort()
            keep, best = [], float("inf")
            for t, cost, first in lab:        # increasing time; keep only if cheaper than all earlier labels
                if cost < best - 1e-9:
                    keep.append((t, cost, first))
                    best = cost
            labels[key] = keep[:max_labels]
        frontier = list(labels)
    best = min((l for lab in labels.values() for l in lab), key=lambda x: x[1])
    return [best[2]], {"status": "DP", "obj": best[1]}
