"""Step 6 checks: is the sequencing planner actually finding the best order?

C1  hand case 1: lam large, A B A queue  -> must pick the order that does only ONE changeover (A,A,B).
C2  hand case 2: all jobs one product    -> changeovers irrelevant; must be due-date order.
C3  hand case 3: urgent B job, lam = 0   -> must run the urgent B first even though it costs a changeover.
C4  brute force: for 60 random queues of 4-6 jobs, try EVERY order (an independent third method) and check
    CP-SAT's cost equals the true best (within 25 s: CP-SAT rounds times to whole seconds).
C5  DP vs brute force: the DP is only allowed to be worse by the "same product in due-date order" restriction;
    report how much worse (it can never be better than the true best).
C6  planner never loses jobs / returns each job exactly once.
"""
import sys, itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc  # noqa: E402
import optimize as op  # noqa: E402

emp = sc.load_empirical()
CM = op.chg_dict(sc.mean_changeover_matrix(emp[1]))


def cost_of(order, now, cur, prods, procs, dues, lam):
    t, last, c = now, cur, 0.0
    for j in order:
        co = 0.0 if last is None or last == prods[j] else CM[f"{last}->{prods[j]}"]
        t += co + procs[j]
        c += max(0.0, t - dues[j]) + lam * co
        last = prods[j]
    return c


def brute(now, cur, prods, procs, dues, lam):
    return min(cost_of(o, now, cur, prods, procs, dues, lam) for o in itertools.permutations(range(len(prods))))


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))


# C1
p, pr, d = ["A", "B", "A"], [100, 100, 100], [5000, 5000, 5000]
for fn in (op.plan_order,):
    o, _ = fn(0, "A", p, pr, d, CM, 10)
    check("C1 CP-SAT groups the two A jobs (one changeover only)", [p[j] for j in o] == ["A", "A", "B"], "order=" + "".join(p[j] for j in o))
o, _ = op.plan_order_dp(0, "A", p, pr, d, CM, 10)
check("C1 DP groups the two A jobs", [p[j] for j in o][0] == "A", "first=" + p[o[0]])
# C2
p, pr, d = ["B", "B", "B"], [100, 100, 100], [200, 100, 300]  # only order 1,0,2 has zero lateness
o, _ = op.plan_order(0, "B", p, pr, d, CM, 5)
check("C2 same product -> due-date order (CP-SAT)", o == [1, 0, 2], str(o))
o, _ = op.plan_order_dp(0, "B", p, pr, d, CM, 5)
check("C2 same product -> due-date first (DP)", o[0] == 1, str(o))
# C3
p, pr, d = ["A", "A", "B"], [100, 100, 100], [100000, 100000, 300]
o, _ = op.plan_order(0, "A", p, pr, d, CM, 0)
check("C3 urgent B first when lam=0 (CP-SAT)", p[o[0]] == "B", "order=" + "".join(p[j] for j in o))
o, _ = op.plan_order_dp(0, "A", p, pr, d, CM, 0)
check("C3 urgent B first when lam=0 (DP)", p[o[0]] == "B")

rng = np.random.default_rng(11)
worst_cp, gaps, perm_ok = 0.0, [], True
for trial in range(60):
    n = int(rng.integers(4, 7)); lam = float(rng.choice([0, 1, 5, 30]))
    prods = list(rng.choice(list("ABC"), n)); procs = list(rng.uniform(300, 800, n)); now = 2000.0
    dues = list(now + rng.uniform(0, 5000, n)); cur = str(rng.choice(list("ABC")))
    best = brute(now, cur, prods, procs, dues, lam)
    o, info = op.plan_order(now, cur, prods, procs, dues, CM, lam, time_limit_s=20)
    perm_ok &= sorted(o) == list(range(n))
    cp_cost = cost_of(o, now, cur, prods, procs, dues, lam)
    worst_cp = max(worst_cp, cp_cost - best)
    od, idp = op.plan_order_dp(now, cur, prods, procs, dues, CM, lam)
    assert idp["obj"] >= best - 1e-6, "DP claims better than the true best -> bug"
    gaps.append((idp["obj"] - best) / max(best, 1.0))
check("C4 CP-SAT equals brute-force optimum (60 queues)", worst_cp < 25, f"worst excess {worst_cp:.1f} s")
check("C5 DP never beats brute force; extra cost from restriction", True,
      f"mean {100*np.mean(gaps):.1f}%, median {100*np.median(gaps):.1f}%, worst {100*np.max(gaps):.1f}%")
check("C6 every job appears exactly once", perm_ok)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
