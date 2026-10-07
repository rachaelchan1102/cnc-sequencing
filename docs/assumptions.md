# Assumptions table (every assumed input; to be copied into the final README)

| # | Assumption | Value / range | Real or assumed | Why this choice |
|---|---|---|---|---|
| 1 | Processing time of a job | resampled from the 10 observed production runs of that product (A 71-140 s, B 231-245 s, C 1,124-1,601 s) | REAL | Dataset |
| 2 | Changeover time A->B etc. | resampled from the observed changeovers, matrices 3-5 (n = 3 per transition) | REAL | Dataset; matrices 1-2 excluded for learning-curve effect (authors' advice); irregular seconds kept in |
| 3 | What a "job" is | one production run = one workpiece | ASSUMED | Dataset states no order quantity; batch size is varied separately in step 8 |
| 4 | Arrival process | Poisson, rate lambda jobs/hour (exponential gaps) | ASSUMED | Standard model of independent orders; no demand data exists |
| 5 | Load levels | light / medium / heavy = 15% / 50% / 75% of the way across the window 1/(p+s) < lambda < 1/p, computed per mix (heavy was 85% until 2026-10-07; at 85% the machine was ~92% busy even with zero changeovers) | ASSUMED (placement) | Plan's formula; below the window nothing queues, above it no strategy keeps up |
| 6 | Product mixes (A, B, C) | balanced 1/3 each; A-heavy 60/20/20; B-heavy 20/60/20; C-heavy 20/20/60 | ASSUMED | Plan's four mixes |
| 7 | Due-date rule | SLK: due = arrival + own processing time + k x (one "job-cycle" = mix-average run time + average changeover an unsorted line pays per job, about 1,200 s = 20 min for the balanced mix) | ASSUMED | Replaces the plan's original TWK rule (due = arrival + k x own processing time), which makes most A jobs impossible to meet: see check_scenarios.py check 8 |
| 8 | Due-date tightness k | Baseline k = 4 (about 80 min of slack); swept 1 to 16 (20 min to 5.3 h) | ASSUMED | Re-scaled on 2026-10-07: the plan's k = 2..8 in units of the average RUN time (9 min) gave only 18-70 min of slack, shorter than one changeover, so nearly every job was late. Baseline chosen so the best rules are neither almost never late nor almost always late; see PROJECT_LOG section 15 |
| 9 | Horizon | 40 hours of arrivals per replication (a warm-up period is discarded in the simulator) | ASSUMED | Long enough for queues to build; length is a choice |
| 10 | Replications | 30 per scenario, seeds fixed | ASSUMED | Plan; common random numbers across strategies |
| 11 | Starting state | The machine starts with no product set up, so the first job needs no changeover | ASSUMED | Simplest choice; affects only the first job of each run |

Notes
- Above the lower limit of the window, first-come-first-served is unstable (its queue keeps growing), so its lateness
  depends on the horizon length. Report that, do not hide it.
- Due-date rule: the slack (SLK) rule is a standard rule in the scheduling literature, but I have not verified a
  citation yet. Check one before citing it in the README.
