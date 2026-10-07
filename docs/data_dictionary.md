# Data dictionary

File: data/data.csv (identical, byte for byte, to Jupyter_notebook/data.csv in the authors' supplementary zip).
52,026 rows x 181 columns. One row = one machine snapshot, 1 per second.
Columns: `time` + 170 control features (SKIP for this project) + Label_01..Label_10.
Row index restarts each day: use reset_index(drop=True).
Time gaps > 60 s exist before rows 10021, 14332, 37100, 38164. Never subtract timestamps across them.

Status key: CONFIRMED = checked against the authors' Data_irregularities log (timestamps) and/or paper text.
INFERRED = decoded from row counts, internally consistent, not stated by the authors.

| Column | Meaning | Status |
|---|---|---|
| time | timestamp, YYYY-MM-DD HH:MM:SS | CONFIRMED |
| Label_03 | 0 = changeover, 1 = production | CONFIRMED (matches paper: Label_08 odd = changeover, even = production; parity identical to Label_03 on all 52,026 rows) |
| Label_08 | 12 values = 6 transitions x (changeover, production); pair (k, k+1) is one order | CONFIRMED |
| Label_08 -> transition | 5 = A->B, 11 = B->C, 7 = C->B, 1 = B->A, 9 = A->C, 3 = C->A (production phase = next even number) | CONFIRMED: 34/41 logged events match transition and matrix exactly; the other 7 name the right transition but fall in the production run (log phases 71-73) |
| Label_06 | tens digit 1 = changeover, 2 = production; ones digit 1 = irregular | INFERRED (counts: 33,173 + 3,183 changeover; 14,964 + 706 production) |
| Label_07 | 0 = regular changeover, 1 = regular production, 2 = irregular | INFERRED |
| Label_01 | finest phases (max 73). 70 = regular production, 73 = irregular production (706 rows, equals Label_06 = 21) | PARTLY CONFIRMED |
| Label_02,04,05,09,10 | other granularities | SKIP |
| 170 features | machine-control signals | SKIP |

Changeover time definition used in this project: number of seconds in the Label_08-odd run (Label_03 = 0) for that order.
Irregular seconds within a changeover = rows with Label_07 = 2 and Label_03 = 0.

Experimental design (from paper): 5 changeover matrices, each B, C, B, A, C, A (A is the product made before each matrix),
so every transition occurs once per matrix. Authors recommend excluding matrix 1 (and 2) because of a learning-curve effect.
Dates: matrix 1 = 2024-03-12; matrices 2-5 span 2024-03-20 and 2024-03-22.
