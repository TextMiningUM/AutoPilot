# evaluate_run.py — composite COLREG-agent evaluation function

Extends `score_scenario.py` (from the MOOS eval missions) with a full
multi-axis score, built around the four-axis framework used in Woerner,
Benjamin, Novitzky & Leonard, *"Quantifying protocol evaluation for
autonomous collision avoidance: toward establishing COLREGS compliance
metrics"* (Autonomous Robots, 2019) — safety, protocol compliance, spatial
efficiency, temporal efficiency — plus two axes you specifically asked for
(manoeuvre count, smoothness). I don't have their exact formulas (paywalled
paper, only abstracts available), so this isn't a reproduction of their
math — it's my own construction using their four-axis *structure* as a
validated starting point, tested against synthetic trajectories designed
specifically to catch the failure mode you flagged.

## The core design decision: two hard gates, not one big weighted sum

If every axis is blended into a single weighted sum, an agent can learn to
game it — stopping trivially maximizes safety and (with no motion at all)
also maximizes manoeuvre-count and smoothness scores, since standing still
involves zero manoeuvres and zero control effort. A naive weighted sum
scored this at **0.70** in testing — far too forgiving for a run that never
completed its mission.

The fix is structural, not just "add more weight to efficiency":

1. **Collision gate**: any run where CPA drops below `collision_radius_m`
   scores exactly **0.0**, full stop, regardless of every other axis.
2. **Mission-incomplete gate**: any run that never reaches the goal is
   capped at **0.2** (only compliance/manoeuvre/smoothness can contribute,
   and even a perfect score on those caps out low), regardless of how
   "clean" the partial trajectory looked.

Only runs that both avoided collision AND reached the goal are scored on
the full weighted blend of all five remaining axes.

## Two real bugs found and fixed while testing this

I built five synthetic trajectories specifically designed to break the
scorer (stop-forever, smooth-and-efficient, jerky-zigzag, actual-collision,
huge-unnecessary-detour) and ran the function against all five before
calling it done. That caught two genuine bugs, not just the gaming problem
above:

1. **The stop-forever gaming problem** described above — fixed via the
   mission-incomplete gate.
2. **The manoeuvre counter missed continuous zigzags.** A trajectory that
   alternates small corrections in opposite directions (e.g. +5°, -5°, +6°,
   -4°, ...) never drops back to a near-zero heading-rate between
   corrections, so the original "count transitions above a deadband" logic
   saw it as one single sustained manoeuvre instead of dozens of discrete
   corrections. Fixed by also counting a new manoeuvre event whenever
   heading-rate *reverses sign* while still above the deadband. Confirmed:
   before the fix, an 80-correction zigzag trajectory counted as 1
   manoeuvre; after the fix, it correctly counts as 80.

## Final validated behaviour (five synthetic test trajectories)

| Scenario | Verdict | Composite |
|---|---|---|
| Actual collision | FAIL | **0.000** |
| Stop and never move | FAIL (incomplete) | **0.200** |
| Huge unnecessary detour (1.5x distance/time) | PASS | 0.791 |
| Jerky zigzag (80 corrections, but fast and direct) | PASS | 0.799 |
| Smooth, decisive, one clean alteration | PASS | **0.947** |

Note that the detour and the zigzag land close together (0.791 vs 0.799)
despite being very different failure modes. That's a consequence of the
default weights (`compliance` 0.30, `temporal`/`spatial` 0.15 each,
`manoeuvre` 0.15, `smoothness` 0.25) — a jerky-but-direct run and a
smooth-but-roundabout run are, under these weights, roughly equally bad.
**This trade-off is genuinely a judgement call, not something with one
correct answer** — if you care more about passenger comfort than transit
time, raise `smoothness`; if fuel/time matters more than manoeuvre
elegance, raise `temporal`/`spatial`. The weights are a parameter, not a
constant, precisely so you can tune this to what your evaluation actually
values.

## Compliance axis: deterministic, band-aware (compliance-rebuild STAP 3, 2026-09-23)

`compliance_axis()` no longer takes pluggable `violation_checks` callables or an
LLM-supplied `llm_compliance_score` -- it computes the score itself, deterministically,
from two inputs the CALLER supplies (see `app/evaluation.py`'s `score_trajectory()`,
which builds both from `app/measurement.py`'s checks and `app/evaluation.py`'s own
STAP-2 structured ground truth, never from a hardcoded/independent risk definition):
  - `checkpoint_codes`: `[(step, [code, ...]), ...]` -- per-decision error codes.
  - `run_level_codes`: `[(code, detail), ...]` -- trajectory-level findings (e.g.
    `P_wrong_side_pass`, `cpa_violation`).
Each code in `COMPLIANCE_WEIGHTS` deducts a fixed amount from a 1.0 starting score, once
per occurrence, clipped to `[0,1]`; an actual collision (`collided=True`) is a hard gate
straight to `0.0`, never just a large deduction. The result's `compliance.breakdown` is a
list of `{"code", "label", "at" (step_or_detail), "deduction"}` dicts -- `label` (added
2026-09-24, `COMPLIANCE_LABELS`) is a short, demo-facing English phrase per code (e.g.
"Hallucinated Risk" for `A_fabricated_risk`), baked in so the raw run JSON is
self-explanatory without a code lookup table. Every score is traceable back to the
specific findings that produced it. Older, already-generated runs still have bare
`[code, step_or_detail, deduction]` list entries (never migrated) -- see
`app/evaluation.py`'s `compliance_finding_parts()` for the one place that reads both
shapes and looks up a label for the old ones at display time.

## Usage

```bash
python3 evaluate_run.py trajectory.csv \\
    --own-ship opship --start-x 0 --start-y 0 --goal-x 2000 --goal-y 0 \\
    --nominal-speed 5.0 --collision-radius 15 --safe-distance 50
```

Or from Python, to plug in pre-computed compliance codes and custom weights:

```python
from evaluate_run import evaluate_run

result = evaluate_run(
    "trajectory.csv", own_vehicle="opship",
    start_xy=(0, 0), goal_xy=(2000, 0), nominal_speed=5.0,
    collision_radius_m=15, safe_distance_m=50,
    checkpoint_codes=[(0, ["B_wrong_direction"])],
    run_level_codes=[("P_wrong_side_pass", "ts1")],
    weights={"compliance": 0.35, "temporal": 0.10, "spatial": 0.10,
             "manoeuvre": 0.15, "smoothness": 0.30},
)
```

