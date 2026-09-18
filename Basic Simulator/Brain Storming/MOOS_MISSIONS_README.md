# COLREG Evaluation Missions for MOOS-IvP

12 multi-vehicle MOOS-IvP missions for testing a COLREG-compliant collision-
avoidance agent, structured like the Imazu-problem benchmark family (single-
target, two-target, three-target buckets of increasing difficulty). Each
scenario places your own-ship ("opship") on a genuine collision course with
one or more straight-running target ships and checks whether she resolves it
per the applicable Rule(s).

## What's included

```
generate_moos_scenarios.py   # regenerates everything below from scratch
scenarios_manifest.json      # machine-readable rule_refs + pass_criteria per scenario
score_scenario.py            # post-run CPA / compliance scoring from a trajectory CSV
missions/
  s01_head_on/
  s02_crossing_stbd_fine/
  s03_crossing_stbd_broad/
  s04_crossing_stbd_abeam/
  s05_crossing_port_broad/
  s06_crossing_port_fine/
  s07_overtaking_os_gives_way/
  s08_overtaking_os_stands_on/
  s09_double_crossing_squeeze/
  s10_headon_plus_crossing/
  s11_overtake_plus_crossing/
  s12_converging_cluster/
    shoreside.moos
    opship.moos / opship.bhv
    ts1.moos / ts1.bhv  (ts2, ts3 as needed)
    launch.sh
    README.md            # scenario-specific rule, roles, pass/fail criteria
```

## Important: read before running

This is a generated scaffold, not something I ran against a live MOOS-IvP
install (not available in this environment). Two tiers of confidence:

**High confidence — the part that matters most:**
The collision **geometry** is computed exactly (`generate_moos_scenarios.py`
solves a closed-form intercept equation per target so that, absent any
avoidance behavior, own-ship and each target genuinely collide at a computed
time). This is the hard-to-get-right part and it's verified by construction —
independent of MOOS-IvP version quirks.

**Needs verification against your install:**
- The inter-process sharing/bridging blocks (`pShare`, `uFldNodeBroker`,
  `uFldShoreBroker`) follow standard multi-community MOOS-IvP conventions,
  but exact block syntax has drifted across releases. If multi-vehicle
  position sharing doesn't come up cleanly on first launch, diff these
  blocks against an official multi-vehicle example mission shipped with
  your install (typically under `moos-ivp/ivp/missions/` or
  `moos-ivp-extend`) and adjust.
- `BHV_AvoidCollision` parameters in `opship.bhv` (`pwt_outer_dist`,
  `pwt_inner_dist`, `completed_dist`, `collision_distance`, etc.) are a
  reasonable starting point but should be checked with `pHelmIvP --alist`
  or your version's behavior `--help` before you trust the numbers.
- `pMarineViewer`'s `TIFF_FILE` references a background image
  (`forrest19.tif`) that ships with some MOOS-IvP example missions but not
  all installs — delete that line if you don't have it, or point it at your
  own chart tile.
- If you're testing your own agent rather than `BHV_AvoidCollision`, replace
  the `Behavior = BHV_AvoidCollision {...}` block in `opship.bhv` with
  whatever interface your agent uses to publish course/speed changes into
  pHelmIvP (a custom behavior, an IvP function generator, or an external
  process posting directly to `DESIRED_COURSE`/`DESIRED_SPEED` if you bypass
  pHelmIvP's arbitration entirely).

## Running a scenario

```bash
cd missions/s01_head_on
chmod +x launch.sh
./launch.sh          # optionally: ./launch.sh 4   (4x time-warp)
```

Then open the shoreside `pMarineViewer` window and click **DEPLOY** on each
vehicle (or poke `DEPLOY=true` into each vehicle's MOOSDB directly).

To kill everything between runs:
```bash
kill -9 $(ps -ef | awk '/MOOSDB|pAntler|pHelmIvP|uSimMarine|pMarinePID|pMarineViewer|pLogger|pShare|pHostInfo|uFldNodeBroker|uFldShoreBroker|pNodeReporter/{print $2}')
```

## Confirming a scenario is a genuine collision course

Before trusting a "PASS" from your avoidance agent, run each scenario once
with avoidance disabled (edit `generate_moos_scenarios.py`'s call to
`bhv_opship(..., use_avoid=False, ...)` for that scenario, or simply comment
out the `BHV_AvoidCollision` block in `opship.bhv`, and re-run). Own-ship and
the target(s) should pass within a few metres of each other — if they don't,
the scenario's geometry isn't actually forcing a decision and any "pass"
from your agent on that scenario is not meaningful.

## Scoring a run

`score_scenario.py` computes closest-point-of-approach (CPA) per target and
a rough starboard/port passing-side heuristic from a trajectory CSV
(`time,vehicle,x,y,heading,speed`). See the docstring at the top of the
script for how to produce that CSV from a real run (easiest: a small pymoos
subscriber logging `NODE_REPORT_LOCAL`; alternative: extract from `.alog`
via your version's `aloggrep`/`alogscan`).

```bash
python3 score_scenario.py trajectory.csv \
    --scenario s01_head_on \
    --safe-distance 50 \
    --manifest scenarios_manifest.json
```

This gives you CPA pass/fail per target automatically. The richer
rule-specific checks in each scenario's `pass_criteria` (e.g. "did own-ship
alter to starboard, not port," "did she hold course as stand-on until it was
necessary to act") are listed in `scenarios_manifest.json` but are **not**
yet automatically graded — the scorer flags this in its output. Extending
`score_scenario.py` with a per-scenario check function (e.g. inspecting the
sign of own-ship's heading change at the moment she first deviates from her
original track) is the natural next step if you want fully automated
grading rather than a human glancing at the trajectory plot.

## Relationship to the Imazu problem / the 500-item Q&A set

These 12 scenarios follow the same escalating structure as the standard
Imazu benchmark (single-target → two-target → three-target encounters) but
are **not** a verbatim reproduction of Imazu's original published figures —
I don't have reliably-sourced exact numeric values for those, so I generated
fresh geometry with the same pedagogical intent and exact, checkable math
instead of guessing at unverified published numbers. If you need the literal
Imazu benchmark for comparability with published papers, you'll want to
source Imazu's original figure or a paper that tabulates it precisely (e.g.
Cai & Hasegawa's evaluation papers cited in the literature overview) and
adapt `generate_moos_scenarios.py`'s scenario list to match.

Each scenario's `rule_refs` field maps directly onto the Rule numbers used
in the 500-item COLREG Q&A dataset, so you can cross-reference: e.g. if
`s05_crossing_port_broad` fails, go back to the Q&A items tagged `Rule 17`
for the specific sub-rule text your agent may be misapplying.
