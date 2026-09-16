# Addendum: pausing the world while the agent thinks

`colreg_llm_bridge_paused.py` replaces the fixed-cadence loop in
`colreg_llm_bridge.py` with a **pause / decide / resume** loop, so an LLM
that takes anywhere from 1 second to several minutes to respond costs
**zero simulated mission time.**

## Why this needed a new connection design

The original bridge only connected to own-ship's MOOSDB. That's not enough
here: `USM_SIM_PAUSED` is local to each vehicle's own `uSimMarine` instance,
and every vehicle (own-ship, ts1, ts2, ts3...) runs its own `uSimMarine` in
its own MOOS community on its own port. To actually freeze the *scenario*
(not just own-ship), the bridge now opens one small pymoos connection per
vehicle purely to flip `USM_SIM_PAUSED` on all of them together.

## Running it against one of the 12 eval missions

Take `s01_head_on` from `moos_colreg_eval_missions.zip`. Its `opship.moos`
and `ts1.moos` already run on ports 9001 and 9002 respectively (see each
scenario's own generated `.moos` files). Launch as before (own-ship using
the LLM-driven `opship_llm.moos` variant, i.e. no `pHelmIvP`), then:

```bash
export ANTHROPIC_API_KEY=sk-...
python3 colreg_llm_bridge_paused.py \
    --own-ship opship:9001 \
    --other-vehicles ts1:9002 \
    --goal-x 0 --goal-y 2400 \
    --qa-dataset colreg_qa_500.json \
    --step-duration 5 \
    --risk-horizon 300
```

For `s09_double_crossing_squeeze` (two targets) or `s12_converging_cluster`
(three targets), just extend `--other-vehicles`:
```bash
--other-vehicles ts1:9002,ts2:9003,ts3:9004
```

## What you'll see in the console

```
Connecting to own-ship 'opship' at localhost:9001 ...
Setting up pause control for: ['opship', 'ts1']
Starting pause/decide/resume loop. step_duration=5.0s risk_horizon=300.0s.

[skip] min_tcpa=612.3s > risk_horizon (300.0s) — no LLM call this cycle
[skip] min_tcpa=487.9s > risk_horizon (300.0s) — no LLM call this cycle
[decision] min_tcpa=284.1s -> LLM took 3.2s (world was frozen for all of it) -> {'heading': 42.0, ...}
[decision] min_tcpa=41.7s -> LLM took 61.4s (world was frozen for all of it) -> {'heading': 55.0, ...}
```

Notice the last line: the LLM took a full minute, and the log line says so
explicitly — but because the world was paused for all of it, that minute
did not let the target ship close another 150+ metres while nobody was
looking. That's the whole point.

## The `--risk-horizon` gate

Calling the LLM on every single pause/resume cycle, including when nothing
is anywhere near own-ship, wastes API calls and money for no benefit. The
gate computes TCPA (time to closest point of approach, assuming no one
manoeuvres) for every contact and only calls the LLM if at least one
contact's TCPA is under `--risk-horizon` seconds. Outside that horizon, the
cycle just pauses briefly, confirms nothing needs attention, and resumes —
own-ship keeps whatever heading/speed `pMarinePID` was last holding (since
there's no `pHelmIvP` running a waypoint behavior to make her keep heading
toward the goal on her own — see "own-ship needs an explicit initial
heading" below).

## Own-ship needs an explicit initial heading

Because `pHelmIvP` isn't running, nothing is autonomously steering own-ship
toward her goal outside of what this bridge publishes. Either:
- publish an initial `DESIRED_HEADING`/`DESIRED_SPEED` toward the goal
  once at startup (before the first pause cycle), or
- accept that own-ship sits still until the first LLM decision arrives.

The simplest fix is a one-line addition right after connecting: compute the
bearing from own-ship's start position to `(--goal-x, --goal-y)` and
publish that as the initial `DESIRED_HEADING` before entering the loop.

## Tuning `--step-duration` vs. `--risk-horizon`

- `--step-duration` controls how far the (unpaused) world is allowed to
  advance between checks. Too large and you might blow past a developing
  close-quarters situation before the next pause catches it; too small
  and you're pausing/resuming (and re-running the TCPA check) constantly
  for no benefit. 5 seconds of simulated time at a modest `MOOSTimeWarp`
  is a reasonable starting point for the single/two-target scenarios in
  this set; tighten it for the three-target scenario where things develop
  faster.
- `--risk-horizon` should comfortably exceed your worst-case LLM latency
  (including retries) — if the LLM can take 60 seconds and your horizon is
  only 30 seconds of simulated time, you risk a contact closing to an
  unsafe range while a still-pending decision hasn't landed. Since the
  world is paused during the call itself this can't actually happen once
  a decision cycle has started (the pause holds until the decision is
  published), but a too-tight horizon means you're triggering that
  (possibly slow) call later than ideal relative to when the risk first
  became meaningful.

## MOOSTimeWarp still matters for the *unpaused* portions

Nothing above changes how fast the unpaused `--step-duration` window
elapses in simulated terms — that's still governed by `MOOSTimeWarp` as
normal. A common combination: set `MOOSTimeWarp` fairly high (e.g. 10-20x)
so the "nothing nearby, just cruising" portions fly by quickly in
wall-clock time, while the pause mechanism separately guarantees the
"something nearby, thinking required" portions never rush the LLM
regardless of warp.
