# Driving a MOOS-IvP vehicle from an external LLM

This folder shows how to let an external LLM (grounded in your 500-item
COLREG Q&A set, or a model you've fine-tuned on it) make own-ship's
collision-avoidance decisions directly, with **no MOOS-native rule logic
(`pHelmIvP` / `BHV_AvoidCollision`) in the loop at all.**

## Files

- `colreg_llm_bridge.py` — the bridge. Connects to own-ship's MOOSDB as an
  independent client, reads state + contacts, calls the LLM on a slow
  cadence, publishes `DESIRED_HEADING` / `DESIRED_SPEED` directly.
- `opship_llm.moos` — example vehicle community with `pHelmIvP` removed
  from `ANTLER`. Everything else (physics, PID, contact sharing, logging)
  is unchanged from a normal mission.
- This README.

## How it fits with the missions I built earlier

Take any scenario from `moos_colreg_eval_missions.zip` (e.g.
`s01_head_on/`) and swap its `opship.moos` for a version of
`opship_llm.moos` with that scenario's `START_POS` (see the scenario's own
`README.md` for the exact start x/y/heading/speed). You don't need
`opship.bhv` at all in this mode — delete the `Run = pHelmIvP` line and
you're done. `ts1.moos`/`ts2.moos`/etc. and `shoreside.moos` are untouched.

## Step by step

**1. Install dependencies for the bridge**
```bash
pip install pymoos anthropic --break-system-packages
```

**2. Launch the MOOS side as usual, minus pHelmIvP on own-ship**
```bash
cd your_scenario_folder
pAntler shoreside.moos &
pAntler opship_llm.moos &     # note: LLM variant, not opship.moos
pAntler ts1.moos &
# ...
```
Deploy `ts1`/`ts2`/etc. from `pMarineViewer` as normal (`DEPLOY=true`).
Own-ship needs no DEPLOY step in this mode — `pMarinePID`'s
`ACTIVE_START=true` means it's already listening.

**3. Launch the bridge, pointed at own-ship's MOOSDB port**
```bash
export ANTHROPIC_API_KEY=sk-...
python3 colreg_llm_bridge.py \
    --moos-port 9001 \
    --moos-community opship_llm_bridge \
    --own-ship-name opship \
    --goal-x 0 --goal-y 2400 \
    --qa-dataset /path/to/colreg_qa_500.json \
    --decision-period 5 \
    --default-speed 2.5
```
`--moos-community` is the bridge's own client name on the MOOSDB, distinct
from `--own-ship-name` (which must match the `NAME=` field own-ship uses in
its `NODE_REPORT`, i.e. the vehicle `Community` in `opship_llm.moos`).

**4. Sanity-check the wiring before spending API calls**
```bash
python3 colreg_llm_bridge.py --moos-port 9001 --moos-community test \
    --own-ship-name opship --dry-run
```
This confirms `NAV_*`/`NODE_REPORT` are arriving and prints what it *would*
publish, without calling the LLM.

## What the LLM actually sees, each decision cycle

For every contact, the bridge computes bearing, range, CPA, TCPA, and a
rough encounter classification (head-on / crossing-from-starboard /
crossing-from-port / overtaking-geometry) purely from geometry — this
classification is only used to select which Q&A items to retrieve as
grounding context, not as the actual decision (the LLM makes the actual
call). The prompt looks roughly like:

```
OWN-SHIP: x=0.0 y=550.2 heading=0.3 speed=2.50 goal=(0.0,2400.0)

CONTACTS:
  ts1: x=71.2 y=612.4 heading=249.8 speed=2.20 | relative_bearing=41.6
       range=105.3m CPA=12.4m TCPA=8.1s encounter_type=crossing_target_on_starboard

RELEVANT COLREG REFERENCE (from training Q&A set):
  - [Rule 15] Q: Define a crossing situation and state who gives way under Rule 15.
    A: When two power-driven vessels are crossing so as to involve risk of
       collision, the vessel which has the other on her own starboard side
       shall keep out of the way...
  - [Rule 16] Q: What must the give-way vessel do under Rule 16?
    A: Take early and substantial action to keep well clear.
  ...
```
The LLM replies with strict JSON (`heading`, `speed`, `rule_applied`,
`reasoning`), which the bridge parses and publishes.

## Important gotchas

- **Decision cadence vs. control rate.** `pMarinePID`/`uSimMarine` run at
  10 Hz; an LLM call takes 1-5+ seconds. The bridge deliberately decouples
  these — it decides every `--decision-period` seconds and pMarinePID just
  holds the last commanded heading/speed in between. Don't try to call the
  LLM every control tick.
- **Time-warp breaks this if you're not careful.** If you run MOOS at
  `MOOSTimeWarp > 1` for faster-than-real-time simulation, LLM API latency
  does *not* speed up with it — 3 seconds of real wall-clock latency at
  10x warp is 30 seconds of *simulated* time with a stale command. Either
  keep warp near 1 while using a live API, or switch to a small
  low-latency local model (e.g. a fine-tuned open model served locally)
  once you're past prompt-design and want to run faster-than-real-time
  regression sweeps across all 12 scenarios.
- **No built-in safety net.** With `pHelmIvP`/`BHV_AvoidCollision` removed,
  there is genuinely nothing else watching for collision risk if the
  bridge process crashes, the API call fails, or the LLM returns a bad
  heading. The script holds the *last* good command on a failed cycle
  (see `except Exception` in `decision_loop`) rather than doing nothing,
  but that's a minimal safeguard, not a real one. For anything beyond a
  desktop test, consider keeping a lightweight independent watchdog
  (even a trivial one: hard-stop own-ship if any contact's live CPA drops
  below some critical distance, checked outside the LLM loop) — exactly
  the kind of "last-resort" layer the MASS Code discussion earlier flags
  as a human-oversight/safety-case expectation, not just a nice-to-have.
- **Retrieval is keyword-based, not semantic.** `QARetriever` matches on
  `rule_ref` substrings against the geometrically-classified encounter
  type. It's intentionally simple so you can see exactly what's being
  retrieved and why. If your fine-tuned model already has the Q&A content
  baked in via training, you can drop the retrieval step entirely (pass
  `--qa-dataset` nothing) and rely on the model's learned knowledge
  instead — that's the more realistic end-state you described (LLM
  reasoning from training, not RAG at inference time).
- **`pymoos` API drift.** `comms.run()`'s exact signature and message
  accessor names have varied across pymoos releases. If you get
  `AttributeError`s on `msg.key()`/`msg.double()`/`msg.string()`, run
  `python3 -c "import pymoos; help(pymoos.comms)"` and adjust the handful
  of calls in `_on_mail`/`_on_connect` accordingly — the overall
  architecture doesn't change, just those specific calls.

## Swapping in your fine-tuned model instead of the raw Anthropic API

`LLMDecisionMaker.__init__` takes an `api_client` — pass anything with a
`.messages.create(...)`-shaped interface, or just rewrite `decide()` to
call your own model's HTTP endpoint instead. The rest of the bridge
(state tracking, geometry, retrieval, publishing) doesn't need to change.
