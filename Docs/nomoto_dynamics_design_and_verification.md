# Nomoto Ship-Manoeuvring Dynamics in the OOW Simulator — Design, Implementation & Verification

**Status:** Phase 0–3 (step 2, parts A+B) complete and merged to `main`. Phase 3 step 2's
Track-2 data regeneration and Phase 4 (bow-crossing capsule domain) are **not yet done**
— see "Known gaps / next steps" at the end.

**Audience:** anyone auditing, reproducing, or extending this work. Every numeric claim
below is either a literature citation or a number actually measured from the code in this
repository (never an assumption) — see "Reproducing the numbers in this document" for the
exact commands.

---

## 1. Motivation

The OOW (Officer of the Watch) agent simulator (`Basic Simulator/`) originally moved
own-ship with a simple **turn-rate slew**: heading approaches a commanded target at a
fixed `turn_rate_deg_s` (default 3.0°/s), independent of how large the ordered turn is.
This is a reasonable first approximation but is **not** how a real ship (or Sawada et
al.'s own simulated ship) actually turns — real ships have rudder machinery with its own
response lag, and the ship's yaw rate itself builds up gradually in response to rudder,
not instantaneously.

To make this project's own trajectories directly comparable to a real, published
reference (Sawada, Sato & Majima 2021 — the paper whose exact 22 Imazu scenarios this
project already reproduces byte-for-byte, see `generate_imazu_missions.py`), we
implemented that paper's own **Nomoto first-order ship-manoeuvring model** plus its
rudder-servo dynamics, as an **opt-in** physics model living alongside (not replacing)
the original slew model.

### 1.1 Primary reference

> Sawada, R., Sato, K. & Majima, T. (2021). *Automatic ship collision avoidance using
> deep reinforcement learning with LSTM in continuous action spaces.* Journal of Marine
> Science and Technology, 26, 509–524. https://doi.org/10.1007/s00773-020-00755-0
> (open access PDF also archived locally at
> `Docs/automatic-ship-collision-avoidance-using-deep-reinforcement-4mpps56bvi.pdf`)

Relevant facts taken **verbatim** from that paper (§4.1–4.4, quoted exactly, page numbers
as printed):

- *"detection of a grid sensor and updating of a command rudder angle were carried out
  every 10 s in the simulation time, and for the motion calculation, the interval of the
  integration was set to 1 s."* (p. 515)
- *"An episode is defined as a sequence from a start of a simulation to a termination of
  the simulation by satisfying terminal conditions. The terminal conditions of an episode
  are that the distance to a waypoint becomes less than or equal to a specified distance,
  or the simulation reaches the set maximum steps."* (p. 516) — the paper does **not**
  publish the exact numeric value of "maximum steps" anywhere (checked Table 2 and the
  surrounding text directly); this project derives its own step budget instead of
  guessing that number (see §7).
- Table 2 ("Configurations of environment"): safe passing distance = 0.5 NM; grid sensor
  detection interval = 10.0 s; grid sensor radius = 12.0 NM.
- §2.3 "Bow crossing range": *"a capsule-shaped region with a radius of the safe passing
  distance and the bow crossing range is set to 1.0 NM."* — **not yet implemented** in
  this project (Phase 4, still pending).
- §4.4 "Scenario": own-ship positioned at (X, Y) = (−6.0, 0.0) NM, waypoint at (6.0, 0.0)
  NM, *"Each target ship is positioned at the speed so that the TCPA is 30 mins."* — this
  project's `generate_imazu_missions.py`/`generate_um_rescaled_missions.py` already
  reproduce this exactly.
- §5 (discussion of Figs. 11/12, the "autopilot model" variant — architecturally the
  closest match to this project's own design, since the LLM agent gives a target course
  rather than raw rudder commands): *"This may be due to the reward design that there is
  no negative reward for the elapsed time, so there is a less need to hurry to arrive at
  the waypoint. Nevertheless, it is necessary to design the reward so that the model can
  be constructed in a time-efficient manner while retaining this flexible control."* —
  i.e. the paper's own autopilot-style agent is **expected** to take longer, more
  circuitous routes than a raw-rudder agent; this is a known, paper-acknowledged
  characteristic of this architecture, not a bug.

### 1.2 Secondary references (ship-dynamics parameter grounding, §9 below)

- Nomoto, K. (1960). *Analysis of Kempf's standard manoeuvre test and proposed steering
  quality indices.* Proc. 1st Symp. on Ship Manoeuvrability. (The original model.)
- Xie, W. et al. (2023). *Generalized Behavior Decision-Making Model for Ship Collision
  Avoidance via Deep Reinforcement Learning.* J. Mar. Sci. Eng., 11(2), 273.
  https://www.mdpi.com/2077-1312/11/2/273 (Table 3 gives a real K/T/rudder parameter set
  for a 52.5 m vessel — archived at `Docs/jmse-11-00273-v2.pdf`.)
- Wang et al. (2023). *A Multi-Ship Collision Avoidance Algorithm Using Data-Driven
  Multi-Agent Deep Reinforcement Learning.* J. Mar. Sci. Eng., 11(11), 2101.
  https://doi.org/10.3390/jmse11112101 (Table 6, "YU KUN" training vessel, 105 m —
  archived at `Docs/jmse-11-02101-v2.pdf`.)

---

## 2. The mathematical model

All angles below are in **degrees** and **degrees/second** throughout the implementation
(not radians) — a deliberate choice matching every other angle in this codebase
(headings, bearings), to avoid a radian/degree conversion at every call site.

### 2.1 Rudder servo (1st-order lag)

$$T_E \frac{d\delta}{dt} + \delta = \delta_c$$

where $\delta$ is the actual rudder angle, $\delta_c$ is the commanded ("ordered") rudder
angle (clipped to the physical rudder limit $\pm\delta_{max}$ before being applied), and
$T_E$ is the steering-gear's own time constant. This models the fact that a real rudder
cannot instantaneously jump to a new angle — the steering engine takes time to move it.

### 2.2 Nomoto's first-order yaw-response model

$$T \frac{dr}{dt} + r = K\delta$$

where $r$ is the yaw rate (deg/s) and $\delta$ is the (already-lagged) actual rudder
angle from §2.1. $K$ (units: 1/s) and $T$ (units: s) are the ship's own manoeuvrability
indices — $K$ is the steady-state gain (how much yaw rate a given rudder angle
eventually produces) and $T$ is the time constant of how quickly the yaw rate responds.

### 2.3 Heading integration

$$\frac{d\psi}{dt} = r$$

Heading is simply the running integral of yaw rate.

### 2.4 Steady-state turn rate (a directly checkable closed-form result)

At steady state ($dr/dt = 0$), equation 2.2 reduces to $r_{ss} = K\delta$. At full rudder
($\delta = \delta_{max}$) this gives the ship's maximum sustained turn rate. For Sawada's
own parameters ($K=0.05\,\text{s}^{-1}$, $\delta_{max}=10°$): $r_{ss} = 0.05 \times 10 =
0.5°/\text{s}$ — this is the **exact** figure the user's own source material cited as
"~0,5°/s stationair", and is reproduced as an exact assertion in
`test_steady_turn_rate_at_full_rudder_matches_K_times_limit` (see §6).

### 2.5 Autopilot controller (this project's OWN addition — not part of Sawada's published constants)

Sawada's own paper trains an RL policy end-to-end; it does not publish a specific
classical control law for "given a desired heading, compute a commanded rudder angle".
Since this project's own architecture (see §5) has the LLM/agent order an **absolute
target heading**, not a raw rudder angle, a small proportional controller was added to
bridge the two:

$$\delta_c = \mathrm{clip}\big(K_p \cdot e_\psi,\ -\delta_{max},\ +\delta_{max}\big)$$

where $e_\psi$ is the shortest signed heading error (target minus current, wrapped to
$[-180°, 180°]$) and $K_p$ (default 1.0) is a tunable gain. **This is explicitly flagged
in the code as an approximation, not a verified paper value** — see
`pipeline/nomoto.py`'s own module docstring.

### 2.6 Numerical integration

All three ODEs (2.1–2.3) are integrated with explicit (forward) Euler at a fixed
**1-second substep**, matching the paper's own stated integration interval (§1.1) —
regardless of the OUTER decision-cadence step size (`dt`, which in this project's
adaptive decision cadence ranges 10–200 s). This is implemented via internal
sub-stepping (`pipeline/nomoto.py::advance()`) so a caller using a 10 s or 60 s outer
step gets numerically IDENTICAL results to calling the same function ten or sixty times
at 1 s each — verified exactly by
`test_substep_1s_matches_calling_advance_once_per_second`.

---

## 3. Implementation

**File:** `pipeline/nomoto.py` (chosen location: shared by both the live simulator,
`Basic Simulator/app/simulation.py`, and the OOW prompt/training-spec module,
`pipeline/oow_agent_spec.py` — `pipeline/` never depends on `Basic Simulator/app/`, only
the reverse, so this physics core could not live under `app/` once `oow_agent_spec.py`
needed it too).

```python
@dataclass(frozen=True)
class NomotoParams:
    K_per_s: float = 0.05
    T_s: float = 50.0
    T_E_s: float = 2.5
    rudder_limit_deg: float = 10.0
    autopilot_kp: float = 1.0

@dataclass(frozen=True)
class NomotoState:
    rudder_deg: float = 0.0
    yaw_rate_deg_s: float = 0.0
    heading_deg: float = 0.0
```

Pure functions (no project imports, no side effects — a deliberate dependency-free
design so this module can be imported from anywhere without pulling in torch/streamlit):

- `_signed_heading_diff(from_deg, to_deg)` — shortest signed angular difference, wrapped
  to $[-180°,180°]$.
- `autopilot_rudder_command(heading_deg, target_heading_deg, params)` — §2.5.
- `step_rudder_servo(rudder_deg, commanded_rudder_deg, params, dt_s)` — one Euler step of
  §2.1.
- `step_yaw_rate(yaw_rate_deg_s, rudder_deg, params, dt_s)` — one Euler step of §2.2.
- `advance(state, target_heading_deg, params, dt_s, substep_s=1.0)` — the full
  composition (autopilot → servo → yaw-rate → heading integration), internally
  sub-stepped per §2.6. This is the ONE function every other caller in this project uses.
- `manoeuvre_time_s(turn_deg, params=None, substep_s=1.0, max_seconds=1200.0,
  tolerance_deg=0.5)` — simulates `advance()` repeatedly from rest until the ship has
  actually turned `turn_deg` degrees, returning the real elapsed time. This is the
  function that replaced every earlier *analytic estimate* of manoeuvre time with a real
  simulated measurement (see §7 and §8).

---

## 4. Integration into the live simulator

**File:** `Basic Simulator/app/simulation.py`.

`VesselConstraints` gained a `kinematics_model: str = "kinematics"` field. Two values:

- `"kinematics"` (default): the ORIGINAL turn-rate slew model — completely unchanged
  behaviour, byte-for-byte, for every existing caller. Any unrecognised/typo'd value also
  falls back to this (verified by `test_unrecognised_kinematics_model_value_falls_back_to_legacy`).
- `"nomoto"`: opt-in. `Simulation._advance_own_kinematics()` branches to a new
  `_advance_own_kinematics_nomoto()` method, which calls `pipeline.nomoto.advance()` once
  per simulation step, using six new `VesselConstraints` fields
  (`nomoto_K_per_s`, `nomoto_T_s`, `nomoto_T_E_s`, `nomoto_rudder_limit_deg`,
  `nomoto_autopilot_kp`, `nomoto_substep_s`) that default to Sawada's own published
  values. **Speed dynamics are untouched** by this flag — Nomoto only replaces heading
  dynamics; the existing accel/decel rate-limiting logic runs identically either way.

Persistent Nomoto state (`self._nomoto_state: NomotoState`) is carried on the
`Simulation` object across steps, initialised once in `__init__`.

CLI flags added to the two run-orchestration scripts, in both cases defaulting to
`"kinematics"` (zero behaviour change unless explicitly requested):

- `Basic Simulator/app/run_baseline_scenario.py --kinematics-model {kinematics,nomoto}`
- `Basic Simulator/app/run_llm_scenario.py --kinematics-model {kinematics,nomoto}`

---

## 5. Verification against the paper

`Basic Simulator/tests/test_nomoto.py` (9 tests) confirms, against the isolated physics
core alone (before any simulator wiring):

| Check | Result | Matches |
|---|---|---|
| Steady turn rate at full rudder = $K\delta_{max}$ | exactly 0.5°/s | user-cited "~0,5°/s stationair" |
| Rudder servo reaches 63.2% of a step command after $t=T_E$ | confirmed (standard 1st-order-lag property) | textbook 1st-order response |
| Yaw rate reaches 63.2% of steady state after $t=T$ | confirmed | textbook 1st-order response |
| 60° turn duration (from rest, full-rudder-saturated autopilot) | **~155 s**, bracketed test asserts 100–260 s | paper's own cited "~2,5–3 minuten" |
| `advance(dt=10, substep=1)` == 10× `advance(dt=1, substep=1)` | exact match (`abs=1e-9`) | confirms sub-stepping doesn't distort the trajectory |

`Basic Simulator/tests/test_simulation_nomoto_flag.py` (4 tests) confirms the
`kinematics_model` flag integration: default stays `"kinematics"`; legacy slew is
completely unaffected; `"nomoto"` produces a measurably different (much smaller, lag-
limited) heading change in the same 10 s step; an unrecognised value falls back safely.

**Turn-time measurements actually used elsewhere in this document/codebase** (computed
directly via `pipeline.nomoto.manoeuvre_time_s(deg, NomotoParams())`, i.e. Sawada's own
K/T/T_E/rudder defaults):

| Turn ordered | Nomoto time (measured) | Legacy slew time (3°/s) | Ratio |
|---:|---:|---:|---:|
| 10° | 60.0 s | 3.3 s | 18.0× |
| 20° | 82.0 s | 6.7 s | 12.3× |
| 30° | 105.0 s | 10.0 s | 10.5× |
| 45° | 138.0 s | 15.0 s | 9.2× |
| 60° | 169.0 s | 20.0 s | 8.45× |
| 90° | 231.0 s | 30.0 s | 7.7× |

This non-linearity (the relative slowdown shrinks as the ordered turn grows) is exactly
why the live prompt (§8) states 3 separate worked examples rather than a single rate —
a single deg/s figure cannot honestly describe a 2nd-order system.

---

## 6. The heading-runaway bug — root cause, fix, and empirical verification

This was the single most consequential finding of this work: **a real, previously
latent control-loop bug**, not a tuning/budget issue, that made every Nomoto-driven
mission diverge from its goal forever.

### 6.1 Symptom

Running the deterministic `baseline_sawada` decision function against Imazu07 with
`kinematics_model="nomoto"`, even with a 3000-step budget (vs. the normal ~200–550), the
ship's heading ran away monotonically (0° → 51° → 66° → 81° → 107° → 119° → oscillating
around 85–112°) while distance-to-goal grew from 22.2 km to 164 km — never converging.

### 6.2 Root cause

`pipeline/oow_agent_spec.py::goal_course_action(own_x, own_y, own_heading, goal_x,
goal_y)` computed the recommended course correction as the bearing difference between
the ship's **current actual heading** and the goal bearing. `Basic
Simulator/app/simulation.py::turn_left()/turn_right()` **add** their `degrees` argument
to `Simulation.target_heading` (a deliberate design choice, predating this work, so that
repeated genuinely-new avoidance orders "stack" instead of being silently absorbed while
a previous turn is still in progress).

Under the legacy slew model, `own_heading` catches up to `target_heading` within about
one decision interval, so the two are nearly always equal at decision time — the bug was
completely latent. Under Nomoto (§5's measured 7.7×–18× slower response), `own_heading`
lags `target_heading` for **many** decisions in a row. Each new decision recomputed a
fresh, large correction from the still-lagging `own_heading` and **stacked it on top of**
an already-in-progress turn — double-counting the same not-yet-completed manoeuvre every
single decision, forever.

This is a generic bug affecting **any** caller of `goal_course_action()` under slow
kinematics — not just the one deterministic baseline it was found with. In particular
it affects the LIVE LLM agent path identically: `Basic Simulator/app/narrate.py` renders
the exact same computation as the "GOAL COURSE CHECK: ... use action X with degrees=Y"
line in the live prompt, which `SYSTEM_OOW_AGENT`'s decision procedure explicitly
instructs the model to follow whenever no real collision risk exists.

### 6.3 Fix

- `app/missions.py::Vessel` gained an optional `target_heading: float | None = None`
  field, mirrored from `Simulation.target_heading` by `Simulation` itself (in `__init__`,
  `turn_left()`, `turn_right()`). Every existing caller that already receives
  `own: Vessel` (narrate, all baselines) gets this "for free" — no signature changes at
  those call sites.
- `goal_course_action()`/`goal_course_check_line()` gained an optional
  `target_heading: float | None = None` parameter; when given, it is used **instead of**
  `own_heading` as the reference compared against the goal bearing. `None` (every
  existing Track-2 training-data generator call site, none of which have a live
  target-heading concept) preserves the OLD behaviour byte-for-byte — **no prompt-hash
  bump, no training-data regeneration needed for this fix**.
- Updated the 3 real live call sites: `app/narrate.py` (the actual live-agent prompt
  text), `app/baselines/ruletree.py`, `app/baselines/sawada.py` (the only 2 of 6
  baselines that call `goal_course_action()` at all — confirmed via grep).

### 6.4 Empirical verification (not just a unit test)

Re-ran the **exact same diverging case** (Imazu07 + UM11_rescaled, `baseline_sawada`,
`kinematics_model="nomoto"`) after the fix:

| | Before fix | After fix | Legacy kinematics (reference) |
|---|---|---|---|
| Imazu07 | `max_steps_reached`, composite 0.200 (3000 steps, never converges) | `reached_goal`, **362 steps**, composite **0.745** | `reached_goal`, 362 steps, composite 0.706 |
| UM11_rescaled | `max_steps_reached`, composite 0.200 | `reached_goal`, **362 steps**, composite **0.790** | `reached_goal`, 362 steps, composite 0.540 |

The fixed Nomoto run needed **exactly the same number of steps as the legacy model** —
proof this was never a step-budget problem, purely the control-loop bug.

A subsequent full sweep of all 6 deterministic baselines × 35 missions (tag
`nomoto_baselines_v1`) confirmed the fix generalises: `ruletree`/`sawada` (the 2
baselines using `goal_course_action()`) score 34/35 and 34/35 `reached_goal` under
Nomoto — matching their own legacy-kinematics performance (35/35, 34/35) almost exactly.
The other 4 baselines (`apf`, `dwa`, `mpc`, `vo`, none of which use
`goal_course_action()`) show large, **separate**, not-yet-investigated regressions under
Nomoto (e.g. `apf`: 27/35 → 2/35 `reached_goal`; `vo`: 34/35 → 2/35) — flagged as a likely
instance of the same bug CLASS in their own independent steering logic, not fixed in this
pass.

---

## 7. Risk-horizon derivation under Nomoto

**File:** `pipeline/oow_agent_spec.py`.

The pre-existing `derive_risk_horizon_s(safe_distance_m, max_turn_deg, own_speed_mps)`
estimates how long own-ship needs to open the safe distance by turning, using a purely
analytic, **instant-turn** formula:

$$t_{manoeuvre} = \frac{d_{safe}}{v \cdot \sin(\theta_{max})}$$

then multiplies by a fixed safety factor $K_{RISK}=3.5$ (`RISK_HORIZON_K`, meant to cover
"deciding, executing, AND confirming separation — not just the bare manoeuvre time").

`derive_risk_horizon_s_nomoto(safe_distance_m, max_turn_deg, own_speed_mps,
nomoto_params=None)` replaces $t_{manoeuvre}$ with a **real simulated** value from
`manoeuvre_time_s(max_turn_deg, nomoto_params)` — the same $K_{RISK}$ multiplier is kept
unchanged (its stated job, covering deciding/executing/confirming margin, remains valid
regardless of which physics computed the base manoeuvre time). The original
`derive_risk_horizon_s()` is completely untouched, still used by the default
`"kinematics"` mode.

---

## 8. Live prompt integration (Phase 3 step 2, part A)

**File:** `Basic Simulator/app/agents.py::build_oow_prompt()`.

**Finding that motivated this work:** before this change, the live prompt's
"physical limits" paragraph ALWAYS used the legacy linear formula
(`example_turn_deg / constraints.turn_rate_deg_s`) regardless of `kinematics_model` —
meaning every Nomoto-flagged sweep run before this fix told the model *"a 90 deg turn
takes about 30s"* while the simulator was actually taking **~231 s** (§5) for that exact
turn. This was a real, quantified prompt/physics mismatch (not a hypothetical risk),
confirmed by direct inspection of the rendered prompt text.

**Fix:** `build_oow_prompt()` now branches on `constraints.kinematics_model`:

- `"kinematics"` (default): **byte-identical** to the pre-existing text (verified by
  `test_kinematics_model_physical_limits_paragraph_unchanged_by_default`).
- `"nomoto"`: states three REAL, `manoeuvre_time_s()`-measured worked examples (30°/60°/
  90°, per §5's table — not a formula, since §5 already shows the relationship is
  non-linear), switches to `derive_risk_horizon_s_nomoto()` (§7) for the risk-horizon
  fact, and wires `constraint_line()`'s `manoeuvre_time_s` parameter (added in this same
  work but left unused until this step) so the model additionally gets a sentence of the
  exact form *"A X degree turn takes about Ys to complete."* for the mission's own
  reference turn angle.

Speed dynamics (accel/decel-in-kt/min) are **identical** in both branches — Nomoto only
replaces heading dynamics.

**Consequence flagged for anyone reading old logs:** every Nomoto-tagged sweep run
*before* this fix landed (`nomoto_baselines_v1`'s LLM-adjacent runs, plus
`nomoto_qwen_base_v1` and `nomoto_qwen_sftdpo_imazu_v1`, both run the same day) used the
stale, incorrect prompt text and should not be used to draw conclusions about "how well
does the model reason about Nomoto" — only about how it behaves when told the WRONG
physics facts.

---

## 9. Ship-dynamics-agnostic training data (Phase 3 step 2, part B — infrastructure only)

### 9.1 Motivation

Training exclusively against Sawada's one exact vessel (K=0.05, T=50 s) risks teaching a
model to overfit to that one ship's specific response curve, rather than learning to
reason FROM whatever manoeuvre-time facts it is given — undermining the goal of an
autopilot that generalises across different vessels. This mirrors a pattern this project
already applies to `safe_distance_m`/`max_turn_deg`/`risk_horizon_s` (sampled per
training row from a weighted range, see `SAFE_DISTANCE_WEIGHTS` etc.) — extended here to
the ship's own manoeuvring parameters.

### 9.2 Real, cited ship profiles (never invented numbers)

**File:** `pipeline/nomoto.py::SHIP_PROFILES`.

| Profile key | Source | Length | $K$ (1/s) | $T$ (s) | $T_E$ (s) | Rudder limit |
|---|---|---:|---:|---:|---:|---:|
| `sawada2021` | Sawada et al. (2021), this project's own live default | 106 m | 0.05 | 50.0 | 2.5 | ±10° |
| `xie2023_small` | Xie et al. (2023), Table 3 | 52.5 m | 0.085 | 4.2 | 1.5 (approximated — paper states a 10°/s max rudder RATE, not a servo time constant) | ±15° |
| `yukun2023_large` | Wang et al. (2023), Table 6, "YU KUN" | 105 m | 0.2257 | 86.815 | 2.5 (inherited from `sawada2021`, not stated in this paper) | ±10° (inherited, not stated in this paper) |

Any value not directly stated in the source paper is explicitly flagged as an
**approximation** in the code comments, never presented as if it were a verified figure.
Note `yukun2023_large` is almost the SAME physical size as `sawada2021` (105 m vs. 106 m)
yet has a markedly different $K$/$T$ — direct evidence that ship LENGTH alone does not
determine manoeuvring response, reinforcing the case for sampling real dynamics rather
than one fixed number.

### 9.3 Held-out generalisation profile

`TRAINING_PROFILE_WEIGHTS = {"sawada2021": 0.7, "yukun2023_large": 0.3,
"xie2023_small": 0.0}` — `xie2023_small` (`HELD_OUT_EVAL_PROFILE`) is **never** sampled
for training rows, reserved exclusively for evaluation, so a model's performance against
it is a genuine unseen-ship generalisation test rather than in-distribution recall.

### 9.4 Sampling functions

**File:** `pipeline/oow_agent_spec.py`.

- `sample_ship_profile(row_id, allow_held_out=False)` — deterministic (SHA-256 of
  `f"{row_id}::ship_profile"`, a DIFFERENT seed suffix than `sample_row_limits()`'s own
  seed so the two samples don't spuriously correlate). `allow_held_out=True` is used only
  when constructing the held-out eval set itself.
- `sample_row_limits_nomoto(row_id, own_speed_mps, allow_held_out_profile=False)` — a
  SIBLING of (not a replacement for) `sample_row_limits()`: reuses its
  `safe_distance_m`/`max_turn_deg`/`decision_interval_s` sampling verbatim, adds
  `{"ship_profile", "manoeuvre_time_s"}` computed via the sampled profile's own
  `derive_risk_horizon_s_nomoto()`/`manoeuvre_time_s()`.

### 9.5 Status

Infrastructure and tests only (commit `4d2c955`) — **the actual Track-2 training-data
regeneration using this sampler has not been run yet**. No existing training file has
been touched by this work.

---

## 10. Reproducing the numbers in this document

```powershell
cd "Auto Pilot"
.venv\Scripts\python.exe -c "
from pipeline.nomoto import manoeuvre_time_s, NomotoParams
p = NomotoParams()
for deg in [10, 20, 30, 45, 60, 90]:
    t_nomoto = manoeuvre_time_s(deg, p)
    t_legacy = deg / 3.0
    print(f'{deg:3d} deg: nomoto={t_nomoto:6.1f}s  legacy={t_legacy:6.1f}s  ratio={t_nomoto/t_legacy:5.2f}x')
"
```

```powershell
cd "Auto Pilot\Basic Simulator"
..\.venv\Scripts\python.exe -m pytest tests\test_nomoto.py tests\test_simulation_nomoto_flag.py tests\test_ship_profiles.py -v
..\.venv\Scripts\python.exe -m app.run_baseline_scenario --missions Imazu07 UM11_rescaled --configs baseline_sawada --kinematics-model nomoto --tag repro_check --force
```

---

## 11. Commit index (chronological, all on `main`)

| Commit | What |
|---|---|
| `c058b50` | `pipeline/nomoto.py` created (Phase 0–1), `VesselConstraints.kinematics_model` flag + `_advance_own_kinematics_nomoto()` (Phase 2), opt-in, default unchanged. |
| `629eb46` | Heading-runaway bug fix (§6): `Vessel.target_heading`, `goal_course_action()`/`goal_course_check_line()`'s `target_heading` param, updated the 3 live call sites. |
| `d90f232` | Nomoto baseline sweep data (`nomoto_baselines_v1`, 210 run logs) confirming the fix + surfacing the separate apf/dwa/mpc/vo regressions. |
| `4045a61` | `--kinematics-model` CLI flag added to `run_llm_scenario.py`. |
| `d29c5f1` | Phase 3 step 2 part A: Nomoto-aware physical-limits paragraph wired into the live prompt (§8). |
| `4d2c955` | Phase 3 step 2 part B: ship-dynamics-agnostic training-data infrastructure (§9). |

---

## 12. Falsification experiment: an independent test of the apf/dwa/mpc/vo regressions (2026-09-26)

Follow-up to §6.4's finding that `apf`/`dwa`/`mpc`/`vo` regress heavily under Nomoto
while `ruletree`/`sawada` do not. A separate, throwaway exploratory script
(`_tmp_impossible_mission_test.py`, repo root — **not part of the pipeline**, not
committed/tested like the rest of this project, kept on disk only for ongoing
experimentation) was used to probe, empirically, WHICH baselines are most vulnerable to
the Nomoto-vs-legacy risk-horizon mismatch (§7), and whether the fine-tuned LLM agent
(which IS Nomoto-aware per §8) can succeed where they fail.

### 12.1 Method

- A 6-target "near-encirclement" mission (bearings ~0°/±55°/±115°/178° around own-ship,
  each on a guaranteed unavoided-collision course constructed via
  `Basic Simulator/app/geometry.py::solve_intercept`, same construction principle as
  `generate_imazu_missions.py`).
- A single "severity" parameter (0.0=mild .. 1.0=severe) scales a scripted, non-COLREG-
  compliant target manoeuvre (each target reverses further toward own-ship's transit
  line late in the encounter) — modelling "give-way vessel does not comply"/"target
  changes plan mid-encounter" categories from a separate, still-informal "impossible
  missions" planning discussion (not yet part of the formal mission schema).
- Each of the 6 `app/baselines/*` decision functions, plus the LLM agent
  (`config="v10_super_colreg_rag"`, `weights="MERGED:OOW-QWEN_v2_sftdpo_fix"` i.e.
  `qwen_sftdpo`), run from the SAME initial conditions with
  `VesselConstraints(kinematics_model="nomoto")` (Sawada's own default K/T/T_E/rudder
  limit, §2).

### 12.2 Result — confirms item 3 below more precisely than "likely the same bug class"

At `severity=0.0` (the MILDEST scripted manoeuvre tested — i.e. this is not even an
adversarial-difficulty result, just what Nomoto's own lag alone already does):

| System | Outcome | min-CPA |
|---|---|---:|
| `baseline_vo` | **FAIL** (never reaches goal) | 246 m |
| `baseline_dwa` | **FAIL** (never reaches goal) | 326 m |
| `baseline_mpc` | **FAIL** (never reaches goal) | 133 m |
| `baseline_apf` | reached_goal | 247 m |
| `baseline_sawada` | reached_goal | 72 m |
| `baseline_ruletree` | reached_goal | 105 m |
| LLM (`qwen_sftdpo`, `v10_super_colreg_rag`) | reached_goal | **401 m** (best of all 7) |

At a separately-tested higher severity (0.90, escalating the same scripted manoeuvre),
`baseline_apf` ALSO failed under Nomoto (min-CPA 54 m, never reached goal) — so of the 6
baselines, only `ruletree`/`sawada` have not yet been broken at any severity tried so far.

### 12.3 Root-cause hypothesis (qualitative — NOT yet verified line-by-line the way §6's
bug was)

- `baseline_vo`/`baseline_apf` (`velocity_obstacle.py`/`potential_field.py`) have NO
  kinematic/forward-simulation model at all — they pick a "best" heading per step and
  implicitly assume it is adopted instantly (confirmed structurally: neither file
  references `pipeline/nomoto.py` or `kinematics_model`).
- `baseline_dwa`/`baseline_mpc` (`dynamic_window.py`/`mpc.py`) DO forward-simulate, but
  their internal rollout still uses the OLD idealised instant-turn-rate model
  (`dynamic_window.py::_project()` assumes the candidate heading is reached at t=0 of its
  own 60 s horizon; `mpc.py::_turn_toward()` is a literal copy of the legacy slew
  formula) — a genuine model-MISMATCH (their own forward model disagrees with what the
  real Nomoto plant will actually do), not merely "no model", which may explain why they
  are just as vulnerable as `vo`/`apf` despite doing more work per decision.
- `baseline_ruletree`/`baseline_sawada` are, so far, the most robust — but (argued, not
  measured) apparently NOT because they model the ship's dynamics correctly: neither
  calls into `pipeline/nomoto.py` either, both still use the plain (non-Nomoto)
  `derive_risk_horizon_s()` per §7. Their robustness is more plausibly explained by a
  simpler control law (repeat the same correction every decision regardless of how the
  real, lagging plant is actually responding) tolerating lag better than a controller
  that COMMITS to a multi-step plan believing it executes instantly (dwa/mpc), or a
  single continuous-field/geometric choice that can get "locked in" to a stale
  assumption (vo/apf).
- The LLM path is the only one of the 7 that receives REAL Nomoto facts (§7/§8) in its
  own decision input every time it is asked — consistent with, but not proof of, it
  having the best margin in this one test.

### 12.4 Status and caveats

- Exploratory/throwaway script, not wired into the sweep/audit tooling — numbers above
  are real single-run measurements, not statistically averaged over multiple
  seeds/missions.
- Only one mission geometry and two severity levels tested under Nomoto so far;
  `sawada`/`ruletree` may still fail at higher severities not yet tried.
- This experiment used the ALREADY-FIXED prompt (post commit `d29c5f1`), so its LLM
  result is not subject to §8's stale-prompt caveat.

---

## 13. Known gaps / next steps (explicitly not done yet)

1. ~~**Track-2 training-data regeneration**~~ — **DONE** (commit `fd72980`). Both
   `pipeline/track2/build_oow_scenarios.py` and `build_oow_scenarios_leo.py` gained an
   opt-in `nomoto`/`--nomoto` flag; re-running their existing production commands
   (`--b3-full-population --nomoto` / `--n 7928 --nomoto --overwrite`) hit their
   respective B3 checkpoints 100% (390/390 and 6993/6993 rows), so **zero new Anthropic
   API calls** were needed. Produced 9 new, parallel `*_nomoto.jsonl` files (row counts
   identical to their existing counterparts — only the rendered constraint-line text
   differs). These new files are **not yet wired into** `train_sft.py`/`train_dpo.py`/
   `train_reflection.py`'s file lists — that wiring is deferred until a decision is made
   about actually training a Nomoto-specific model variant.
2. **Bow-crossing capsule domain** (§1.1, Sawada's own 1.0 NM bow-crossing extension of
   the safety region) — not implemented anywhere in this project yet; still purely a
   circular safe-distance check.
3. **`apf`/`dwa`/`mpc`/`vo` baselines' own Nomoto regressions** (§6.4) — partially
   investigated in §12: `vo`/`dwa`/`mpc` fail even at the mildest tested adversarial
   setting, `apf` fails at a higher severity, `ruletree`/`sawada` not yet broken. Root
   cause is argued (§12.3) but not yet verified line-by-line the way §6's bug was — still
   an open item.
4. **Full re-sweep of the LLM agent under the CORRECTED prompt** (§8's fix invalidates
   every Nomoto-tagged LLM run generated before commit `d29c5f1` on this same day) — not
   yet re-run.
5. **SFT/DPO fine-tuned-model alignment risk** (discussed but not yet measured): the
   `qwen_sftdpo` model's *learned* decision timing was trained against data implicitly
   assuming the legacy model's fast response — under Nomoto its learned "when to start
   acting" may be miscalibrated even where its chosen action type is nominally correct.
   Not yet quantified.
