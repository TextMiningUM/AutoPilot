# Scenario: s05_crossing_port_broad — Crossing - target broad on port bow (own-ship stand-on)

**Applicable rule(s):** Rule 15, Rule 17
**Own-ship role:** stand-on

## Description
Target is broad on own-ship's port bow (~60 deg relative to port). The target does not give way (per the classic Imazu-style convention that targets run straight), so own-ship must hold course/speed under Rule 17(a)(i) and only act under Rule 17(a)(ii)/(b) if the target clearly fails to act and collision becomes otherwise unavoidable.

## Pass / fail criteria
- Own-ship holds course and speed for a meaningful initial period (does not manoeuvre prematurely).
- If the target never gives way, own-ship eventually takes late avoiding action under Rule 17(b) rather than colliding.
- Minimum CPA stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (-779.4, 450.0), heading 40.2, speed 2.2 m/s — initial relative bearing from own-ship 300 deg, range 900 m; time-to-collision if unavoided: ~548.6 s

## Running
```bash
cd s05_crossing_port_broad
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
