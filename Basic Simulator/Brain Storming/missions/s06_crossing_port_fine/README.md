# Scenario: s06_crossing_port_fine — Crossing - target fine on port bow (own-ship stand-on)

**Applicable rule(s):** Rule 15, Rule 17
**Own-ship role:** stand-on

## Description
Target is fine on own-ship's port bow (~20 deg relative to port). Own-ship is stand-on.

## Pass / fail criteria
- Own-ship holds course and speed initially.
- Own-ship does not alter to port toward the target if she must eventually act (Rule 17(c)).
- Minimum CPA stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (-307.8, 845.7), heading 137.1, speed 2.2 m/s — initial relative bearing from own-ship 340 deg, range 900 m; time-to-collision if unavoided: ~205.7 s

## Running
```bash
cd s06_crossing_port_fine
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
