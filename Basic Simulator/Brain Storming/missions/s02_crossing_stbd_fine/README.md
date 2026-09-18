# Scenario: s02_crossing_stbd_fine — Crossing - target fine on starboard bow

**Applicable rule(s):** Rule 15, Rule 16
**Own-ship role:** give-way

## Description
Target is on own-ship's starboard bow at a fine angle (~20 deg relative). Own-ship must give way.

## Pass / fail criteria
- Own-ship is give-way: she alters course (normally to starboard) and/or speed early.
- Own-ship avoids crossing ahead of the target if circumstances allow.
- Minimum CPA stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (307.8, 845.7), heading 222.9, speed 2.2 m/s — initial relative bearing from own-ship 20 deg, range 900 m; time-to-collision if unavoided: ~205.7 s

## Running
```bash
cd s02_crossing_stbd_fine
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
