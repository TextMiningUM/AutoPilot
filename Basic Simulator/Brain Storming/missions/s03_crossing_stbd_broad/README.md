# Scenario: s03_crossing_stbd_broad — Crossing - target broad on starboard bow

**Applicable rule(s):** Rule 15, Rule 16
**Own-ship role:** give-way

## Description
Target is broad on own-ship's starboard bow (~65 deg relative). Own-ship must give way.

## Pass / fail criteria
- Own-ship is give-way: early, substantial action to keep well clear.
- Minimum CPA stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (815.7, 380.4), heading 305.6, speed 2.6 m/s — initial relative bearing from own-ship 65 deg, range 900 m; time-to-collision if unavoided: ~386.0 s

## Running
```bash
cd s03_crossing_stbd_broad
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
