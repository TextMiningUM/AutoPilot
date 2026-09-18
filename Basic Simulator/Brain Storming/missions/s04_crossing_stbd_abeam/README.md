# Scenario: s04_crossing_stbd_abeam — Crossing - target near starboard beam

**Applicable rule(s):** Rule 15, Rule 16
**Own-ship role:** give-way

## Description
Target is nearly abeam on own-ship's starboard side (~90 deg relative). Own-ship must give way.

## Pass / fail criteria
- Own-ship is give-way and passes astern of the target.
- Minimum CPA stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (800.0, 0.0), heading 321.4, speed 3.2 m/s — initial relative bearing from own-ship 90 deg, range 800 m; time-to-collision if unavoided: ~400.5 s

## Running
```bash
cd s04_crossing_stbd_abeam
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
