# Scenario: s08_overtaking_os_stands_on — Overtaking - own-ship is overtaken (stand-on)

**Applicable rule(s):** Rule 13, Rule 17
**Own-ship role:** stand-on (being overtaken)

## Description
A faster target approaches from astern of own-ship and must keep clear as the overtaking vessel.

## Pass / fail criteria
- Own-ship holds course and speed while being overtaken, absent evidence the target is not keeping clear.
- Minimum CPA stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (0.0, -600.0), heading 0.0, speed 3.5 m/s — initial relative bearing from own-ship 180.0 deg, range 600 m; time-to-collision if unavoided: ~600.0 s

## Running
```bash
cd s08_overtaking_os_stands_on
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
