# Scenario: s01_head_on — Head-on

**Applicable rule(s):** Rule 14
**Own-ship role:** mutual (both alter to starboard)

## Description
Own-ship meets a single target nearly dead ahead on a reciprocal course.

## Pass / fail criteria
- Own-ship alters course to starboard (not port).
- Minimum CPA (closest point of approach) stays above the configured safe-distance threshold.
- Own-ship returns to her original track/heading after the target is past and clear.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (76.7, 1097.3), heading 188.0, speed 2.5 m/s — initial relative bearing from own-ship 4 deg, range 1100 m; time-to-collision if unavoided: ~220.5 s

## Running
```bash
cd s01_head_on
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
