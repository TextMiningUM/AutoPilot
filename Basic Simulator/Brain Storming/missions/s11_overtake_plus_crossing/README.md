# Scenario: s11_overtake_plus_crossing — Two-target: overtaking plus crossing from starboard

**Applicable rule(s):** Rule 13, Rule 15, Rule 16
**Own-ship role:** give-way to both (different rules)

## Description
A slow target is ahead on own-ship's track (overtaking case) while a second target simultaneously crosses from own-ship's starboard bow (crossing case).

## Pass / fail criteria
- Own-ship keeps clear of the overtaken target (Rule 13) while also giving way to the crossing target (Rule 15/16) without conflating the two into a single miscategorised manoeuvre.
- Minimum CPA to BOTH targets stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (0.0, 700.0), heading 0.0, speed 1.0 m/s — initial relative bearing from own-ship 0.0 deg, range 700 m; time-to-collision if unavoided: ~466.7 s
- **ts2**: start (689.4, 578.5), heading 303.2, speed 2.0 m/s — initial relative bearing from own-ship 50 deg, range 900 m; time-to-collision if unavoided: ~412.2 s

## Running
```bash
cd s11_overtake_plus_crossing
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
