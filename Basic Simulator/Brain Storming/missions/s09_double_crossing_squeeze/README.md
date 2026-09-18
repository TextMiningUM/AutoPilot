# Scenario: s09_double_crossing_squeeze — Two-target: crossing from both sides

**Applicable rule(s):** Rule 15, Rule 16, Rule 17, Rule 8(d)
**Own-ship role:** give-way to TS1 / stand-on to TS2

## Description
TS1 crosses from own-ship's starboard bow (give-way) while TS2 simultaneously crosses from own-ship's port bow (stand-on). Own-ship must satisfy both encounters with one coherent manoeuvre plan.

## Pass / fail criteria
- Own-ship's manoeuvre for TS1 (starboard give-way) does not create a new close-quarters situation with TS2 (Rule 8(d)/8(c)).
- Minimum CPA to BOTH targets stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (636.4, 636.4), heading 278.5, speed 2.2 m/s — initial relative bearing from own-ship 45 deg, range 900 m; time-to-collision if unavoided: ~292.5 s
- **ts2**: start (-636.4, 636.4), heading 81.5, speed 2.2 m/s — initial relative bearing from own-ship 315 deg, range 900 m; time-to-collision if unavoided: ~292.5 s

## Running
```bash
cd s09_double_crossing_squeeze
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
