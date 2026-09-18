# Scenario: s10_headon_plus_crossing — Two-target: head-on plus crossing from starboard

**Applicable rule(s):** Rule 14, Rule 15, Rule 16, Rule 8(d)
**Own-ship role:** mutual with TS1 / give-way to TS2

## Description
TS1 is head-on; TS2 crosses from own-ship's starboard bow at the same time. Both encounters call for a starboard alteration, testing whether a single coherent manoeuvre satisfies both.

## Pass / fail criteria
- Own-ship alters to starboard (satisfying both Rule 14 and Rule 15/16 simultaneously).
- Minimum CPA to BOTH targets stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (0.0, 1100.0), heading 180.0, speed 2.5 m/s — initial relative bearing from own-ship 0 deg, range 1100 m; time-to-collision if unavoided: ~220.0 s
- **ts2**: start (779.4, 450.0), heading 310.3, speed 2.3 m/s — initial relative bearing from own-ship 60 deg, range 900 m; time-to-collision if unavoided: ~444.2 s

## Running
```bash
cd s10_headon_plus_crossing
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
