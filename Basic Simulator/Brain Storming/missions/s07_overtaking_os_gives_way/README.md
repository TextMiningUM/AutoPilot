# Scenario: s07_overtaking_os_gives_way — Overtaking - own-ship overtakes a slower target

**Applicable rule(s):** Rule 13
**Own-ship role:** give-way (overtaking)

## Description
A slower target is on own-ship's track ahead of her. Own-ship closes from astern and must keep clear as the overtaking vessel until finally past and clear.

## Pass / fail criteria
- Own-ship's initial approach is recognisably from more than 22.5 deg abaft the target's beam.
- Own-ship keeps clear of the target throughout the pass (does not cut in front).
- Minimum CPA stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (0.0, 700.0), heading 0.0, speed 1.0 m/s — initial relative bearing from own-ship 0.0 deg, range 700 m; time-to-collision if unavoided: ~466.7 s

## Running
```bash
cd s07_overtaking_os_gives_way
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
