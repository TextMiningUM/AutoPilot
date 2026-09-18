# Scenario: s12_converging_cluster — Three-target: converging cluster

**Applicable rule(s):** Rule 14, Rule 15, Rule 16, Rule 17, Rule 8(d)
**Own-ship role:** mixed give-way/stand-on across three simultaneous encounters

## Description
TS1 crosses from starboard, TS2 crosses from port, and TS3 is head-on, all converging on own-ship at roughly the same time - the hardest single-agent test in this set, analogous to the higher-numbered multi-target Imazu cases.

## Pass / fail criteria
- Own-ship finds a single trajectory that keeps a safe CPA from all three targets simultaneously.
- Own-ship does not alter to port toward TS2 while manoeuvring for TS1/TS3.
- Minimum CPA to ALL THREE targets stays above the safe-distance threshold.

## Vehicle configuration

- **opship**: start (0, 0), heading 000, speed 2.5 m/s, goal (0, 2400)
- **ts1**: start (546.4, 651.1), heading 273.5, speed 2.0 m/s — initial relative bearing from own-ship 40 deg, range 850 m; time-to-collision if unavoided: ~273.7 s
- **ts2**: start (-546.4, 651.1), heading 86.5, speed 2.0 m/s — initial relative bearing from own-ship 320 deg, range 850 m; time-to-collision if unavoided: ~273.7 s
- **ts3**: start (0.0, 1200.0), heading 180.0, speed 2.3 m/s — initial relative bearing from own-ship 0 deg, range 1200 m; time-to-collision if unavoided: ~250.0 s

## Running
```bash
cd s12_converging_cluster
chmod +x launch.sh
./launch.sh
# then click DEPLOY on all vehicles in pMarineViewer
```

## Baseline (no avoidance) run
To confirm this is a genuine collision course before testing your avoidance logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the generator script) and re-run — the own-ship and target(s) should pass within a few metres of each other, well under the safe-distance threshold.
