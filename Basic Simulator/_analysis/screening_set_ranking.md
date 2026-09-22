# Screening-set mission ranking (all missions in the current sweep)

Tag: `units_v1`. 20 missions with run data at generation time.

| Rank | Mission | Type | Configs | Composite std | Compliance std | Verdict disagreement | Combined score | Mean composite | Verdicts seen | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Imazu07 | multi_target | 8/8 | 0.2703 | 0.3307 | 3 | **0.877** | 0.518 | FAIL -- collision occurred, FAIL -- did not reach the goal, PASS |  |
| 2 | Imazu08 | multi_target | 8/8 | 0.3427 | 0.348 | 2 | **0.8333** | 0.57 | FAIL -- collision occurred, PASS |  |
| 3 | Imazu04 | crossing_stand_on | 8/8 | 0.24 | 0.3172 | 3 | **0.8168** | 0.486 | FAIL -- collision occurred, FAIL -- did not reach the goal, PASS |  |
| 4 | Imazu05 | multi_target | 8/8 | 0.3331 | 0.3307 | 2 | **0.7962** | 0.551 | FAIL -- collision occurred, PASS |  |
| 5 | UM10 | multi_target | 5/8 | 0.317 | 0.2236 | 3 | **0.7919** | 0.318 | FAIL -- collision occurred, FAIL -- did not reach the goal, PASS | LOW COVERAGE (5/8 configs) |
| 6 | Imazu10 | multi_target | 3/8 | 0.3333 | 0.3118 | 2 | **0.7701** | 0.436 | FAIL -- did not reach the goal, PASS | LOW COVERAGE (3/8 configs) |
| 7 | UM09 | overtaking | 8/8 | 0.2308 | 0.2778 | 3 | **0.7494** | 0.255 | FAIL -- collision occurred, FAIL -- did not reach the goal, PASS |  |
| 8 | Imazu01 | head_on | 8/8 | 0.309 | 0.3172 | 2 | **0.7444** | 0.505 | FAIL -- collision occurred, PASS |  |
| 9 | UM08 | overtaking | 8/8 | 0.2404 | 0.248 | 3 | **0.7211** | 0.493 | FAIL -- collision occurred, FAIL -- did not reach the goal, PASS |  |
| 10 | UM04 | crossing_give_way | 8/8 | 0.2429 | 0.2421 | 3 | **0.7163** | 0.501 | FAIL -- collision occurred, FAIL -- did not reach the goal, PASS |  |
| 11 | Imazu02 | crossing_give_way | 8/8 | 0.2524 | 0.3172 | 2 | **0.6671** | 0.579 | FAIL -- collision occurred, PASS |  |
| 12 | Imazu03 | overtaking | 8/8 | 0.1961 | 0.3172 | 2 | **0.5901** | 0.691 | FAIL -- did not reach the goal, PASS |  |
| 13 | UM06 | crossing_stand_on | 8/8 | 0.2043 | 0.3046 | 2 | **0.5838** | 0.318 | FAIL -- did not reach the goal, PASS |  |
| 14 | Imazu06 | multi_target | 8/8 | 0.1593 | 0.3307 | 2 | **0.5586** | 0.56 | FAIL -- did not reach the goal, PASS |  |
| 15 | UM07 | crossing_stand_on | 8/8 | 0.211 | 0.2142 | 2 | **0.4673** | 0.558 | FAIL -- did not reach the goal, PASS |  |
| 16 | UM03 | crossing_give_way | 8/8 | 0.1484 | 0.2633 | 2 | **0.45** | 0.569 | FAIL -- did not reach the goal, PASS |  |
| 17 | UM05 | crossing_give_way | 8/8 | 0.1842 | 0.174 | 2 | **0.3747** | 0.507 | FAIL -- did not reach the goal, PASS |  |
| 18 | UM02 | quiet | 8/8 | 0.1097 | 0.3248 | 1 | **0.316** | 0.939 | PASS | ALL CONFIGS PASS |
| 19 | UM01 | quiet | 8/8 | 0.0988 | 0.3292 | 1 | **0.3072** | 0.953 | PASS | ALL CONFIGS PASS |
| 20 | Imazu09 | multi_target | 8/8 | 0.1797 | 0.1083 | 2 | **0.2772** | 0.428 | FAIL -- did not reach the goal, PASS |  |

## Missions where every available config failed

(none)

## Missions where every available config passed

- `UM01` (quiet): ['PASS']
- `UM02` (quiet): ['PASS']
