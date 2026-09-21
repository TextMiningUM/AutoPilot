"""Verifies Data/missions/Imazu01.json .. Imazu22.json exactly reproduce Sawada et al.
(2021) Table 4 (J. Mar. Sci. Technol. 26(2), 509-524). This is an INDEPENDENT
transcription of the table, not imported from generate_imazu_missions.py -- so a
transcription bug in the generator's own copy wouldn't silently pass here too."""
import json
from pathlib import Path

MISSIONS_DIR = Path(__file__).resolve().parent.parent / "Data" / "missions"

TABLE4 = {
    1:  [(6.000, 0.000, 180.0)],
    2:  [(0.000, 6.000, 90.0)],
    3:  [(-4.200, 0.000, 0.0)],
    4:  [(-4.243, -4.243, 45.0)],
    5:  [(6.000, 0.000, 180.0), (0.000, 6.000, -90.0)],
    6:  [(-5.909, 1.042, -10.0), (-4.243, 4.243, -45.0)],
    7:  [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0)],
    8:  [(6.000, 0.000, 180.0), (0.000, 6.000, -90.0)],
    9:  [(-5.196, 3.000, -30.0), (0.000, 6.000, -90.0)],
    10: [(0.000, 6.000, -90.0), (-5.796, -1.553, 15.0)],
    11: [(0.000, -6.000, 90.0), (-5.196, 3.000, -30.0)],
    12: [(-4.243, 4.243, -45.0), (-5.909, 1.042, -10.0)],
    13: [(6.000, 0.000, 180.0), (-5.909, -1.042, 10.0), (-4.243, -4.243, 45.0)],
    14: [(-5.909, 1.042, -10.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
    15: [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
    16: [(-2.970, -2.970, 45.0), (0.000, -6.000, 90.0), (0.000, 6.000, -90.0)],
    17: [(-4.200, 0.000, 0.0), (-5.909, -1.042, 10.0), (-4.243, 4.243, -45.0)],
    18: [(4.243, 4.243, -135.0), (-5.796, 1.553, -15.0), (-5.196, 3.000, -30.0)],
    19: [(-5.796, -1.553, 15.0), (-5.796, 1.553, -15.0), (4.243, 4.243, -135.0)],
    20: [(-4.200, 0.000, 0.0), (-5.796, 1.553, -15.0), (0.000, 6.000, -90.0)],
    21: [(-5.796, 1.553, -15.0), (-5.796, -1.553, 15.0), (0.000, 6.000, -90.0)],
    22: [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
}
# Case 2 ts1's published heading (90.0) is a one-off typo -- required-bearing-to-origin
# analysis gives 270.0 exactly, the only mismatch across all 50 target instances in
# Table 4 (see generate_imazu_missions.py's module docstring for the full derivation).
CASE2_TS1_CORRECTED_HEADING = 270.0

# ts1 in exactly these cases sits at 4.200 NM from the origin (8.4kt x 0.5h TCPA) instead
# of 6.000 NM (12.0kt x 0.5h) -- derived from the paper's own TCPA=30min design rule, not
# guessed from the literature (see generate_imazu_missions.py's module docstring).
SLOW_TARGET_CASES = {3, 7, 15, 16, 17, 20, 22}

FAST_KT, SLOW_KT = 12.0, 8.4
OWN_START_NM = (0.0, -6.0)   # Sawada (X=-6, Y=0) -> our (x_nm=Y=0, y_nm=X=-6)
GOAL_NM = (0.0, 6.0)         # Sawada (X=6, Y=0)
TOL = 0.001


def _load(case: int) -> dict:
    return json.loads((MISSIONS_DIR / f"Imazu{case:02d}.json").read_text(encoding="utf-8"))


def test_all_22_cases_exist() -> None:
    for case in range(1, 23):
        assert (MISSIONS_DIR / f"Imazu{case:02d}.json").is_file(), case


def test_own_ship_and_goal_identical_across_all_cases() -> None:
    for case in range(1, 23):
        d = _load(case)
        own = d["own_ship"]
        assert own["x_nm"] == OWN_START_NM[0], case
        assert own["y_nm"] == OWN_START_NM[1], case
        assert own["heading_deg"] == 0.0, case
        assert own["speed_kn"] == FAST_KT, case
        assert d["goal"]["x_nm"] == GOAL_NM[0], case
        assert d["goal"]["y_nm"] == GOAL_NM[1], case


def test_targets_match_table4_exactly() -> None:
    for case, targets in TABLE4.items():
        d = _load(case)
        assert len(d["targets"]) == len(targets), case
        for i, (sawada_x, sawada_y, heading) in enumerate(targets):
            t = d["targets"][i]
            # Sawada (X, Y) -> our (x_nm=east=Y, y_nm=north=X).
            expected_x, expected_y = sawada_y, sawada_x
            expected_heading = heading % 360
            if case == 2 and i == 0:
                expected_heading = CASE2_TS1_CORRECTED_HEADING
            expected_speed = SLOW_KT if (case in SLOW_TARGET_CASES and i == 0) else FAST_KT

            assert abs(t["x_nm"] - expected_x) < TOL, (case, i, t)
            assert abs(t["y_nm"] - expected_y) < TOL, (case, i, t)
            assert abs(t["heading_deg"] - expected_heading) < TOL, (case, i, t)
            assert t["speed_kn"] == expected_speed, (case, i, t)


def test_case2_ts1_documents_its_heading_correction() -> None:
    d = _load(2)
    assert "source_note" in d["targets"][0]
    assert "270.0" in d["targets"][0]["source_note"]
    assert "90.0" in d["targets"][0]["source_note"]


def test_slow_targets_are_exactly_the_expected_seven_cases() -> None:
    for case in range(1, 23):
        d = _load(case)
        ts1_speed = d["targets"][0]["speed_kn"]
        if case in SLOW_TARGET_CASES:
            assert ts1_speed == SLOW_KT, case
        else:
            assert ts1_speed == FAST_KT, case
