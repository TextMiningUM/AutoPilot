"""Regression tests (2026-09-28) for app.missions.duplicate_of() and the EXACT-duplicate
removal in generate_imazu_missions.py -- Imazu05/Imazu08 and Imazu15/Imazu22 were confirmed
byte-identical geometric duplicates in Sawada et al. (2021)'s own Table 4 (see
generate_imazu_missions.py's REMOVED_CASES comment); Imazu08/Imazu22 were removed entirely
(not just tagged) per explicit user request, across missions AND all existing run results
(every model, including baselines)."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.missions import MISSIONS_DIR, duplicate_of, list_mission_ids, load_mission  # noqa: E402


def test_imazu08_and_imazu22_no_longer_exist():
    ids = list_mission_ids()
    assert "Imazu08" not in ids
    assert "Imazu22" not in ids


def test_imazu08_and_imazu22_files_are_gone():
    assert not (MISSIONS_DIR / "Imazu08.json").exists()
    assert not (MISSIONS_DIR / "Imazu22.json").exists()


def test_imazu12_is_tagged_duplicate_of_imazu06():
    """Same two targets as Imazu06, listed in reversed (ts1/ts2-swapped) order in Sawada
    et al.'s own Table 4 -- geometrically the same combined scenario regardless of label.
    KEPT (not removed) since it's only equivalent up to relabelling, not byte-identical."""
    assert duplicate_of("Imazu12") == "Imazu06"


def test_imazu05_itself_is_not_a_duplicate():
    assert duplicate_of("Imazu05") is None


def test_untagged_mission_is_not_a_duplicate():
    assert duplicate_of("Imazu01") is None


def test_remaining_imazu_missions_still_load():
    m5 = load_mission("Imazu05")
    m15 = load_mission("Imazu15")
    assert len(m5.targets) == 2
    assert len(m15.targets) == 3

