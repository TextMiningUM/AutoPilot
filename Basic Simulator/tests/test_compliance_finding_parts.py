"""Regression test for app/evaluation.py's compliance_finding_parts() -- the single place
that turns one compliance.breakdown entry into (code, label, at, deduction), handling both
the current dict shape and pre-2026-09-24 bare [code, step_or_detail, deduction] entries
from already-generated runs (never migrated -- see evaluate_run.py's COMPLIANCE_LABELS
docstring)."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.evaluation import compliance_finding_parts  # noqa: E402


def test_new_dict_shaped_entry() -> None:
    entry = {"code": "B_wrong_direction", "label": "Wrong Turn Direction", "at": 0.0, "deduction": -0.15}
    assert compliance_finding_parts(entry) == ("B_wrong_direction", "Wrong Turn Direction", 0.0, -0.15)


def test_legacy_list_shaped_entry_gets_its_label_looked_up() -> None:
    entry = ["P_wrong_side_pass", "ts1", -0.3]
    assert compliance_finding_parts(entry) == ("P_wrong_side_pass", "Wrong Passing Side", "ts1", -0.3)


def test_legacy_tuple_shaped_entry_with_unknown_code_falls_back_to_the_code_itself() -> None:
    entry = ("some_future_code", 5.0, -0.1)
    assert compliance_finding_parts(entry) == ("some_future_code", "some_future_code", 5.0, -0.1)
