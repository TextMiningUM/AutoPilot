"""Regression test (2026-09-28) for app.sweep_dashboard's tag->kinematics_model inference
now that "nomoto_v2" exists -- it's a substring superset of "nomoto", so a naive
"nomoto" in tag.lower() check would misclassify a nomoto_v2-tagged run as plain "nomoto"."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.sweep_dashboard import _baseline_tag_kinematics_model, KINEMATICS_MODELS  # noqa: E402


def test_kinematics_models_has_three_values():
    assert KINEMATICS_MODELS == ["kinematics", "nomoto", "nomoto_v2"]


def test_nomoto_v2_tag_is_not_misread_as_plain_nomoto():
    assert _baseline_tag_kinematics_model("nomoto_v2") == "nomoto_v2"
    assert _baseline_tag_kinematics_model("some_nomoto_v2_reference") == "nomoto_v2"


def test_plain_nomoto_tag_still_reads_as_nomoto():
    assert _baseline_tag_kinematics_model("baseline_Nomoto_all") == "nomoto"


def test_plain_tag_reads_as_kinematics():
    assert _baseline_tag_kinematics_model("baseline") == "kinematics"
