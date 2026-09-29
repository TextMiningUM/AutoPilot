"""Repo-hygiene guard (2026-09-29): deterministic-baseline run logs must ALWAYS use one of
the 3 canonical tags (app.llm_runs.CANONICAL_BASELINE_TAG_BY_KINEMATICS), shared across
EVERY mission -- never a per-mission or ad-hoc tag. This exact mistake (mission-specific
baseline tags cluttering app.sweep_dashboard.py's Baseline-tag dropdown) happened twice for
IMP14 alone. This test fails immediately -- for this mission or any future one -- if it
happens again, regardless of which script or process wrote the offending file.
"""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.baselines import BASELINE_CONFIGS  # noqa: E402
from app.llm_runs import (  # noqa: E402
    CANONICAL_BASELINE_TAG_BY_KINEMATICS, RUNS_DIR, parse_run_filename,
)


def test_canonical_tags_cover_all_three_kinematics_models():
    assert set(CANONICAL_BASELINE_TAG_BY_KINEMATICS) == {"kinematics", "nomoto", "nomoto_v2"}


def test_every_baseline_run_on_disk_uses_a_canonical_tag():
    canonical = set(CANONICAL_BASELINE_TAG_BY_KINEMATICS.values())
    offenders = []
    for path in RUNS_DIR.glob("*.json"):
        if path.name.startswith("_sweep_"):
            continue
        parsed = parse_run_filename(path)
        if parsed["config"] in BASELINE_CONFIGS and parsed["tag"] not in canonical:
            offenders.append(path.name)
    assert not offenders, (
        f"{len(offenders)} baseline run log(s) use a non-canonical tag (must be one of "
        f"{sorted(canonical)}) -- rename/regenerate them under a canonical tag instead of "
        f"a per-mission/ad-hoc one: {offenders[:10]}{'...' if len(offenders) > 10 else ''}")
