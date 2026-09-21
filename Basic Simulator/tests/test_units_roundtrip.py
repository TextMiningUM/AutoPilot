"""Phase 6 of the NM/kt units migration: end-to-end regression test.

Loads every Sawada-schema mission (x_nm/y_nm/heading_deg/speed_kn) straight off disk,
lets app.missions.load_mission() convert it to the internal SI representation, runs a
few real simulation steps through that SI-only physics core, then converts back to
NM/kt at a display surface (mirroring narrate.py/streamlit_app.py) and confirms the
round-tripped numbers still match the original on-disk NM/kt values (t=0) and remain
physically sane after stepping (t>0) -- i.e. the load-time and display-time conversions
in app/units.py are exact inverses of each other, not just individually correct.
"""
import json
import math
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.missions import MISSIONS_DIR, list_mission_ids, load_mission  # noqa: E402
from app.simulation import Simulation, VesselConstraints  # noqa: E402
from app.units import kn_to_mps, m_to_nm, mps_to_kn, nm_to_m  # noqa: E402

IMAZU_IDS = [mid for mid in list_mission_ids() if mid.startswith("Imazu")]


@pytest.mark.parametrize("mission_id", IMAZU_IDS)
def test_load_then_display_roundtrips_to_original_nm_kt(mission_id):
    raw = json.loads((MISSIONS_DIR / f"{mission_id}.json").read_text(encoding="utf-8"))
    mission = load_mission(mission_id)

    assert m_to_nm(mission.own_ship.x) == pytest.approx(raw["own_ship"]["x_nm"], abs=1e-6)
    assert m_to_nm(mission.own_ship.y) == pytest.approx(raw["own_ship"]["y_nm"], abs=1e-6)
    assert mission.own_ship.heading == pytest.approx(raw["own_ship"]["heading_deg"], abs=1e-6)
    assert mps_to_kn(mission.own_ship.speed) == pytest.approx(raw["own_ship"]["speed_kn"], abs=1e-6)

    assert m_to_nm(mission.goal[0]) == pytest.approx(raw["goal"]["x_nm"], abs=1e-6)
    assert m_to_nm(mission.goal[1]) == pytest.approx(raw["goal"]["y_nm"], abs=1e-6)

    for tgt, raw_tgt in zip(mission.targets, raw["targets"]):
        assert m_to_nm(tgt.x) == pytest.approx(raw_tgt["x_nm"], abs=1e-6)
        assert m_to_nm(tgt.y) == pytest.approx(raw_tgt["y_nm"], abs=1e-6)
        assert mps_to_kn(tgt.speed) == pytest.approx(raw_tgt["speed_kn"], abs=1e-6)


@pytest.mark.parametrize("mission_id", IMAZU_IDS[:5])
def test_sim_steps_then_display_conversion_stays_consistent(mission_id):
    """Runs a few SI-only physics steps, then re-derives NM/kt purely from the stepped SI
    state and cross-checks it against an independent straight-line NM/kt projection --
    guards against the physics core silently drifting into a unit mismatch (e.g. metres
    accidentally treated as NM somewhere) once actual motion is involved, not just at the
    static t=0 load."""
    mission = load_mission(mission_id)
    sim = Simulation(mission, VesselConstraints())
    for _ in range(5):
        sim.step(dt=10.0)

    heading_rad = math.radians(mission.own_ship.heading)
    expected_x_nm = m_to_nm(mission.own_ship.x) + mps_to_kn(mission.own_ship.speed) * (50.0 / 3600.0) * math.sin(heading_rad)
    expected_y_nm = m_to_nm(mission.own_ship.y) + mps_to_kn(mission.own_ship.speed) * (50.0 / 3600.0) * math.cos(heading_rad)

    assert m_to_nm(sim.own.x) == pytest.approx(expected_x_nm, abs=1e-3)
    assert m_to_nm(sim.own.y) == pytest.approx(expected_y_nm, abs=1e-3)


def test_units_conversions_are_exact_inverses():
    for value in [0.0, 1.0, 6.0, 12.0, 8.4, 123.456]:
        assert m_to_nm(nm_to_m(value)) == pytest.approx(value, abs=1e-9)
        assert mps_to_kn(kn_to_mps(value)) == pytest.approx(value, abs=1e-9)
