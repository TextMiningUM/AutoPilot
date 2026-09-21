"""Round-trip tests for Basic Simulator/app/units.py -- the single source of truth for
NM<->m and knot<->m/s conversion (see units.py's docstring for why this exists as one
central, well-tested module instead of scattered inline conversions -- the same
one-source-of-truth lesson as the mission.targets staleness bug).
"""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.units import NM_TO_M, kn_to_mps, m_to_nm, mps_to_kn, nm_to_m  # noqa: E402

TOL = 1e-9


def test_nm_to_m_known_value() -> None:
    # 1 NM = 1852 m exactly (international definition).
    assert nm_to_m(1.0) == 1852.0
    assert nm_to_m(6.0) == 11112.0


def test_kn_to_mps_known_value() -> None:
    # 1 kt = 1 NM/hour = 1852 m / 3600 s.
    assert abs(kn_to_mps(1.0) - (NM_TO_M / 3600.0)) < TOL
    # Sawada et al. (2021) nominal speed 12.0 kt.
    assert abs(kn_to_mps(12.0) - 6.173333333) < 1e-6
    # Sawada et al. (2021) overtaken-target speed 8.4 kt.
    assert abs(kn_to_mps(8.4) - 4.321333333) < 1e-6


def test_nm_m_roundtrip() -> None:
    for x in (0.0, 1.0, -6.0, 6.009, 12345.678, -0.001):
        assert abs(m_to_nm(nm_to_m(x)) - x) < TOL, x
        assert abs(nm_to_m(m_to_nm(x)) - x) < TOL, x


def test_kn_mps_roundtrip() -> None:
    for x in (0.0, 12.0, 8.4, 20.0, -3.5, 0.001):
        assert abs(mps_to_kn(kn_to_mps(x)) - x) < TOL, x
        assert abs(kn_to_mps(mps_to_kn(x)) - x) < TOL, x
