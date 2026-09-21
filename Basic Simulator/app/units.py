"""Single source of truth for unit conversion between the Basic Simulator's external
(human/paper-facing) units -- nautical miles and knots, matching Sawada et al. (2021),
the canonical Imazu-problem reference paper -- and its INTERNAL physics/kinematics units
-- metres and metres/second.

Only the edges of the system should ever call these: mission loading (NM/kt -> m/m/s,
before anything touches the simulation engine) and display surfaces (m/m/s -> NM/kt, for
narrate()'s situation_report, evaluation summaries, and the Streamlit UI). The physics
core itself (Simulation, VesselConstraints, narrate.cpa_tcpa/classify_encounter, ...)
stays entirely in metres/m/s/deg/s -- never rewritten in NM/kt -- exactly the same
one-source-of-truth lesson as the mission.targets staleness bug: a single, well-tested
conversion module instead of scattered, potentially-inconsistent inline conversions.

1 nautical mile = 1852 m exactly (international definition); 1 knot = 1 NM/hour.
"""
from __future__ import annotations

NM_TO_M = 1852.0
SECONDS_PER_HOUR = 3600.0


def nm_to_m(nm: float) -> float:
    """Nautical miles -> metres."""
    return nm * NM_TO_M


def m_to_nm(m: float) -> float:
    """Metres -> nautical miles."""
    return m / NM_TO_M


def kn_to_mps(kn: float) -> float:
    """Knots (NM/hour) -> metres/second."""
    return kn * NM_TO_M / SECONDS_PER_HOUR


def mps_to_kn(mps: float) -> float:
    """Metres/second -> knots (NM/hour)."""
    return mps * SECONDS_PER_HOUR / NM_TO_M
