"""Captain walking-skeleton Phase 1 tests: discrete-event scheduler + mission-sim clock
(design_captain_missions.md Sec 13.A.1/13.A.3/13.A.7/13.A.9). No GPU/API key needed."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.mission_sim import (  # noqa: E402
    EventScheduler, EventType, MissionSim, RestHoursLedger, calibrate_fuel_rate,
    haversine_nm, route_distance_nm, schedule_routine_reports, watch_roster_off_hours,
)
from app.missions import load_mission  # noqa: E402
from app.simulation import Simulation  # noqa: E402
from pipeline.captain_types import MissionOrder  # noqa: E402

# Sec 8.2's own worked example numbers, reused here for a clean round-trip calibration test.
_FUEL_TONNES = 850.0
_FUEL_RESERVE_PCT = 15.0
_ROUTE_NM = 600.0
_SOA_KN = 12.0


# --- EventScheduler -----------------------------------------------------------------

def test_scheduler_pops_by_timestamp_first():
    s = EventScheduler()
    s.schedule(100.0, EventType.MONITOR_TICK, "late")
    s.schedule(50.0, EventType.MONITOR_TICK, "early")
    assert s.pop_next().kind == "early"
    assert s.pop_next().kind == "late"


def test_scheduler_breaks_same_timestamp_ties_by_rank():
    s = EventScheduler()
    s.schedule(10.0, EventType.MONITOR_TICK, "tick")
    s.schedule(10.0, EventType.ROUTINE_REPORT, "report")
    s.schedule(10.0, EventType.EMERGENCY_TRIGGER, "emergency")
    s.schedule(10.0, EventType.MANDATORY_TRIGGER, "mandatory")
    order = [s.pop_next().kind for _ in range(4)]
    assert order == ["emergency", "mandatory", "report", "tick"]


def test_scheduler_breaks_same_timestamp_same_rank_ties_by_insertion_order():
    s = EventScheduler()
    s.schedule(10.0, EventType.MONITOR_TICK, "first")
    s.schedule(10.0, EventType.MONITOR_TICK, "second")
    assert s.pop_next().kind == "first"
    assert s.pop_next().kind == "second"


def test_scheduler_pop_order_is_deterministic_across_repeated_runs():
    def build_and_drain() -> list[str]:
        s = EventScheduler()
        s.schedule(5.0, EventType.WAYPOINT_ARRIVAL, "a")
        s.schedule(5.0, EventType.WORLD_RESPONDER_TIMER, "b")
        s.schedule(2.0, EventType.MONITOR_TICK, "c")
        out = []
        while len(s):
            out.append(s.pop_next().kind)
        return out
    assert build_and_drain() == build_and_drain()


def test_scheduler_peek_does_not_remove():
    s = EventScheduler()
    s.schedule(1.0, EventType.MONITOR_TICK, "only")
    assert s.peek_next_timestamp() == 1.0
    assert len(s) == 1
    assert s.pop_next().kind == "only"
    assert s.peek_next_timestamp() is None


def test_schedule_routine_reports_stays_within_mission_duration():
    s = EventScheduler()
    events = schedule_routine_reports(s, reporting_interval_hours=24.0, mission_duration_s=60 * 3600.0)
    assert [e.timestamp_s for e in events] == [24 * 3600.0, 48 * 3600.0]  # 72h would exceed 60h


# --- Great-circle distance -----------------------------------------------------------

def test_haversine_one_degree_latitude_is_about_sixty_nm():
    assert haversine_nm((0.0, 0.0), (1.0, 0.0)) == pytest.approx(60.04, rel=0.01)


def test_route_distance_sums_consecutive_legs():
    waypoints = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert route_distance_nm(waypoints) == pytest.approx(2 * haversine_nm((0.0, 0.0), (1.0, 0.0)))


# --- Fuel calibration -----------------------------------------------------------------

def test_fuel_calibration_round_trips_the_budget_at_constant_soa():
    model = calibrate_fuel_rate(_FUEL_TONNES, _FUEL_RESERVE_PCT, _ROUTE_NM, _SOA_KN)
    fuel_budget = _FUEL_TONNES * (1 - _FUEL_RESERVE_PCT / 100.0)
    t_h = _ROUTE_NM / _SOA_KN
    assert model.fuel_rate_tonnes_per_h(_SOA_KN) * t_h == pytest.approx(fuel_budget, rel=1e-9)


def test_fuel_rate_increases_with_speed():
    model = calibrate_fuel_rate(_FUEL_TONNES, _FUEL_RESERVE_PCT, _ROUTE_NM, _SOA_KN)
    assert model.fuel_rate_tonnes_per_h(_SOA_KN * 2) > model.fuel_rate_tonnes_per_h(_SOA_KN)


# --- Rest-hours ledger ----------------------------------------------------------------

def test_watch_roster_off_hours_one_full_cycle():
    assert watch_roster_off_hours(0.0, 12 * 3600.0) == pytest.approx(8.0)


def test_watch_roster_off_hours_zero_during_on_watch_phase():
    assert watch_roster_off_hours(0.0, 4 * 3600.0) == pytest.approx(0.0)


def test_oow_ledger_meets_stcw_minimums_with_no_disruptions():
    ledger = RestHoursLedger.for_oow_watch_roster()
    now_s = 7 * 24 * 3600.0
    assert ledger.rest_hours_in_trailing_window(now_s, 24.0) == pytest.approx(16.0)
    assert not ledger.breaches_stcw_minimum(now_s)


def test_oow_ledger_disruption_can_breach_stcw_minimum():
    ledger = RestHoursLedger.for_oow_watch_roster()
    now_s = 24 * 3600.0
    ledger.add_disruption(at_s=12 * 3600.0, hours_lost=10.0)  # eats nearly all of the day's 16h rest
    assert ledger.breaches_stcw_minimum(now_s)


def test_captain_ledger_starts_fully_rested_and_disruption_reduces_it():
    ledger = RestHoursLedger.for_captain()
    now_s = 24 * 3600.0
    assert ledger.rest_hours_in_trailing_window(now_s, 24.0) == pytest.approx(24.0)
    ledger.add_disruption(at_s=5 * 3600.0, hours_lost=3.0)
    assert ledger.rest_hours_in_trailing_window(now_s, 24.0) == pytest.approx(21.0)


# --- MissionSim -------------------------------------------------------------------------

def _build_sim(**overrides) -> MissionSim:
    kwargs = dict(
        waypoints=[(51.95, 4.14), (52.3, 4.3), (52.96, 4.75)],
        soa_kn=_SOA_KN, fuel_tonnes_at_departure=_FUEL_TONNES,
        fuel_reserve_margin_pct=_FUEL_RESERVE_PCT, t0_utc="2026-10-12T18:00:00Z",
        dt_mission_s=600.0,
    )
    kwargs.update(overrides)
    return MissionSim(**kwargs)


def test_missionsim_requires_at_least_two_waypoints():
    with pytest.raises(ValueError):
        _build_sim(waypoints=[(0.0, 0.0)])


def test_missionsim_reaches_destination_after_stepping_the_full_route():
    sim = _build_sim(waypoints=[(0.0, 0.0), (0.0, _ROUTE_NM / 60.0)])  # ~600nm due east
    assert sim.total_route_distance_nm == pytest.approx(_ROUTE_NM, rel=0.01)
    steps = 0
    while not sim.reached_destination() and steps < 1000:
        sim.step()
        steps += 1
    assert sim.reached_destination()
    assert sim.remaining_distance_nm() == pytest.approx(0.0, abs=1e-6)


def test_missionsim_fuel_consumption_matches_calibration_at_constant_soa():
    sim = _build_sim(waypoints=[(0.0, 0.0), (0.0, _ROUTE_NM / 60.0)])
    while not sim.reached_destination():
        sim.step()
    fuel_budget = _FUEL_TONNES * (1 - _FUEL_RESERVE_PCT / 100.0)
    assert sim.state.fuel_tonnes == pytest.approx(_FUEL_TONNES - fuel_budget, abs=1e-6)


def test_missionsim_eta_is_inf_when_stopped_and_finite_otherwise():
    sim = _build_sim()
    assert sim.eta_s() == pytest.approx(sim.remaining_distance_nm() / _SOA_KN * 3600.0)
    sim.state.current_speed_kn = 0.0
    assert sim.eta_s() == float("inf")


def test_missionsim_current_utc_advances_with_elapsed_time():
    sim = _build_sim()
    before = sim.current_utc()
    sim.step(600.0)
    after = sim.current_utc()
    assert (after - before).total_seconds() == pytest.approx(600.0)


def test_missionsim_advance_to_lands_exactly_on_target_without_overshoot():
    sim = _build_sim(dt_mission_s=600.0)
    sim.advance_to(950.0)  # not a multiple of dt_mission_s
    assert sim.state.elapsed_s == pytest.approx(950.0)


def test_missionsim_run_to_next_scheduled_advances_and_pops():
    sim = _build_sim(reporting_interval_hours=1.0)  # short interval -- default route is well under 24h
    event = sim.run_to_next_scheduled()
    assert event.kind == "routine_report_1"
    assert sim.state.elapsed_s == pytest.approx(1 * 3600.0)


def test_missionsim_from_mission_order_builds_a_consistent_sim():
    order = MissionOrder(
        mission_id="MSN-TEST-0001", issued_by="Fleet Operations Centre", issued_at="2026-10-12T14:00:00Z",
        vessel="MV Example Trader", t0_utc="2026-10-12T18:00:00Z", dt_mission_s=600.0,
        situation={}, goal="Deliver cargo intact", success_criteria=["cargo delivered with zero damage"],
        waypoints=[(0.0, 0.0), (0.0, _ROUTE_NM / 60.0)], speed_of_advance_kn=_SOA_KN, restricted_zones=[],
        rules_of_conduct="COLREG 1972 + company SMS",
        admin_logistics={"resources": {"fuel_tonnes": _FUEL_TONNES, "fuel_reserve_margin_pct": _FUEL_RESERVE_PCT}},
        command_signal={"reporting_interval_hours": 24.0},
    )
    sim = MissionSim.from_mission_order(order)
    assert sim.total_route_distance_nm == pytest.approx(_ROUTE_NM, rel=0.01)
    assert sim.state.fuel_tonnes == pytest.approx(_FUEL_TONNES)
    assert len(sim.scheduler) > 0  # routine reports were scheduled


# --- Backward compatibility: existing missions/Simulation are untouched ----------------

def test_legacy_mission_and_simulation_still_work_unaffected_by_mission_sim_module():
    mission = load_mission("UM01_rescaled")
    sim = Simulation(mission)
    sim.step(10.0)
    assert sim.t > 0.0  # ordinary encounter-sim stepping still works exactly as before
