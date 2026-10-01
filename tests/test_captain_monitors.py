"""Captain walking-skeleton Phase 8 tests: the Sec 13.B.7 trigger monitors + the Sec
13.B.9 event-to-monitor mapping and urgency-class deadlines. Pure pipeline/ module -- no
app/ dependency, no GPU/API key needed."""
import pytest

from pipeline.captain_monitors import (
    EVENT_MONITOR_MAP, alarm_unresolved, company_instruction_received, distress_signal_received,
    engaged_within_deadline, machinery_fault_reported, position_uncertainty_above,
    track_deviation_above, urgency_deadline_s, visibility_below,
)

# --- Continuous monitors ------------------------------------------------------------------


def test_visibility_below_true_when_under_threshold():
    assert visibility_below(current_visibility_m=400.0, threshold_m=1000.0)


def test_visibility_below_false_when_at_or_above_threshold():
    assert not visibility_below(current_visibility_m=1000.0, threshold_m=1000.0)
    assert not visibility_below(current_visibility_m=2000.0, threshold_m=1000.0)


def test_position_uncertainty_above_true_when_over_threshold():
    assert position_uncertainty_above(current_uncertainty_m=600.0, threshold_m=500.0)


def test_position_uncertainty_above_false_when_at_or_below_threshold():
    assert not position_uncertainty_above(current_uncertainty_m=500.0, threshold_m=500.0)


def test_track_deviation_above_requires_both_distance_and_duration():
    # Over the distance threshold but not yet sustained long enough -- not tripped.
    assert not track_deviation_above(deviation_nm=25.0, elapsed_since_deviation_s=1000.0,
                                     threshold_nm=20.0, within_s=21600.0)
    # Sustained long enough but under the distance threshold -- not tripped.
    assert not track_deviation_above(deviation_nm=10.0, elapsed_since_deviation_s=30000.0,
                                     threshold_nm=20.0, within_s=21600.0)
    # Both conditions hold -- tripped.
    assert track_deviation_above(deviation_nm=25.0, elapsed_since_deviation_s=30000.0,
                                 threshold_nm=20.0, within_s=21600.0)


def test_track_deviation_above_uses_sensible_defaults():
    assert track_deviation_above(deviation_nm=25.0, elapsed_since_deviation_s=21600.0)


def test_alarm_unresolved_false_when_no_alarm_is_active():
    assert not alarm_unresolved(alarm_active_since_s=None, now_s=1000.0, timeout_s=600.0)


def test_alarm_unresolved_false_before_the_timeout_elapses():
    assert not alarm_unresolved(alarm_active_since_s=1000.0, now_s=1400.0, timeout_s=600.0)


def test_alarm_unresolved_true_once_the_timeout_elapses():
    assert alarm_unresolved(alarm_active_since_s=1000.0, now_s=1600.0, timeout_s=600.0)


# --- Discrete monitors ---------------------------------------------------------------------


def test_discrete_monitors_mirror_their_own_active_flag():
    assert machinery_fault_reported(True)
    assert not machinery_fault_reported(False)
    assert distress_signal_received(True)
    assert not distress_signal_received(False)
    assert company_instruction_received(True)
    assert not company_instruction_received(False)


# --- Event -> monitor mapping (Sec 13.B.9) -------------------------------------------------


def test_event_monitor_map_covers_all_5_v1_events():
    assert set(EVENT_MONITOR_MAP) == {"engine_failure", "fog", "distress_call", "whale_zone",
                                      "commercial_instruction"}


def test_whale_zone_is_excluded_from_the_monitor_mapping():
    mapping = EVENT_MONITOR_MAP["whale_zone"]
    assert mapping.monitor_name is None
    assert mapping.urgency is None


def test_the_other_4_v1_events_are_all_mapped_to_mandatory():
    for event_type in ("engine_failure", "fog", "distress_call", "commercial_instruction"):
        mapping = EVENT_MONITOR_MAP[event_type]
        assert mapping.monitor_name is not None
        assert mapping.urgency == "mandatory"


# --- Urgency-class deadlines (Sec 13.B.7) --------------------------------------------------


def test_emergency_deadline_is_the_same_instant():
    assert urgency_deadline_s("emergency", trigger_fired_at_s=1000.0, dt_mission_s=600.0) == 1000.0


def test_mandatory_deadline_is_one_dt_mission_s_later():
    assert urgency_deadline_s("mandatory", trigger_fired_at_s=1000.0, dt_mission_s=600.0) == 1600.0


def test_routine_deadline_is_the_next_report_cycles_own_timestamp():
    assert urgency_deadline_s("routine", trigger_fired_at_s=1000.0, dt_mission_s=600.0,
                             next_report_cycle_s=21600.0) == 21600.0


def test_routine_deadline_raises_without_a_report_cycle():
    with pytest.raises(ValueError):
        urgency_deadline_s("routine", trigger_fired_at_s=1000.0, dt_mission_s=600.0)


def test_urgency_deadline_raises_for_an_unknown_class():
    with pytest.raises(ValueError):
        urgency_deadline_s("urgent", trigger_fired_at_s=1000.0, dt_mission_s=600.0)


def test_engaged_within_deadline_true_for_an_immediate_response():
    assert engaged_within_deadline(trigger_fired_at_s=1000.0, responded_at_s=1000.0,
                                   urgency="mandatory", dt_mission_s=600.0)


def test_engaged_within_deadline_true_right_at_the_boundary():
    assert engaged_within_deadline(trigger_fired_at_s=1000.0, responded_at_s=1600.0,
                                   urgency="mandatory", dt_mission_s=600.0)


def test_engaged_within_deadline_false_when_the_response_is_late():
    assert not engaged_within_deadline(trigger_fired_at_s=1000.0, responded_at_s=1601.0,
                                       urgency="mandatory", dt_mission_s=600.0)
