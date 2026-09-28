"""Regression test (2026-09-28) for run_baseline_scenario.py's adaptive decision cadence --
baselines now use the SAME app.narrate.live_decision_interval() cadence run_llm_scenario.py
uses by default, instead of a fixed decision every simulation step, so a fair comparison
against sparse-cadence LLM runs is possible. --decision-interval remains a fixed override."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.run_baseline_scenario import run_one  # noqa: E402


def test_default_cadence_is_adaptive_and_sparser_than_every_step(tmp_path, monkeypatch):
    import app.llm_runs as llm_runs
    monkeypatch.setattr(llm_runs, "RUNS_DIR", tmp_path)
    import app.run_baseline_scenario as rbs
    monkeypatch.setattr(rbs, "RUNS_DIR", tmp_path)

    out_path = run_one("Imazu01", "baseline_ruletree", tag="_test_adaptive", force=True)
    log = __import__("json").loads(out_path.read_text(encoding="utf-8"))
    assert log["params"]["decision_interval_mode"] == "adaptive"
    checkpoints = log["checkpoints"]
    assert len(checkpoints) >= 1
    # At least one checkpoint's own adaptive interval must be > 1 step -- confirms this
    # is NOT silently still deciding every single step.
    assert any(cp["decision_interval_steps"] > 1 for cp in checkpoints)


def test_fixed_decision_interval_override_still_works(tmp_path, monkeypatch):
    import app.llm_runs as llm_runs
    monkeypatch.setattr(llm_runs, "RUNS_DIR", tmp_path)
    import app.run_baseline_scenario as rbs
    monkeypatch.setattr(rbs, "RUNS_DIR", tmp_path)

    out_path = run_one("Imazu01", "baseline_ruletree", tag="_test_fixed",
                       decision_interval=1, force=True)
    log = __import__("json").loads(out_path.read_text(encoding="utf-8"))
    assert log["params"]["decision_interval_mode"] == "fixed"
    assert log["params"]["decision_interval"] == 1
    assert all(cp["decision_interval_steps"] == 1 for cp in log["checkpoints"])
