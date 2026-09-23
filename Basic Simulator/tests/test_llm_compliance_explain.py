"""Regression test for compliance-rebuild STAP 4 -- the LLM explanation step
(app/evaluation.py's llm_compliance_check()) must never make an API call when there are
no findings to explain."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.evaluation import llm_compliance_check  # noqa: E402


def test_no_findings_returns_empty_list_no_api_call() -> None:
    """No ANTHROPIC_API_KEY needed here -- an empty findings list must short-circuit
    before anthropic/core.io.load_env are ever touched, let alone a real network call."""
    result = llm_compliance_check([])
    assert result == {"explanations": []}
