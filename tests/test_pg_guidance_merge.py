"""Tests for pg_guidance.load_merged_pg() -- used by Basic Simulator/app/agents.py's
v8_super_cot_pg/v9_super_all configs (pg="scenario+incident") to combine guidance from
oow_pg_scenario.json and oow_pg_incident.json, which independently use the SAME
"pg_0000", "pg_0001", ... node-id scheme and would collide on a naive concatenation."""
import json

from sentence_transformers import SentenceTransformer

from core import AgentPaths, EMBEDDER_MODEL
from pipeline.ingest.pg_guidance import load_merged_pg

_paths = AgentPaths.oow()
_CACHE = _paths.cache_dir
_SCENARIO_FILE = _CACHE / "oow_pg_scenario.json"
_INCIDENT_FILE = _CACHE / "oow_pg_incident.json"
_RULE_FILE = _CACHE / "oow_pg_rule.json"


def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDER_MODEL, device="cpu")


def test_merged_graph_has_no_node_id_collisions() -> None:
    scenario = json.loads(_SCENARIO_FILE.read_text(encoding="utf-8"))
    incident = json.loads(_INCIDENT_FILE.read_text(encoding="utf-8"))
    g = load_merged_pg([_SCENARIO_FILE, _INCIDENT_FILE], _embedder())
    # Both sources' own id schemes overlap ("pg_0000", "pg_0001", ...) -- the merge must
    # namespace them so nothing gets silently overwritten (count == sum of the two sources).
    assert len(g.node_ids) == len(scenario["nodes"]) + len(incident["nodes"])
    assert len(set(g.node_ids)) == len(g.node_ids)  # no duplicate ids


def test_merged_graph_can_retrieve_nodes_from_both_sources() -> None:
    scenario = json.loads(_SCENARIO_FILE.read_text(encoding="utf-8"))
    incident = json.loads(_INCIDENT_FILE.read_text(encoding="utf-8"))
    scenario_label = next(iter(scenario["nodes"].values()))["label"]
    incident_label = next(iter(incident["nodes"].values()))["label"]

    g = load_merged_pg([_SCENARIO_FILE, _INCIDENT_FILE], _embedder())
    # Querying each source's OWN label text must resolve back to a node namespaced under
    # that source's index (0=scenario, 1=incident) -- proving both halves are genuinely
    # reachable through the one merged graph, not just present-but-unmatchable.
    matched_id, score = g.match(scenario_label)
    assert matched_id is not None and matched_id.startswith("0:"), (matched_id, score)
    matched_id, score = g.match(incident_label)
    assert matched_id is not None and matched_id.startswith("1:"), (matched_id, score)


def test_merged_graph_excludes_rule_graph() -> None:
    scenario = json.loads(_SCENARIO_FILE.read_text(encoding="utf-8"))
    incident = json.loads(_INCIDENT_FILE.read_text(encoding="utf-8"))
    rule = json.loads(_RULE_FILE.read_text(encoding="utf-8"))
    g = load_merged_pg([_SCENARIO_FILE, _INCIDENT_FILE], _embedder())
    # oow_pg_rule.json is a separate, optional test arm -- never part of "scenario+incident".
    # Node-count equality (not just "rule labels aren't found") is the strongest available
    # proof no third source's nodes snuck into the merge.
    assert len(g.node_ids) == len(scenario["nodes"]) + len(incident["nodes"]) != len(rule["nodes"])
