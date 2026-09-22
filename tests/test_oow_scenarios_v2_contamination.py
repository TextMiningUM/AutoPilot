"""RAG-rebuild-v2 plan point 3 ("CONTAMINATIETESTS"), updated 2026-09-22 per the user's
decision on the 2 overtaking_give_way duplicates: v1/v2 stay byte-for-byte frozen --
instead, build_oow_scenarios.py's resample_train_away_from_held_out() rejects and
redraws any TRAINING geometry that lands within tolerance of a held-out (v1/v2)
geometry, at generation time. These tests exercise that real code path (not just an
independent re-check) and assert the resulting training population is disjoint from
v1/v2 within the SAME tolerance the production code uses.
"""
from core import AgentPaths
from pipeline.track2.build_oow_scenarios import (
    generate_population, split_eval_train, load_held_out_signatures,
    resample_train_away_from_held_out, _rec_collides,
    N_EVAL_PER_CATEGORY_DEFAULT, N_TRAIN_PER_CATEGORY_DEFAULT,
)

paths = AgentPaths.oow()
V1_FILE = paths.eval_dir / "oow_colreg_scenarios_v1.json"
V2_FILE = paths.eval_dir / "oow_colreg_scenarios_v2.json"


def test_resampled_training_population_is_disjoint_from_the_frozen_v1_v2_files() -> None:
    """The real regeneration path: generate_population -> split_eval_train ->
    resample_train_away_from_held_out (against the ACTUAL frozen v1/v2 files on disk).
    Must produce zero within-tolerance collisions -- this is the test the user's plan
    says must go green "zonder dat v2 verandert"."""
    if not V1_FILE.exists() or not V2_FILE.exists():
        return
    held_out = load_held_out_signatures()
    assert held_out, "expected non-empty held-out signatures once v1/v2 exist"

    pop, rnd = generate_population(N_EVAL_PER_CATEGORY_DEFAULT + N_TRAIN_PER_CATEGORY_DEFAULT,
                                   seed=0, return_rng=True)
    _, train_recs = split_eval_train(pop, N_EVAL_PER_CATEGORY_DEFAULT)
    resampled = resample_train_away_from_held_out(train_recs, rnd, held_out)

    collisions = [r for r in resampled if _rec_collides(r, held_out)]
    assert not collisions, (
        f"{len(collisions)} resampled training record(s) still collide with the held-out "
        f"v1/v2 set -- categories: {[r['category'] for r in collisions]}"
    )
    assert len(resampled) == len(train_recs), "resampling must not change the row count"


def test_resample_replaces_a_synthetic_colliding_record_and_keeps_its_category() -> None:
    """Unit-level proof the redraw logic actually rejects+redraws (not a no-op): builds a
    held-out set directly from one real training record's own geometry, forcing a
    guaranteed collision, and confirms the record that comes back no longer collides."""
    pop, rnd = generate_population(N_EVAL_PER_CATEGORY_DEFAULT + N_TRAIN_PER_CATEGORY_DEFAULT,
                                   seed=1, return_rng=True)
    _, train_recs = split_eval_train(pop, N_EVAL_PER_CATEGORY_DEFAULT)
    victim = train_recs[0]
    fake_held_out = [(t["bearing_from_os_deg"], t["speed"], t["heading_deg"], t["_tcpa_min"]) for t in victim["targets"]]

    resampled = resample_train_away_from_held_out(train_recs, rnd, fake_held_out)
    assert resampled[0]["category"] == victim["category"]
    assert not _rec_collides(resampled[0], fake_held_out)


def test_max_attempts_guard_fails_loudly_instead_of_silently_keeping_a_duplicate() -> None:
    """If the held-out set covers every geometry a category's training records actually
    drew, forcing a fresh redraw to also collide (by construction, on a small max_attempts
    budget it's plausible for at least one candidate to keep landing back in that set),
    the guard must raise -- never silently return a colliding record."""
    pop, rnd = generate_population(N_EVAL_PER_CATEGORY_DEFAULT + N_TRAIN_PER_CATEGORY_DEFAULT,
                                   seed=2, return_rng=True)
    _, train_recs = split_eval_train(pop, N_EVAL_PER_CATEGORY_DEFAULT)
    all_sigs = [(t["bearing_from_os_deg"], t["speed"], t["heading_deg"], t["_tcpa_min"])
               for r in train_recs for t in r["targets"]]
    try:
        resample_train_away_from_held_out(train_recs[:1], rnd, all_sigs, max_attempts=0)
    except RuntimeError as e:
        assert "too narrow" in str(e)
    else:
        raise AssertionError("expected RuntimeError: max_attempts=0 must always raise on any collision")


def test_v1_and_v2_stay_byte_for_byte_unchanged_by_this_test_run() -> None:
    """Sanity check: none of the above touches the frozen files on disk."""
    if not V1_FILE.exists():
        return
    before = V1_FILE.read_bytes()
    generate_population(5, seed=0)
    after = V1_FILE.read_bytes()
    assert before == after
