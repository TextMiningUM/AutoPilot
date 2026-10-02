"""Tests for core/text_segmentation.py -- pure algorithmic logic, synthetic
embeddings only (no model/GPU/network needed, per project test conventions)."""
from __future__ import annotations

import numpy as np

from core.text_segmentation import (
    cosine_similarities,
    depth_scores,
    join_hyphenated_linebreaks,
    percentile,
    semantic_split_sentence_indices,
    split_sentences,
)


# ── join_hyphenated_linebreaks ──────────────────────────────────────────────
def test_joins_a_word_broken_across_a_linebreak():
    assert join_hyphenated_linebreaks("This is a contin-\nued sentence.") == \
        "This is a continued sentence."


def test_does_not_touch_a_real_compound_hyphen_on_one_line():
    assert join_hyphenated_linebreaks("a well-known fact") == "a well-known fact"


def test_does_not_touch_a_hyphen_before_a_capitalized_new_line():
    # far more likely a real dash/em-dash followed by a new sentence/heading than a
    # broken word -- e.g. "...see Annex I-\nAnnex II covers..." should stay as-is.
    text = "See Annex I-\nAnnex II covers this."
    assert join_hyphenated_linebreaks(text) == text


def test_does_not_touch_a_hyphen_at_a_paragraph_break():
    text = "end of paragraph-\n\nNew paragraph starts lowercase."
    # no lowercase letter immediately after the single \n (there's a blank line), so
    # the regex (which requires \n directly followed by a lowercase letter) must not fire
    assert join_hyphenated_linebreaks(text) == text


# ── split_sentences ──────────────────────────────────────────────────────────
def test_splits_on_sentence_boundaries():
    text = "First sentence. Second sentence! Third one?"
    assert split_sentences(text) == ["First sentence.", "Second sentence!", "Third one?"]


def test_split_sentences_drops_empty_pieces():
    assert split_sentences("  One sentence.   ") == ["One sentence."]


# ── cosine_similarities ──────────────────────────────────────────────────────
def test_cosine_similarities_identical_vectors_is_one():
    embs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert cosine_similarities(embs) == [1.0, 1.0]


def test_cosine_similarities_orthogonal_vectors_is_zero():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert cosine_similarities(embs) == [0.0]


def test_cosine_similarities_needs_at_least_two_rows():
    assert cosine_similarities(np.array([[1.0, 0.0]])) == []
    assert cosine_similarities(np.zeros((0, 2))) == []


# ── depth_scores ──────────────────────────────────────────────────────────
def test_depth_scores_flat_similarity_is_all_zero():
    assert depth_scores([0.9, 0.9, 0.9, 0.9]) == [0.0, 0.0, 0.0, 0.0]


def test_depth_scores_flags_a_deep_valley_higher_than_a_shallow_dip():
    # a deep valley (index 2) surrounded by high similarity on both sides should score
    # much higher than a shallow dip (index 0) at the very start.
    sims = [0.80, 0.95, 0.95, 0.10, 0.95, 0.95]
    scores = depth_scores(sims)
    assert scores[3] > scores[0]
    assert scores[3] == max(scores)


# ── percentile ──────────────────────────────────────────────────────────
def test_percentile_of_empty_list_is_zero():
    assert percentile([], 85.0) == 0.0


def test_percentile_basic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 100.0) == 5.0


# ── semantic_split_sentence_indices ──────────────────────────────────────────
def _two_topic_embeddings() -> np.ndarray:
    """8 sentences: first 4 about 'topic A' ([1,0]-ish), last 4 about 'topic B'
    ([0,1]-ish), with a tiny bit of per-sentence noise so within-topic similarity
    isn't a perfect 1.0 (more realistic than identical vectors)."""
    rng = np.random.RandomState(0)
    a = np.array([1.0, 0.05]) + rng.normal(0, 0.01, size=(4, 2))
    b = np.array([0.05, 1.0]) + rng.normal(0, 0.01, size=(4, 2))
    embs = np.vstack([a, b])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


def test_detects_a_genuine_topic_shift_between_two_distinct_blocks():
    embs = _two_topic_embeddings()
    splits = semantic_split_sentence_indices(embs, percentile_cutoff=85.0, min_sentences=6)
    assert splits == [4]


def test_no_split_when_all_sentences_are_on_the_same_topic():
    rng = np.random.RandomState(1)
    embs = np.array([1.0, 0.05]) + rng.normal(0, 0.01, size=(8, 2))
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    assert semantic_split_sentence_indices(embs, percentile_cutoff=85.0, min_sentences=6) == []


def test_no_split_when_too_few_sentences():
    embs = _two_topic_embeddings()[:4]  # only 4, below default min_sentences=6
    assert semantic_split_sentence_indices(embs, min_sentences=6) == []
