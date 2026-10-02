"""Tests for pipeline/ingest/build_rag.py's chunker, including the new embedding-based
topic-boundary detection (Docs/rag_chunking_design_and_verification.md).

Uses a FAKE embedding model + tokenizer (no sentence-transformers model load, no GPU/
network) -- per project test conventions, mock any model call. The fake embedder maps
a sentence to one of two orthogonal directions purely by keyword presence, so topic
shifts are deterministic and don't depend on a real model's actual semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.ingest import build_rag


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


class _FakeModel:
    """2D fake embedder: 'whale' -> [1, 0], 'fog' -> [0, 1], anything else -> [0.7, 0.7]
    (all L2-normalized), so two keyword-distinguished "topics" are deterministically
    orthogonal-ish and everything else is a neutral middle ground."""

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vecs = []
        for t in texts:
            low = t.lower()
            if "whale" in low:
                v = np.array([1.0, 0.0])
            elif "fog" in low:
                v = np.array([0.0, 1.0])
            else:
                v = np.array([0.7, 0.7])
            vecs.append(v / np.linalg.norm(v))
        return np.array(vecs)


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    monkeypatch.setattr(build_rag, "model", _FakeModel())
    monkeypatch.setattr(build_rag, "tokenizer", _FakeTokenizer())


# ── _split_section_by_topic ──────────────────────────────────────────────────
def test_splits_a_section_that_internally_drifts_topic():
    section = {
        "section_id": "s1", "title": "Mixed", "type": "reference",
        "text": (
            "The whale population here is large. Whale sightings are common near the coast. "
            "Whale watching tours operate daily. Fog often rolls in during the morning. "
            "Fog reduces visibility significantly. Fog can last for hours."
        ),
        "topics": ["marine"], "pages": [1],
    }
    pieces = build_rag._split_section_by_topic(section)
    assert len(pieces) == 2
    assert "whale" in pieces[0]["text"].lower()
    assert "fog" in pieces[1]["text"].lower()
    assert pieces[0]["semantic_split"] is True
    assert pieces[0]["section_id"] == "s1_t1"
    assert pieces[1]["section_id"] == "s1_t2"


def test_does_not_split_a_section_with_too_few_sentences():
    section = {
        "section_id": "s1", "title": "Short", "type": "reference",
        "text": "Whale sightings are common. Fog rolls in sometimes.",
        "topics": ["marine"], "pages": [1],
    }
    pieces = build_rag._split_section_by_topic(section)
    assert pieces == [section]


def test_does_not_split_a_single_topic_section():
    section = {
        "section_id": "s1", "title": "Whales", "type": "reference",
        "text": (
            "Whale sightings are common near the coast. Whale watching tours operate daily. "
            "Whales migrate through this channel every spring. Whale calls can be heard "
            "underwater. Whale populations have recovered since the ban. Whale researchers "
            "visit every summer."
        ),
        "topics": ["marine"], "pages": [1],
    }
    pieces = build_rag._split_section_by_topic(section)
    assert len(pieces) == 1


# ── chunk_chapter: embedding-based merge-blocking ───────────────────────────
def _section(sid: str, text: str, topics: list[str]) -> dict:
    return {"section_id": sid, "title": sid, "type": "reference", "text": text,
            "topics": topics, "pages": [1]}


def test_chunk_chapter_blocks_a_merge_across_a_real_topic_shift_even_with_matching_tags():
    # All 4 sections share the SAME hand-assigned topic tag (so the coarse
    # keyword-Jaccard check alone would allow every merge) -- only the embedding-based
    # boundary check should block merging section 2 into section 3.
    sections = [
        _section("s1", "Whale sightings are common near the coast today.", ["marine"]),
        _section("s2", "Whale watching tours operate daily in this bay.", ["marine"]),
        _section("s3", "Fog reduces visibility significantly in this channel.", ["marine"]),
        _section("s4", "Fog can last for hours during the autumn months.", ["marine"]),
    ]
    chapter = {"title": "Chapter", "sections": sections}
    doc = {"document_id": "doc1", "source_file": "f.txt", "source_type": "guide"}
    chunks = build_rag.chunk_chapter(chapter, doc, 0)
    section_id_groups = [c["section_ids"] for c in chunks]
    # s1+s2 (whale/whale) and s3+s4 (fog/fog) should each merge; s2+s3 must NOT.
    assert ["s1", "s2"] in section_id_groups
    assert ["s3", "s4"] in section_id_groups


def test_contains_semantic_split_flag_propagates_to_the_chunk():
    long_mixed_text = (
        "The whale population here is large. Whale sightings are common near the coast. "
        "Whale watching tours operate daily. Fog often rolls in during the morning. "
        "Fog reduces visibility significantly. Fog can last for hours."
    )
    sections = [_section("s1", long_mixed_text, ["marine"])]
    chapter = {"title": "Chapter", "sections": sections}
    doc = {"document_id": "doc1", "source_file": "f.txt", "source_type": "guide"}
    chunks = build_rag.chunk_chapter(chapter, doc, 0)
    assert any(c["contains_semantic_split"] for c in chunks)
