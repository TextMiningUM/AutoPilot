"""Fase C1 (RAG-rebuild-v2 plan) tests for pipeline/track1/extract_chirp_reasoning.py --
the CHIRP-article half of C1 (grouping chunks into whole articles + collision-relevance
filtering), pure-logic only (no API calls)."""
from pipeline.track1.extract_chirp_reasoning import group_collision_relevant_articles, _slug


def _chunk(document_id, chapter_title, text, concepts, source_file="issue.pdf",
          source_type="chirp_newsletter"):
    return {"document_id": document_id, "chapter_title": chapter_title, "text": text,
           "concepts": concepts, "source_file": source_file, "source_type": source_type}


def test_groups_chunks_into_one_article_per_document_and_chapter_title() -> None:
    chunks = [
        _chunk("issue1", "FIRE IN DRYDOCK", "part 1 text", ["fatigue"]),
        _chunk("issue1", "FIRE IN DRYDOCK", "part 2 text", []),
        _chunk("issue1", "BREAKAWAY FROM MOORINGS", "unrelated text", ["environmental"]),
    ]
    docs = group_collision_relevant_articles(chunks)
    assert docs == []  # neither article has a collision-relevant concept


def test_keeps_only_articles_with_a_collision_relevant_concept() -> None:
    chunks = [
        _chunk("issue1", "CROSSING INCIDENT", "part 1", ["crossing"]),
        _chunk("issue1", "CROSSING INCIDENT", "part 2", []),
        _chunk("issue1", "FIRE IN DRYDOCK", "unrelated", ["fatigue"]),
    ]
    docs = group_collision_relevant_articles(chunks)
    assert len(docs) == 1
    assert docs[0]["document_id"] == "chirp_issue1_crossing_incident"
    assert docs[0]["full_text"] == "part 1\n\npart 2"
    assert docs[0]["colreg_hits"] == ["crossing"]


def test_ignores_non_chirp_newsletter_chunks() -> None:
    chunks = [_chunk("issue1", "CROSSING INCIDENT", "part 1", ["crossing"], source_type="regulation")]
    assert group_collision_relevant_articles(chunks) == []


def test_slug_is_filesystem_and_id_safe() -> None:
    assert _slug("FIRE IN DRYDOCK") == "fire_in_drydock"
    assert _slug("Rule 14/15 & COLREG") == "rule_14_15_colreg"
