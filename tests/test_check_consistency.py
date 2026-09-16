"""Unit tests for pipeline/eval/check_consistency.py using synthetic section text."""
from pipeline.eval.check_consistency import (
    check_phonetic_variants, check_repeat_counts, check_number_pronunciation, scan_document,
)


def test_phonetic_variant_detected():
    text = 'The letter A is spoken as Alpha. The letter B is spoken as Bravo.'
    findings = check_phonetic_variants(text)
    assert len(findings) == 1
    assert findings[0]["canonical"] == "ALFA"
    assert findings[0]["found"] == "Alpha"


def test_phonetic_correct_spelling_not_flagged():
    text = 'The letter A is spoken as Alfa. The letter J is spoken as Juliett.'
    assert check_phonetic_variants(text) == []


def test_repeat_count_wrong_flagged():
    text = 'Transmit: "MAYDAY MAYDAY, THIS IS VESSEL X, OVER."'
    findings = check_repeat_counts(text)
    assert len(findings) == 1
    assert findings[0]["proword"] == "MAYDAY"
    assert findings[0]["found"] == 2
    assert findings[0]["expected"] == 3


def test_repeat_count_correct_not_flagged():
    text = 'Transmit: "MAYDAY MAYDAY MAYDAY, THIS IS VESSEL X, OVER."'
    assert check_repeat_counts(text) == []


def test_narrative_mention_not_flagged():
    text = 'A MAYDAY call takes priority over all other traffic on the channel.'
    assert check_repeat_counts(text) == []


def test_number_pronunciation_wrong_flagged_in_digit_readout():
    text = 'Example: "POSITION FIVE ZERO DECIMAL ONE TWO NORTH, OVER."'
    findings = check_number_pronunciation(text)
    found = {f["found"] for f in findings}
    assert found == {"FIVE", "ONE", "TWO"}
    assert all(f["canonical"] in {"FIFE", "WUN", "TOO"} for f in findings)


def test_number_pronunciation_correct_not_flagged():
    text = 'Example: "POSITION FIFE ZERO DECIMAL WUN TOO NORTH, OVER."'
    assert check_number_pronunciation(text) == []


def test_number_pronunciation_ordinary_prose_not_flagged():
    text = 'This is the fifth example in this guide, covering four different scenarios.'
    assert check_number_pronunciation(text) == []


def test_number_pronunciation_whole_word_channel_not_flagged():
    text = 'Example: "STANDING BY ON CHANNEL SIXTEEN, OUT."'
    assert check_number_pronunciation(text) == []


def test_scan_document_attaches_section_metadata():
    doc = {
        "source_file": "fake.txt",
        "chapters": [{
            "sections": [{
                "section_id": "s1",
                "title": "Phonetic table",
                "text": "The letter A is spoken as Alpha.",
            }],
        }],
    }
    findings = scan_document(doc)
    assert len(findings) == 1
    assert findings[0]["source_file"] == "fake.txt"
    assert findings[0]["section_id"] == "s1"


if __name__ == "__main__":
    test_phonetic_variant_detected()
    test_phonetic_correct_spelling_not_flagged()
    test_repeat_count_wrong_flagged()
    test_repeat_count_correct_not_flagged()
    test_narrative_mention_not_flagged()
    test_number_pronunciation_wrong_flagged_in_digit_readout()
    test_number_pronunciation_correct_not_flagged()
    test_number_pronunciation_ordinary_prose_not_flagged()
    test_number_pronunciation_whole_word_channel_not_flagged()
    test_scan_document_attaches_section_metadata()
    print("OK — check_consistency.py behaves as expected.")
