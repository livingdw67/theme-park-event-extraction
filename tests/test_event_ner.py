import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from park_ner.event_ner import (RuleTagger, attraction_names, entity_scores, event_date_proposals, parse_date,
                                 resolve_overlaps, split_sentences)
from park_ner.ner_models import BIO, _spans_from_tags, _token_labels

RULES = RuleTagger(["MEN IN BLACK™ Alien Attack!", "Revenge of the Mummy"])


def tagged(sentence, tagger=RULES):
    return [(sentence[a:b], label) for a, b, label in tagger(sentence)]


def test_rules_find_each_type_with_punctuation_tolerant_names():
    s = "Men in Black: Alien Attack opened at Universal Studios Florida on April 7, 2000, before Halloween Horror Nights 10."
    assert tagged(s) == [("Men in Black: Alien Attack", "ATTRACTION"), ("Universal Studios Florida", "PARK"),
                         ("April 7, 2000", "DATE"), ("Halloween Horror Nights 10", "EVENT")]


def test_date_pattern_keeps_modifiers_and_skips_counts_of_three_digits():
    assert tagged("It closed in late September 2012 and reopened in the 1990s.") == [
        ("late September 2012", "DATE"), ("1990s", "DATE")]
    assert tagged("About 900 riders per hour.") == []


def test_longest_overlapping_span_wins():
    assert resolve_overlaps([(0, 20, "ATTRACTION"), (0, 7, "ATTRACTION"), (25, 29, "DATE")]) == [
        (0, 20, "ATTRACTION"), (25, 29, "DATE")]


def test_attraction_names_skip_single_rider_lines_and_event_houses():
    rides = {65: [("New York", "Revenge of the Mummy™"), ("New York", "Revenge of the Mummy™ Single Rider"),
                  ("Halloween", "Hellraiser")]}
    assert attraction_names(rides) == ["Revenge of the Mummy"]


def test_split_sentences_drops_headings_and_keeps_initials():
    text = "== History ==\nE.T. Adventure opened in 1990 at the park. It is still open today at Universal Studios Florida."
    assert split_sentences(text) == ["E.T. Adventure opened in 1990 at the park.",
                                     "It is still open today at Universal Studios Florida."]


def test_entity_scores_count_exact_matches_only():
    gold = [[(0, 5, "DATE"), (10, 20, "PARK")]]
    pred = [[(0, 5, "DATE"), (10, 19, "PARK"), (30, 35, "EVENT")]]
    scores = entity_scores(gold, pred)
    assert scores["DATE"]["f1"] == 1.0
    assert scores["PARK"]["recall"] == 0.0
    assert scores["overall"] == {"precision": round(1 / 3, 3), "recall": 0.5, "f1": 0.4, "support": 2}


def test_parse_date_forms():
    assert parse_date("September 1, 2023") == date(2023, 9, 1)
    assert parse_date("Sept. 1", default_year=2023) == date(2023, 9, 1)
    assert parse_date("1 September 2023") == date(2023, 9, 1)
    assert parse_date("summer 2025") is None


def test_event_run_from_one_sentence():
    s = "Halloween Horror Nights 33 ran from September 1 to November 4, 2023."
    [proposal] = event_date_proposals(s, RuleTagger([])(s))
    assert (proposal["event"], proposal["start_date"], proposal["end_date"]) == (
        "Halloween Horror Nights 33", date(2023, 9, 1), date(2023, 11, 4))


def test_bio_tags_round_trip_to_spans():
    # "Epic Universe opened" split as Epic | Uni | verse | opened, plus [CLS] and [SEP]
    offsets = [(0, 0), (0, 4), (5, 8), (8, 13), (14, 20), (0, 0)]
    spans = [(0, 13, "PARK")]
    tags = _token_labels(offsets, spans)
    assert [BIO[t] if t >= 0 else None for t in tags] == [None, "B-PARK", "I-PARK", "I-PARK", "O", None]
    assert _spans_from_tags(offsets, [t if t >= 0 else 0 for t in tags]) == spans


def test_unlabeled_capitalized_words_are_left_out_of_training():
    from park_ner.ner_models import _unlabeled_capitalized

    text = "In 2002 Kongfrontation closed at Universal Studios Florida."
    spans = [(3, 7, "DATE"), (33, 58, "PARK")]
    unknown = _unlabeled_capitalized(text, spans)
    assert set(range(8, 22)) <= unknown          # Kongfrontation: not on any list, so unknown
    assert 0 not in unknown and 33 not in unknown  # the first word and labeled spans stay labeled
    offsets = [(0, 0), (0, 2), (3, 7), (8, 12), (12, 22), (23, 29), (0, 0)]
    tags = _token_labels(offsets, spans, unknown)
    assert tags[3] == -100 and tags[5] == BIO.index("O")


def test_change_proposals_find_the_attraction_between_verb_and_date():
    from park_ner.event_ner import change_proposals

    tagger = RuleTagger(["Kongfrontation", "E.T. Adventure"])
    s = "Universal Studios Florida closed Kongfrontation on September 8, 2002, and Universal Studios Hollywood closed E.T. Adventure on March 14, 2003."
    found = [(p["name"], p["change"], p["date"]) for p in change_proposals(s, tagger(s))]
    assert found == [("Kongfrontation", "close", date(2002, 9, 8)), ("E.T. Adventure", "close", date(2003, 3, 14))]
    s = "The ride closed on September 3, 2018, and reopened on July 12, 2019."
    assert change_proposals(s, tagger(s)) == []  # no named subject in the sentence
