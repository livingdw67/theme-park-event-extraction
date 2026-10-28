import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from park_ner.display import highlight
from park_ner.labels import locate


def test_locate_skips_occurrences_already_labeled():
    s = "On September 8, 2025, it won the Amusement Today 2025 award."
    second = s.index("2025", 20)
    assert locate(s, [("September 8, 2025", "DATE"), ("2025", "DATE")]) == [(3, 20, "DATE"), (second, second + 4, "DATE")]


def test_locate_treats_apostrophes_ellipses_and_dashes_alike():
    s = "Twister…Ride It Out replaced Bluto’s Bilge-Rat Barges (1996–2017)."
    spans = locate(s, [("Twister...Ride It Out", "ATTRACTION"), ("Bluto's Bilge-Rat Barges", "ATTRACTION"),
                       ("1996-2017", "DATE")])
    assert [s[a:b] for a, b, _ in spans] == ["Twister…Ride It Out", "Bluto’s Bilge-Rat Barges", "1996–2017"]


def test_locate_is_case_sensitive_and_reports_missing_text():
    s = "It's a Small World, and it's a small world after all."
    assert locate(s, [("It's a Small World", "ATTRACTION")]) == [(0, 18, "ATTRACTION")]
    with pytest.raises(ValueError):
        locate(s, [("Kongfrontation", "ATTRACTION")])


def test_highlight_escapes_text_and_labels_each_span():
    out = highlight("<b> Epcot", [(4, 9, "PARK")])
    assert "&lt;b&gt;" in out and "Epcot" in out and ">PARK</sup>" in out
