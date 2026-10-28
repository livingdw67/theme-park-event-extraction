"""Loading the corpus files written by scripts/fetch_corpus.py and the hand-labeled test set."""

import json
from pathlib import Path

from park_ner.event_ner import RuleTagger, attraction_names

DATA = Path(__file__).resolve().parents[2] / "data"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sentences(split=None, data=DATA):
    return [s for s in read_jsonl(data / "sentences.jsonl") if split is None or s["split"] == split]


def rule_tagger(data=DATA):
    """The rule tagger with attraction names from the saved queue-times ride lists."""
    rides = json.loads((data / "queue_times_rides.json").read_text(encoding="utf-8"))
    return RuleTagger(attraction_names(rides))


def gold(data=DATA):
    """Hand-labeled test sentences: dicts with id, text and spans [(start, end, label), ...]."""
    rows = read_jsonl(data / "gold.jsonl")
    for row in rows:
        row["spans"] = [tuple(s) for s in row["spans"]]
    return rows
