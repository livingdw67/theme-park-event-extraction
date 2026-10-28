"""Pick the test sentences to hand-label and pre-tag them as a starting point.

Takes 150 test-split sentences round-robin across articles (seeded shuffle within each article), so
no single long article dominates. Writes data/gold_candidates.jsonl with the rule and zero-shot
GLiNER spans for each sentence; the labels themselves are written by hand into data/gold.jsonl.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from park_ner.corpus import DATA, rule_tagger, sentences, write_jsonl  # noqa: E402
from park_ner.ner_models import ZeroShotTagger  # noqa: E402

N, SEED = 150, 7


def main():
    by_article = {}
    for s in sentences("test"):
        by_article.setdefault(s["article"], []).append(s)
    rng = random.Random(SEED)
    for rows in by_article.values():
        rng.shuffle(rows)
    picked, depth = [], 0
    while len(picked) < N:
        layer = [rows[depth] for rows in by_article.values() if depth < len(rows)]
        if not layer:
            break
        picked += layer[:N - len(picked)]
        depth += 1

    rules, zero_shot = rule_tagger(), ZeroShotTagger()
    out = []
    for s in picked:
        text = s["text"]
        out.append({**s, "rules": [(text[a:b], label) for a, b, label in rules(text)],
                    "gliner": [(text[a:b], label) for a, b, label in zero_shot(text)]})
    write_jsonl(DATA / "gold_candidates.jsonl", out)
    print(f"{len(out)} sentences from {len({s['article'] for s in out})} articles")


if __name__ == "__main__":
    main()
