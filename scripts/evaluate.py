"""Score every tagger on the hand-labeled test set (data/gold.jsonl).

Taggers: the rules, zero-shot GLiNER, and each fine-tuned model in models/. Reports exact-match
precision, recall and F1 per label, a 95% bootstrap interval for overall F1 (sentences resampled,
2,000 draws), the paired difference from the rules, and recall on "unlisted" names: gold PARK,
ATTRACTION and EVENT spans the rules can't find because the name isn't on their lists.

Writes results/scores.json and results/predictions.jsonl (every tagger's spans per sentence), which
the app reads.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from park_ner.corpus import gold, rule_tagger, write_jsonl  # noqa: E402
from park_ner.event_ner import LABELS, entity_scores  # noqa: E402
from park_ner.ner_models import FineTunedTagger, ZeroShotTagger  # noqa: E402

RESULTS = ROOT / "results"
DRAWS, SEED = 2000, 0


def counts(gold_spans, predicted):
    """Per-sentence true positive, false positive and false negative counts."""
    out = []
    for g, p in zip(gold_spans, predicted):
        g, p = set(g), set(p)
        out.append((len(g & p), len(p - g), len(g - p)))
    return np.array(out)


def f1(c):
    tp, fp, fn = c.sum(axis=0)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def bootstrap(per_sentence, baseline, rng):
    """95% intervals for F1 and for F1 minus the baseline's, resampling the same sentences for both."""
    n = len(per_sentence)
    draws = rng.integers(0, n, size=(DRAWS, n))
    scores = np.array([f1(per_sentence[d]) for d in draws])
    diffs = scores - np.array([f1(baseline[d]) for d in draws])
    return [round(float(x), 3) for x in np.percentile(scores, [2.5, 97.5])], \
        [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])]


def main():
    rows = gold()
    texts, gold_spans = [r["text"] for r in rows], [r["spans"] for r in rows]
    taggers = {"rules": rule_tagger(), "gliner": ZeroShotTagger()}
    for model_dir in sorted((ROOT / "models").glob("*-seed*")):
        taggers[f"distilbert-{model_dir.name}"] = FineTunedTagger.load(model_dir)

    predictions = {name: [tagger(t) for t in texts] for name, tagger in taggers.items()}
    rule_spans = predictions["rules"]
    unlisted = [[s for s in g if s[2] != "DATE" and s not in set(r)] for g, r in zip(gold_spans, rule_spans)]
    n_unlisted = sum(len(u) for u in unlisted)

    rng = np.random.default_rng(SEED)
    baseline = counts(gold_spans, rule_spans)
    status = {k: sum(r["status"] == k for r in rows) for k in ("checked", "accepted", "draft")}
    scores = {"sentences": len(rows), "review": status,
              "entities": {label: sum(s[2] == label for g in gold_spans for s in g) for label in LABELS},
              "unlisted_entities": n_unlisted, "taggers": {}}
    for name, predicted in predictions.items():
        per_sentence = counts(gold_spans, predicted)
        ci, diff_ci = bootstrap(per_sentence, baseline, rng)
        found = sum(len(set(u) & set(p)) for u, p in zip(unlisted, predicted))
        scores["taggers"][name] = {**entity_scores(gold_spans, predicted), "overall_f1_ci": ci,
                                   "f1_minus_rules_ci": diff_ci,
                                   "unlisted_recall": round(found / n_unlisted, 3) if n_unlisted else None}

    for variant in ("plain", "ignore_caps"):
        runs = [v for k, v in scores["taggers"].items() if k.startswith(f"distilbert-{variant}-")]
        if runs:
            f1s = [r["overall"]["f1"] for r in runs]
            scores[f"distilbert_{variant}_seeds"] = {"n": len(runs), "mean_f1": round(float(np.mean(f1s)), 3),
                                                     "sd_f1": round(float(np.std(f1s, ddof=1)), 3) if len(runs) > 1 else None,
                                                     "mean_unlisted_recall": round(float(np.mean([r["unlisted_recall"] for r in runs])), 3)}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "scores.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")
    write_jsonl(RESULTS / "predictions.jsonl", [
        {"id": r["id"], "article": r["article"], "text": r["text"], "gold": [list(s) for s in r["spans"]],
         **{name: [list(s) for s in predictions[name][k]] for name in predictions}} for k, r in enumerate(rows)])

    print(f"{scores['sentences']} sentences (review {status}), entities {scores['entities']}, "
          f"unlisted {n_unlisted}")
    print(f"{'tagger':34} {'P':>6} {'R':>6} {'F1':>6}  {'95% CI':>13}  {'F1-rules CI':>15}  unlisted R  "
          + "  ".join(f"{label[:4]} F1" for label in LABELS))
    for name, s in scores["taggers"].items():
        o = s["overall"]
        print(f"{name:34} {o['precision']:6.3f} {o['recall']:6.3f} {o['f1']:6.3f}  {str(s['overall_f1_ci']):>13}  "
              f"{str(s['f1_minus_rules_ci']):>15}  {s['unlisted_recall']!s:>10}  "
              + "  ".join(f"{s[label]['f1']:7.3f}" for label in LABELS))


if __name__ == "__main__":
    main()
