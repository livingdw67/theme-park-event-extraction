# Theme Park Event Extraction

Find **parks, attractions, events and dates** in text and turn them into rows for a park events
calendar, e.g. *"Universal Studios Florida closed Kongfrontation on September 8, 2002"* →
`Kongfrontation · close · 2002-09-08`, with the sentence kept as evidence.

Three taggers are compared on 150 hand-labeled sentences from Wikipedia articles that none of them
were built from:

1. **Rules**: Queue-Times ride lists for the four Universal Orlando parks, park and event lists, and a date pattern.
2. **Zero-shot GLiNER** (`urchade/gliner_small-v2.1`), told only the label names.
3. **Fine-tuned DistilBERT** (`distilbert-base-cased`, token classification) trained on the rules' labels
   ("silver" labels) for 2,999 sentences from 29 other articles, in two variants: `plain`, and
   `ignore_caps`, which leaves capitalized words the rules didn't match out of the loss.

## Results (provisional: seed 42 only; seeds 7 and 13 in progress)

Exact-match scores on 150 sentences (131 entities: 44 attractions, 26 parks, 10 events, 51 dates).
"Unlisted" are the 45 hand-labeled parks, attractions and events the rules can't find because the
name isn't on their lists.

| Tagger | Precision | Recall | F1 (95% CI) | Recall on unlisted names |
|---|---|---|---|---|
| Rules | 0.90 | 0.65 | 0.76 (0.67–0.84) | 0 |
| Zero-shot GLiNER | 0.48 | 0.73 | 0.58 (0.50–0.66) | 0.40 |
| DistilBERT, plain | 0.89 | 0.66 | 0.75 (0.67–0.83) | 0.07 |
| DistilBERT, ignore_caps | 0.34 | 0.75 | 0.47 (0.39–0.54) | 0.31 |

Trained on the rules' labels alone, DistilBERT learns to copy the rules. Taking unlisted capitalized
words out of the loss lets it find a third of the unlisted names, but it then tags characters and
titles as well, and precision collapses. GLiNER finds the most unlisted names, also at low precision.

## Layout

| Path | What |
|---|---|
| `src/park_ner/` | Taggers (`event_ner.py`, `ner_models.py`), corpus loading, span helpers, display |
| `scripts/fetch_corpus.py` | Queue-Times ride lists and Wikipedia articles, split into train and test by article |
| `scripts/sample_test_set.py`, `write_gold.py`, `review_gold.py` | Test-set sampling, hand labels, and the review app |
| `scripts/train.py`, `evaluate.py` | Fine-tuning and scoring (bootstrap CIs, paired difference from the rules) |
| `app/streamlit_app.py` | Try the taggers live; results and error browser |
| `docs/annotation_guide.md` | What counts as each label |
| `data/` | Corpus, ride lists, `gold.jsonl` (150 labeled sentences, all reviewed) |
| `results/` | `scores.json`, `predictions.jsonl` |

```
python -m venv .venv && .venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q tests
.venv\Scripts\python scripts/train.py --variant plain --seed 42
.venv\Scripts\python scripts/evaluate.py
.venv\Scripts\streamlit run app/streamlit_app.py
```

## Data

Text from English Wikipedia (CC BY-SA 4.0); ride lists Powered by [Queue-Times.com](https://queue-times.com/).
The test labels were drafted by hand under the annotation guide and reviewed sentence by sentence;
the rule and GLiNER outputs were shown alongside as a checklist, never accepted automatically.
