"""Theme Park Event Extraction: try the three taggers on any text, and see how they scored."""

import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from park_ner.corpus import read_jsonl, rule_tagger  # noqa: E402
from park_ner.display import COLORS, highlight, legend  # noqa: E402
from park_ner.event_ner import LABELS, change_proposals, event_date_proposals, split_sentences  # noqa: E402

RESULTS = ROOT / "results"
# Both fine-tuned variants run live (first seed of each), so the trade-off between them is visible.
APP_MODELS = {"plain": "plain-seed42", "ignore_caps": "ignore_caps-seed42"}
MODEL_ROOT = os.environ.get("PARK_NER_MODELS") or str(ROOT / "models")
NAMES = {"rules": "Rules", "gliner": "Zero-shot GLiNER", "plain": "DistilBERT, plain",
         "ignore_caps": "DistilBERT, ignore_caps"}
EXAMPLES = [
    "Halloween Horror Nights 33 ran from September 1 to November 4, 2023 at Universal Studios Florida.",
    "Universal Studios Florida closed Kongfrontation on September 8, 2002, and Universal Studios Hollywood "
    "closed E.T. Adventure on March 14, 2003.",
    "It opened with the park on June 7, 1990, and was later closed and demolished in 1998, being replaced in "
    "the same year by the now-defunct water play area Curious George Goes to Town.",
    "Fast & Furious: Supercharged opened on April 23, 2018, replacing Disaster! and Beetlejuice's Rock and "
    "Roll Graveyard Revue.",
    "On May 28, 1999, Islands of Adventure officially opened to the public, with Doctor Doom's Fearfall being "
    "one of its debut attractions.",
]


@st.cache_resource(show_spinner="Loading the rule tagger…")
def load_rules():
    return rule_tagger()


@st.cache_resource(show_spinner="Loading GLiNER (first use takes a minute)…")
def load_gliner():
    from park_ner.ner_models import ZeroShotTagger
    return ZeroShotTagger()


@st.cache_resource(show_spinner="Loading a fine-tuned DistilBERT…")
def load_finetuned(variant):
    from park_ner.ner_models import FineTunedTagger
    return FineTunedTagger.load(f"{MODEL_ROOT}/{APP_MODELS[variant]}")


@st.cache_data
def load_results():
    scores = json.loads((RESULTS / "scores.json").read_text(encoding="utf-8"))
    return scores, read_jsonl(RESULTS / "predictions.jsonl")


st.set_page_config(page_title="Park Event Extraction", page_icon="🎢", layout="wide")
st.title("Theme Park Event Extraction")
st.markdown("Three ways to find **parks, attractions, events and dates** in text, and turn them into rows for "
            "a park events calendar: hand-written rules, a zero-shot GLiNER model, and DistilBERT fine-tuned on "
            "labels the rules produced. All three are scored on 150 hand-labeled sentences from Wikipedia "
            "articles none of them were built from.")
try_tab, results_tab, how_tab = st.tabs(["Try it", "Results", "How it works"])

with try_tab:
    example = st.selectbox("Start from an example, or write your own below", ["(write my own)"] + EXAMPLES)
    text = st.text_area("Text", "" if example == "(write my own)" else example, height=110,
                        placeholder="Paste a sentence or paragraph about a theme park…")
    chosen = st.multiselect("Taggers", list(NAMES), default=list(NAMES), format_func=NAMES.get)
    if text.strip():
        sentences = split_sentences(text) or [text.strip()]
        loaders = {"rules": load_rules, "gliner": load_gliner,
                   "plain": lambda: load_finetuned("plain"), "ignore_caps": lambda: load_finetuned("ignore_caps")}
        st.markdown(legend(), unsafe_allow_html=True)
        tagged = {}
        for name in chosen:
            tagger = loaders[name]()
            tagged[name] = [tagger(s) for s in sentences]
            st.markdown(f"**{NAMES[name]}**")
            for s, spans in zip(sentences, tagged[name]):
                st.markdown(highlight(s, spans), unsafe_allow_html=True)

        if tagged:
            st.divider()
            source = st.radio("Calendar rows from", list(tagged), format_func=NAMES.get, horizontal=True,
                              index=0)
            runs, changes = [], []
            for s, spans in zip(sentences, tagged[source]):
                runs += event_date_proposals(s, spans)
                changes += change_proposals(s, spans)
            st.markdown("**Proposed calendar rows**: event runs (an event with a start and end date) and "
                        "openings or closings (a name, an open/close verb and a single-day date) found in one sentence.")
            if runs or changes:
                rows = [{"what": r["event"], "change": "event run", "start": r["start_date"], "end": r["end_date"],
                         "evidence": r["evidence"]} for r in runs]
                rows += [{"what": c["name"], "change": c["change"], "start": c["date"], "end": None,
                          "evidence": c["evidence"]} for c in changes]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.caption("No event runs or openings/closings with a full date in this text.")

with results_tab:
    if not (RESULTS / "scores.json").exists():
        st.info("Results appear here after `python scripts/evaluate.py`.")
    else:
        scores, predictions = load_results()
        taggers = scores["taggers"]
        finetuned = [k for k in taggers if k.startswith("distilbert-")]
        st.markdown(f"Exact-match scores on **{scores['sentences']} hand-labeled sentences** "
                    f"({', '.join(f'{v} {k.lower()}' for k, v in scores['entities'].items())}). "
                    "A prediction counts only if its start, end and label all match the hand label.")
        if scores["review"]["draft"]:
            st.warning(f"Provisional: {scores['review']['draft']} of {scores['sentences']} test labels not yet reviewed.")

        table = pd.DataFrame([{
            "tagger": k.replace("distilbert-", "DistilBERT ").replace("-seed", ", seed ")
                       .replace("rules", "Rules").replace("gliner", "Zero-shot GLiNER"),
            "precision": v["overall"]["precision"], "recall": v["overall"]["recall"], "F1": v["overall"]["f1"],
            "F1 95% CI": f"{v['overall_f1_ci'][0]:.2f}–{v['overall_f1_ci'][1]:.2f}",
            "F1 vs. rules, 95% CI": "" if k == "rules" else f"{v['f1_minus_rules_ci'][0]:+.2f} to {v['f1_minus_rules_ci'][1]:+.2f}",
            "recall on unlisted names": v["unlisted_recall"],
            **{f"{label.title()} F1": v[label]["f1"] for label in LABELS}} for k, v in taggers.items()])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"Unlisted names: the {scores['unlisted_entities']} hand-labeled parks, attractions and events "
                   "the rules miss because the name isn't on their lists. That's the case the learned models are for.")

        by_label = pd.DataFrame([{"tagger": k, "label": label, "F1": v[label]["f1"]}
                                 for k, v in taggers.items()
                                 if k in ("rules", "gliner", *(f"distilbert-{m}" for m in APP_MODELS.values()))
                                 for label in LABELS])
        by_label["tagger"] = by_label["tagger"].map({"rules": "Rules", "gliner": "Zero-shot GLiNER",
                                                     **{f"distilbert-{m}": NAMES[v] for v, m in APP_MODELS.items()}})
        chart = alt.Chart(by_label).mark_bar().encode(
            x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("tagger:N", title=None),
            color=alt.Color("tagger:N", legend=None, scale=alt.Scale(range=["#6f86a3", "#f0a742", "#5fc4c0", "#a99bd6"])),
            row=alt.Row("label:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        ).properties(height=70, width=520)
        st.altair_chart(chart)

        st.subheader("Where they go wrong")
        pick = st.selectbox("Tagger", ["rules", "gliner"] + finetuned,
                            format_func=lambda k: NAMES.get(k, k.replace("distilbert-", "DistilBERT ")))
        only_errors = st.toggle("Only sentences it got wrong", value=True)
        shown = 0
        for row in predictions:
            gold_spans, pred_spans = {tuple(s) for s in row["gold"]}, {tuple(s) for s in row[pick]}
            if only_errors and gold_spans == pred_spans:
                continue
            shown += 1
            with st.container(border=True):
                st.caption(row["article"])
                st.markdown("Hand label " + highlight(row["text"], gold_spans, size="0.95rem"), unsafe_allow_html=True)
                st.markdown("Prediction " + highlight(row["text"], pred_spans, size="0.95rem"), unsafe_allow_html=True)
            if shown >= 40:
                st.caption("First 40 shown.")
                break

with how_tab:
    st.markdown("""
**The task.** Find four kinds of names in a sentence: **PARK**, **ATTRACTION**, **EVENT** and **DATE**. Then turn
sentences like *"Halloween Horror Nights 33 ran from September 1 to November 4, 2023"* or *"the park closed
Kongfrontation on September 8, 2002"* into proposed rows for a park events calendar, each with the sentence
it came from as evidence.

**Three taggers.**
1. **Rules**: ride names from the Queue-Times ride lists for the four Universal Orlando parks, a list of
   parks and events, and a date pattern. Precise, but blind to any name that isn't on a list: closed rides,
   other resorts' attractions, renamed events.
2. **Zero-shot GLiNER** (`urchade/gliner_small-v2.1`): told only the label names, nothing from this data.
3. **Fine-tuned DistilBERT** (`distilbert-base-cased`, token classification): trained on the rules' labels
   for 2,999 sentences from 29 Wikipedia articles. Those labels are *silver*: they miss every unlisted name,
   so the `ignore_caps` variant leaves capitalized words the rules didn't match out of the loss instead of
   teaching the model they're "not an entity". Three seeds per variant.

**The test.** 150 sentences sampled across 16 *other* articles (so no test sentence's article was used for
training), labeled by hand under a written guide, and scored on exact spans. 95% intervals come from
resampling sentences; the comparison with the rules resamples the same sentences for both.

**Data.** Text from English Wikipedia (CC BY-SA 4.0); ride lists Powered by
[Queue-Times.com](https://queue-times.com/). Code:
[github.com/livingdw67/theme-park-event-extraction](https://github.com/livingdw67/theme-park-event-extraction).

**Limits.** One annotator. Only 10 events in the test set, so EVENT scores are rough. Wikipedia prose is
cleaner than news or social posts.
""")
