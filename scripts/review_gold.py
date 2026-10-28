"""Review the hand-labeled test set one sentence at a time.

    streamlit run scripts/review_gold.py

Each sentence shows its labels highlighted and as editable lines ("LABEL | text"). "Correct" marks the
row checked as it is; "Save edits" re-locates the edited lines in the sentence and marks it checked.
Changes are written to data/gold.jsonl immediately.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from park_ner.corpus import DATA, read_jsonl, write_jsonl  # noqa: E402
from park_ner.display import COLORS, highlight  # noqa: E402
from park_ner.event_ner import LABELS  # noqa: E402
from park_ner.labels import locate  # noqa: E402

GOLD = DATA / "gold.jsonl"


def as_lines(text, spans):
    return "\n".join(f"{label} | {text[s:e]}" for s, e, label in sorted(spans))


def parse_lines(lines):
    pairs = []
    for line in lines.splitlines():
        if not line.strip():
            continue
        label, _, text = line.partition("|")
        label, text = label.strip().upper(), text.strip()
        if label not in LABELS or not text:
            raise ValueError(f"Can't read {line!r}: use LABEL | text, with LABEL one of {', '.join(LABELS)}")
        pairs.append((text, label))
    return pairs


st.set_page_config(page_title="Test Set Review", layout="centered")
rows = read_jsonl(GOLD)
if "i" not in st.session_state:
    st.session_state.i = next((k for k, r in enumerate(rows) if r["status"] == "draft"), 0)
i = st.session_state.i
row = rows[i]

checked = sum(r["status"] == "checked" for r in rows)
st.progress(checked / len(rows), text=f"{checked} of {len(rows)} checked")
with st.sidebar:
    st.markdown("**Labels**: " + " · ".join(f"<span style='color:{c}'>{lab}</span>" for lab, c in COLORS.items()),
                unsafe_allow_html=True)
    st.markdown("Rules in [docs/annotation_guide.md](../docs/annotation_guide.md). Tag named parks, rides and shows, "
                "park events and calendar dates; not lands, films, characters, haunted houses, durations or bare months.")
    jump = st.number_input("Go to sentence", 1, len(rows), i + 1)
    if jump - 1 != i:
        st.session_state.i = jump - 1
        st.rerun()

st.caption(f"Sentence {i + 1} of {len(rows)} · {row['article']} · {row['status']}")
st.markdown(highlight(row["text"], row["spans"], size="1.15rem"), unsafe_allow_html=True)
edited = st.text_area("Labels (one per line: LABEL | text)", as_lines(row["text"], row["spans"]), height=140,
                      key=f"edit-{i}")

left, middle, right = st.columns(3)


def save_and_next(spans):
    rows[i]["spans"], rows[i]["status"] = [list(s) for s in spans], "checked"
    write_jsonl(GOLD, rows)
    st.session_state.i = min(i + 1, len(rows) - 1)
    st.rerun()


if left.button("Correct ✓", type="primary", width="stretch"):
    save_and_next(row["spans"])
if middle.button("Save edits", width="stretch"):
    try:
        save_and_next(locate(row["text"], parse_lines(edited)))
    except ValueError as err:
        st.error(str(err))
if right.button("Skip", width="stretch"):
    st.session_state.i = min(i + 1, len(rows) - 1)
    st.rerun()
