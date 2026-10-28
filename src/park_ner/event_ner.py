"""Event extraction: find parks, attractions, events and dates in text, then turn them into proposed
rows for seeds/events.csv.

Three ways to tag the same four entity types, compared on a hand-checked test set:

- rules: name lists from our own data (queue-times ride names, the park list, known events) plus a
  date pattern; also the source of the "silver" training labels
- zero-shot: a pretrained GLiNER model, told only the label names
- fine-tuned: DistilBERT trained on the silver labels (token classification, BIO tags)

Spans are (start, end, label) character offsets into the sentence, end exclusive.
"""

import re
from datetime import date

LABELS = ["EVENT", "ATTRACTION", "PARK", "DATE"]

PARKS = [
    "Universal Studios Florida", "Universal Islands of Adventure", "Islands of Adventure", "Universal Epic Universe",
    "Epic Universe", "Universal Volcano Bay", "Volcano Bay", "Universal Studios Hollywood", "Universal Studios Japan",
    "Universal Studios Singapore", "Universal Studios Beijing", "Universal Orlando Resort", "Universal Orlando",
    "Walt Disney World", "Disneyland", "Magic Kingdom", "Epcot", "Disney's Hollywood Studios", "SeaWorld Orlando",
    "Busch Gardens Tampa", "Disney California Adventure", "Wet 'n Wild Orlando",
]
EVENTS = [
    r"Halloween Horror Nights(?: (?:\d{1,2}|[IVXL]+)\b)?", r"\bHHN(?: \d{1,2}\b)?", "Mardi Gras", "Rock the Universe", "Grinchmas",
    "Universal's Holidays", "Macy's Holiday Parade", "A Celebration of Harry Potter", "Fan Fest Nights",
]

_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
          r"Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sept?\.|Oct\.|Nov\.|Dec\.)")
_YEAR = r"(?:19\d\d|20[0-4]\d)"
_MOD = r"(?:(?:early|mid|late)[- ])?"
_DATE = re.compile("|".join([
    rf"{_MONTH} \d{{1,2}}(?:\s*[–-]\s*\d{{1,2}})?,? {_YEAR}",         # September 1, 2023 / September 1–3, 2023
    rf"\d{{1,2}} {_MONTH} {_YEAR}",                                    # 1 September 2023
    rf"{_MONTH} \d{{1,2}}\b",                                          # September 1
    rf"{_MOD}{_MONTH},? {_YEAR}",                                      # (late) September 2023
    rf"{_MOD}(?:spring|summer|fall|autumn|winter)(?: of)? {_YEAR}",    # (early) summer 2025
    rf"(?:early|mid|late)[- ]{_YEAR}s?",                               # late 2024
    rf"{_YEAR}s?\b",                                                   # 2023 / 1990s
]), re.IGNORECASE)

# queue-times lists these as rides, but they are lines, train runs or shows within an attraction
_NOT_ATTRACTIONS = re.compile(r"single rider|first train|last train|virtual line", re.IGNORECASE)


def _name_pattern(name):
    """Case-insensitive, punctuation-tolerant pattern: 'MEN IN BLACK Alien Attack' matches
    'Men in Black: Alien Attack'."""
    words = re.findall(r"[A-Za-z0-9']+", name.replace("™", "").replace("®", ""))
    return r"(?<![A-Za-z0-9])" + r"[\W_]{0,3}".join(re.escape(w) for w in words) + r"(?![A-Za-z0-9])"


def attraction_names(queue_times_rides):
    """Clean attraction names from queue-times ride lists ({park_id: [(land, ride), ...]}), skipping
    single-rider lines and Halloween Horror Nights houses (land "Halloween")."""
    names = set()
    for rides in queue_times_rides.values():
        for land, ride in rides:
            if land == "Halloween" or _NOT_ATTRACTIONS.search(ride):
                continue
            names.add(re.sub(r"\s+", " ", ride.replace("™", "").replace("®", "")).strip())
    return sorted(names)


class RuleTagger:
    """Tags with name lists and the date pattern; longest match wins where matches overlap."""

    def __init__(self, attractions, parks=PARKS, events=EVENTS):
        self.patterns = [(re.compile(e, re.IGNORECASE), "EVENT") for e in events]
        self.patterns += [(re.compile(_name_pattern(p), re.IGNORECASE), "PARK") for p in parks]
        self.patterns += [(re.compile(_name_pattern(a), re.IGNORECASE), "ATTRACTION") for a in attractions]
        self.patterns.append((_DATE, "DATE"))

    def __call__(self, sentence):
        found = [(m.start(), m.end(), label) for pattern, label in self.patterns for m in pattern.finditer(sentence)]
        return resolve_overlaps(found)


def resolve_overlaps(spans):
    """Keep the longest of any overlapping spans (earlier label order breaks ties)."""
    order = {label: i for i, label in enumerate(LABELS)}
    kept = []
    for s in sorted(spans, key=lambda s: (-(s[1] - s[0]), order.get(s[2], 9), s[0])):
        if all(s[1] <= k[0] or s[0] >= k[1] for k in kept):
            kept.append(s)
    return sorted(kept)


def split_sentences(text):
    """Sentences from Wikipedia plain text: headings dropped, split on . ! ? before a capital or digit,
    except after initials and common abbreviations (E.T., U.S., Mr.)."""
    sentences = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("=="):
            continue
        parts = re.split(r"(?<!\b[A-Z]\.)(?<!\b(?:Mr|Ms|Dr|St|Jr|Sr|vs|No)\.)(?<=[.!?])\s+(?=[A-Z0-9\"“])", paragraph)
        sentences += [p.strip() for p in parts if len(p.strip()) > 20]
    return sentences


def entity_scores(gold, predicted, labels=LABELS):
    """Exact-match precision, recall and F1 per label and overall (micro). gold and predicted are lists
    (one per sentence) of span lists."""
    def counts(label=None):
        tp = fp = fn = 0
        for g, p in zip(gold, predicted):
            g = {s for s in g if label is None or s[2] == label}
            p = {s for s in p if label is None or s[2] == label}
            tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "support": tp + fn}
    return {label: counts(label) for label in labels} | {"overall": counts()}


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_date(text, default_year=None):
    """A calendar date from a DATE span, or None when the span isn't a single day ('summer 2025').
    Ranges like 'September 1–3, 2023' return their first day."""
    m = re.match(r"([A-Za-z]+)\.? (\d{1,2})(?:\s*[–-]\s*\d{1,2})?,? ?(\d{4})?$", text.strip()) or None
    if m:
        month, day, year = _MONTHS.get(m.group(1)[:3].lower()), int(m.group(2)), m.group(3)
    else:
        m = re.match(r"(\d{1,2}) ([A-Za-z]+) (\d{4})$", text.strip())
        if not m:
            return None
        day, month, year = int(m.group(1)), _MONTHS.get(m.group(2)[:3].lower()), m.group(3)
    year = int(year) if year else default_year
    if not month or not year:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def event_date_proposals(sentence, spans):
    """Event runs stated in one sentence: an EVENT followed by a start and an end date, e.g.
    'Halloween Horror Nights 33 ran from September 1 to November 4, 2023'. The year of the last date
    fills in an earlier date that has none."""
    events = [s for s in spans if s[2] == "EVENT"]
    dates = [sentence[s[0]:s[1]] for s in spans if s[2] == "DATE"]
    if not events or len(dates) < 2:
        return []
    year = next((int(y) for y in reversed(re.findall(r"\b(\d{4})\b", " ".join(dates)))), None)
    parsed = [parse_date(d, year) for d in dates]
    parsed = [d for d in parsed if d]
    if len(parsed) < 2 or parsed[1] < parsed[0]:
        return []
    name = sentence[events[0][0]:events[0][1]]
    return [{"event": name, "start_date": parsed[0], "end_date": parsed[1], "evidence": sentence}]


_CHANGE = re.compile(r"\b(re-?opened|reopens|opened|opens|debuted|debuts|closed|closes|shut down)\b", re.IGNORECASE)


def change_proposals(sentence, spans):
    """Attraction or park openings and closings stated in one sentence: an open/close verb followed by a
    single-day date. The subject is an attraction between the verb and the date ('the park closed
    Kongfrontation on September 8, 2002'), else the nearest attraction or park before the verb."""
    names = [s for s in spans if s[2] in ("ATTRACTION", "PARK")]
    dates = [s for s in spans if s[2] == "DATE"]
    out = []
    for m in _CHANGE.finditer(sentence):
        when = min((d for d in dates if d[0] >= m.end()), key=lambda d: d[0], default=None)
        if not when:
            continue
        day = parse_date(sentence[when[0]:when[1]])
        if not day:
            continue
        between = [s for s in names if s[0] >= m.end() and s[1] <= when[0] and s[2] == "ATTRACTION"]
        before = [s for s in names if s[1] <= m.start()]
        subject = between[0] if between else (max(before, key=lambda s: s[1]) if before else None)
        if not subject:
            continue
        verb = m.group(1).lower()
        change = "reopen" if verb.startswith("re") else "close" if verb.startswith(("clos", "shut")) else "open"
        out.append({"name": sentence[subject[0]:subject[1]], "kind": subject[2], "change": change, "date": day,
                    "evidence": sentence})
    return out
