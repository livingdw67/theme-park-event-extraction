"""Turning hand-written (text, label) pairs into character spans."""

import re


def pattern(text):
    if text.startswith("re:"):
        return re.compile(text[3:])
    parts = []
    for ch in text:
        if ch in "'’":
            parts.append("['’]")
        elif ch in "-–—":
            parts.append("[-–—]")
        else:
            parts.append(re.escape(ch))
    return re.compile("".join(parts).replace(re.escape(".") * 3, "(?:\\.\\.\\.|…)"))


def locate(sentence, labels):
    spans = []
    for text, label in labels:
        for m in pattern(text).finditer(sentence):
            if all(m.end() <= s or m.start() >= e for s, e, _ in spans):
                spans.append((m.start(), m.end(), label))
                break
        else:
            raise ValueError(f"{text!r} not found in {sentence!r}")
    return sorted(spans)
