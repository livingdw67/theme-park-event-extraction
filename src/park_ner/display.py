"""HTML for showing tagged sentences."""

import html

COLORS = {"PARK": "#5fc4c0", "ATTRACTION": "#f0a742", "EVENT": "#a99bd6", "DATE": "#7fa6d1"}


def highlight(text, spans, size="1.05rem"):
    """The sentence with each span underlined in its label's colour and the label in superscript."""
    out, pos = [], 0
    for start, end, label in sorted(spans):
        color = COLORS.get(label, "#6f86a3")
        out.append(html.escape(text[pos:start]))
        out.append(f"<span style='background:{color}2e;border-bottom:2px solid {color};border-radius:3px;"
                   f"padding:1px 3px'>{html.escape(text[start:end])}"
                   f"<sup style='color:{color};font-size:0.62em;font-weight:600;margin-left:3px'>{label}</sup></span>")
        pos = end
    out.append(html.escape(text[pos:]))
    return f"<div style='font-size:{size};line-height:2.0'>" + "".join(out) + "</div>"


def legend():
    return " &nbsp; ".join(f"<span style='border-bottom:2px solid {c};padding:0 3px'>{label}</span>"
                           for label, c in COLORS.items())
