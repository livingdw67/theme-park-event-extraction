"""The two learned taggers for event_ner: zero-shot GLiNER and a fine-tuned DistilBERT.

Both return spans in event_ner's form, (start, end, label) character offsets. transformers, torch and
gliner are imported inside the functions so the rest of parks_lib doesn't need them.
"""

import random
import re

import numpy as np

from park_ner.event_ner import LABELS, resolve_overlaps

BIO = ["O"] + [f"{p}-{label}" for label in LABELS for p in ("B", "I")]
GLINER_LABELS = {"theme park": "PARK", "ride or attraction": "ATTRACTION", "park event": "EVENT", "date": "DATE"}


class ZeroShotTagger:
    """GLiNER told only the label names; nothing from our data."""

    def __init__(self, model_name="urchade/gliner_small-v2.1", threshold=0.5):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_name)
        self.threshold = threshold

    def __call__(self, sentence):
        found = self.model.predict_entities(sentence, list(GLINER_LABELS), threshold=self.threshold)
        return resolve_overlaps([(e["start"], e["end"], GLINER_LABELS[e["label"]]) for e in found])


def _unlabeled_capitalized(text, spans):
    """Character positions in capitalized words (not the sentence's first word) that no span covers."""
    positions = set()
    for m in re.finditer(r"(?<!^)\b[A-Z][\w'’.-]*", text):
        if not any(m.start() < e and m.end() > s for s, e, _ in spans):
            positions.update(range(m.start(), m.end()))
    return positions


def _token_labels(offsets, spans, unknown=frozenset()):
    """BIO tag ids for each token, from character spans. Special tokens get -100 (ignored by the loss),
    and so do tokens starting at a position in `unknown`."""
    tags = []
    for start, end in offsets:
        if start == end or start in unknown:
            tags.append(-100)
            continue
        tag = "O"
        for s_start, s_end, label in spans:
            if start >= s_start and end <= s_end:
                tag = f"{'B' if start == s_start else 'I'}-{label}"
                break
        tags.append(BIO.index(tag))
    return tags


def _spans_from_tags(offsets, tag_ids):
    """Character spans from per-token BIO predictions; an I- without a matching open span starts one."""
    spans, current = [], None
    for (start, end), tag_id in zip(offsets, tag_ids):
        if start == end:
            continue
        tag = BIO[tag_id]
        if tag == "O":
            current = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and current and current[2] == label:
            current[1] = end
        else:
            current = [start, end, label]
            spans.append(current)
    return [tuple(s) for s in spans]


class FineTunedTagger:
    """DistilBERT (cased) trained for token classification on (sentence, spans) examples."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    @classmethod
    def train(cls, examples, base="distilbert-base-cased", epochs=4, lr=5e-5, batch_size=16, seed=42,
              ignore_unlabeled_capitalized=False, log=print):
        """Fine-tune on (sentence, spans) examples. Labels made from name lists miss names that aren't on
        the lists; with ignore_unlabeled_capitalized, capitalized words outside every span don't count as
        "not an entity" (no loss), so the model isn't taught the lists' blind spots."""
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForTokenClassification.from_pretrained(
            base, num_labels=len(BIO), id2label=dict(enumerate(BIO)), label2id={t: i for i, t in enumerate(BIO)})
        encoded = []
        for text, spans in examples:
            enc = tokenizer(text, truncation=True, max_length=128, return_offsets_mapping=True)
            unknown = _unlabeled_capitalized(text, spans) if ignore_unlabeled_capitalized else frozenset()
            enc["labels"] = _token_labels(enc.pop("offset_mapping"), spans, unknown)
            encoded.append(enc)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        steps = epochs * ((len(encoded) + batch_size - 1) // batch_size)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / steps))
        model.train()
        for epoch in range(epochs):
            random.shuffle(encoded)
            total = 0.0
            for i in range(0, len(encoded), batch_size):
                batch = encoded[i:i + batch_size]
                width = max(len(e["input_ids"]) for e in batch)
                def pad(key, value):
                    return torch.tensor([e[key] + [value] * (width - len(e[key])) for e in batch])
                loss = model(input_ids=pad("input_ids", tokenizer.pad_token_id), attention_mask=pad("attention_mask", 0),
                             labels=pad("labels", -100)).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total += loss.item() * len(batch)
            log(f"epoch {epoch + 1}/{epochs}: loss {total / len(encoded):.4f}")
        model.eval()
        return cls(model, tokenizer)

    def __call__(self, sentence):
        import torch

        enc = self.tokenizer(sentence, truncation=True, max_length=128, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            tag_ids = self.model(**enc).logits[0].argmax(-1).tolist()
        return _spans_from_tags(offsets, tag_ids)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        return cls(AutoModelForTokenClassification.from_pretrained(path).eval(), AutoTokenizer.from_pretrained(path))
