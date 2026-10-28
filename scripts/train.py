"""Fine-tune DistilBERT on silver labels: the rule tagger's spans on every train-split sentence.

    python scripts/train.py --variant plain --seed 42
    python scripts/train.py --variant ignore_caps --seed 42

"ignore_caps" leaves capitalized words that no rule matched out of the loss, so names missing from the
lists aren't taught as "not an entity". Saves to models/<variant>-seed<seed>/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from park_ner.corpus import rule_tagger, sentences  # noqa: E402
from park_ner.ner_models import FineTunedTagger  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["plain", "ignore_caps"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    rules = rule_tagger()
    examples = [(s["text"], rules(s["text"])) for s in sentences("train")]
    print(f"{len(examples)} train sentences, {sum(len(spans) for _, spans in examples)} silver entities")
    started = time.time()
    tagger = FineTunedTagger.train(examples, epochs=args.epochs, seed=args.seed,
                                   ignore_unlabeled_capitalized=args.variant == "ignore_caps")
    out = ROOT / "models" / f"{args.variant}-seed{args.seed}"
    tagger.save(out)
    (out / "training.json").write_text(json.dumps({
        "variant": args.variant, "seed": args.seed, "epochs": args.epochs, "train_sentences": len(examples),
        "minutes": round((time.time() - started) / 60, 1)}, indent=1), encoding="utf-8")
    print(f"saved {out} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
