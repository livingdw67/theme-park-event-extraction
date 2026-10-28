"""Turn the hand labels below into data/gold.jsonl with character offsets.

Labels are written as (text, label) per sentence, in the order of data/gold_candidates.jsonl. Each
text is located as the first occurrence not already covered by an earlier label, with straight and
curly apostrophes, '...' and '…', and hyphens and dashes treated as the same character. A text
starting with 're:' is a regular expression. Rows keep status "draft" until reviewed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from park_ner.corpus import DATA, read_jsonl, write_jsonl  # noqa: E402
from park_ner.labels import locate  # noqa: E402

A, P, E, D = "ATTRACTION", "PARK", "EVENT", "DATE"
LABELS = {
    0: [("Halloween Horror Nights", E)],
    1: [("late February 2011", D)],
    4: [("March 14, 2011", D)],
    5: [("Harry Potter and the Forbidden Journey", A), ("Islands of Adventure", P)],
    9: [("HHN XVII", E)],
    12: [("June 20", D)],
    15: [("June 24, 2024", D)],
    16: [("2021", D)],
    17: [("March 2009", D)],
    19: [("Cartoon World", P), ("Islands of Adventure", P), ("Popeye's Adventure", A), ("Bluto's Bilge-Rat Barges", A)],
    21: [("2003", D)],
    22: [("A Day in the Park with Barney", A)],
    25: [("Earthquake: The Big One", A), ("Studio Tour", A), ("Universal Studios Hollywood", P)],
    30: [("November 30, 2020", D)],
    33: [("September 9", D), ("September 8", D)],
    34: [("Universal Studios Japan", P), ("2001", D), ("Jurassic Park: The Ride", A)],
    37: [("The Amazing Adventures of Spider-Man", A), ("Shrek 4-D", A), ("Revenge of the Mummy", A)],
    40: [("Fast & Furious: Supercharged", A), ("April 23, 2018", D), ("Disaster!", A),
         ("Beetlejuice's Rock and Roll Graveyard Revue", A)],
    41: [("HHN XXII", E)],
    44: [("June 19, 2006", D)],
    45: [("Stardust Racers", A), ("May 2024", D)],
    46: [("Toy Story Mania", A)],
    47: [("August 1, 2019", D), ("Universal Orlando", P), ("Epic Universe", P)],
    48: [("Fright Nights", E), ("Universal Studios Florida", P)],
    51: [("February 12, 2021", D)],
    57: [("HHN XVIII", E), ("HHN IX", E)],
    59: [("Islands of Adventure", P)],
    64: [("1998", D)],
    65: [("March 3, 2010", D)],
    67: [("Popeye's Adventure", A), ("Bluto's Bilge-Rat Barges", A)],
    68: [("Country Bear Jamboree", A), ("Magic Kingdom", P), ("Tokyo Disneyland", P), ("It's a Small World", A)],
    70: [("2023", D)],
    71: [("spring of 1997", D)],
    76: [("May 1998", D), ("Woody Woodpecker's Nuthouse Coaster", A), ("Curious George Goes to Town", A)],
    77: [("Stardust Racers", A), ("2025", D)],
    79: [("September 8, 2025", D), ("2025", D), ("2026", D)],
    82: [("Jurassic Park: The Ride", A)],
    83: [("Bluto's Bilge-Rat Barges", A)],
    86: [("Extreme Ghostbusters: The Great Fright Way", A), ("StreetBusters", A)],
    89: [("HHN XIX", E)],
    91: [("Islands of Adventure", P)],
    92: [("Woody Woodpecker's Nuthouse Coaster", A)],
    93: [("January 30, 2024", D), ("Universal Orlando", P), ("Starfall Racers", A), ("Epic Universe", P)],
    95: [("August 9, 2025", D), ("Monsters Unchained", A), ("Epic Universe", P)],
    96: [("Halloween Horror Nights", E)],
    97: [("2021", D)],
    99: [("1980", D)],
    101: [("Harry Potter and the Forbidden Journey", A)],
    102: [("June 7, 1990", D), ("1998", D), ("Curious George Goes to Town", A)],
    104: [("Fast & Furious: Supercharged", A), ("summer of 2015", D), ("Curse of the Mummy's Tomb", A)],
    105: [("August 2015", D), ("September 8, 2015", D), ("Fast & Furious: Supercharged", A)],
    106: [("1991", D), ("Universal Studios Florida", P)],
    108: [("1999", D)],
    111: [("late 2021", D), ("Harry Potter and the Forbidden Journey", A)],
    112: [("1994", D), ("Halloween Horror Nights", E)],
    113: [("March 5, 2012", D)],
    115: [("Islands of Adventure", P), ("Cartoon World", P)],
    122: [("May 28, 1999", D), ("Islands of Adventure", P), ("Doctor Doom's Fearfall", A)],
    123: [("1993", D), ("Islands of Adventure", P)],
    124: [("Woody Woodpecker's Nuthouse Coaster", A), ("Universal Studios Florida", P)],
    125: [("Stardust Racers", A), ("Universal Epic Universe", P)],
    129: [("Rock the Universe", E), ("Universal Studios Florida", P)],
    130: [("Dinosaur", A), ("Disney's Animal Kingdom", P)],
    134: [("Terminator 2: 3-D Battle Across Time", A), ("re:1996.2017", D), ("The Bourne Stuntacular", A),
          ("Horror Make-Up Show", A)],
    135: [("October 27, 2015", D), ("re:Twister(?:\\.\\.\\.|…)Ride It Out", A), ("November 2, 2015", D)],
    137: [("re:A Major Motion Picture Ride(?:\\.\\.\\.|…)Starring You!", A)],
    139: [("October 1999", D), ("February 2000", D)],
    140: [("Universal Orlando Resort", P)],
    144: [("2004", D), ("Islands of Adventure", P), ("2005", D)],
    145: [("2017", D), ("September 9", D)],
    147: [("May 28, 1999", D)],
}


def main():
    existing = DATA / "gold.jsonl"
    if existing.exists() and "--force" not in sys.argv and any(r["status"] != "draft" for r in read_jsonl(existing)):
        sys.exit("data/gold.jsonl has reviewed rows; rerun with --force to replace them with the draft")
    rows = []
    for i, c in enumerate(read_jsonl(DATA / "gold_candidates.jsonl")):
        rows.append({"id": c["id"], "article": c["article"], "text": c["text"],
                     "spans": locate(c["text"], LABELS.get(i, [])), "status": "draft"})
    write_jsonl(DATA / "gold.jsonl", rows)
    n = sum(len(r["spans"]) for r in rows)
    print(f"{len(rows)} sentences, {n} entities, {sum(1 for r in rows if not r['spans'])} with none")
    for label in (A, P, E, D):
        print(f"  {label}: {sum(s[2] == label for r in rows for s in r['spans'])}")


if __name__ == "__main__":
    main()
