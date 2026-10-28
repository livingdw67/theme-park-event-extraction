"""Download the corpus: queue-times ride lists (the rule tagger's attraction names) and Wikipedia
articles split into sentences.

Articles are split into train and test by article, not by sentence, so the test sentences come from
pages the fine-tuned model never saw. Writes data/queue_times_rides.json, data/articles.json (title,
revision id, URL, split) and data/sentences.jsonl.

Wikipedia text is CC BY-SA 4.0; ride lists are Powered by Queue-Times.com.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from park_ner.event_ner import split_sentences  # noqa: E402

DATA = ROOT / "data"
USER_AGENT = "theme-park-event-extraction/0.1 (https://github.com/livingdw67/theme-park-event-extraction)"
QUEUE_TIMES_PARKS = {65: "Universal Studios Florida", 64: "Islands of Adventure", 334: "Epic Universe", 67: "Volcano Bay"}

TRAIN = [
    "Universal Studios Florida", "Universal Islands of Adventure", "Universal Epic Universe", "Universal Volcano Bay",
    "Universal Orlando Resort", "Universal Studios Hollywood", "Universal Studios Japan",
    "Universal Studios Singapore", "Universal Studios Beijing", "The Wizarding World of Harry Potter (Universal Orlando Resort)",
    "Revenge of the Mummy", "Jurassic World VelociCoaster", "Hagrid's Magical Creatures Motorbike Adventure",
    "The Incredible Hulk Coaster", "Men in Black: Alien Attack", "Kongfrontation", "Jaws (ride)",
    "Back to the Future: The Ride", "E.T. Adventure", "Harry Potter and the Escape from Gringotts",
    "Hogwarts Express (Universal Orlando Resort)", "The Amazing Adventures of Spider-Man", "Dragon Challenge",
    "Skull Island: Reign of Kong", "Race Through New York Starring Jimmy Fallon", "Transformers: The Ride",
    "The Simpsons Ride", "Despicable Me Minion Mayhem", "Hollywood Rip Ride Rockit",
]
TEST = [
    "Halloween Horror Nights", "Rock the Universe", "Jurassic Park River Adventure", "Popeye & Bluto's Bilge-Rat Barges",
    "Dudley Do-Right's Ripsaw Falls", "Harry Potter and the Forbidden Journey", "Earthquake: The Big One",
    "Twister...Ride it Out", "Fast & Furious: Supercharged", "Disaster!", "Doctor Doom's Fearfall",
    "Storm Force Accelatron", "Woody Woodpecker's Nuthouse Coaster", "Stardust Racers", "Mario Kart: Bowser's Challenge",
    "Monsters Unchained: The Frankenstein Experiment",
]


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def wikipedia_article(title):
    """Plain text, final title and revision id of one article, following redirects; None if missing."""
    params = {"action": "query", "format": "json", "formatversion": "2", "redirects": "1", "titles": title,
              "prop": "extracts|revisions", "explaintext": "1", "rvprop": "ids"}
    page = get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params))["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return {"title": page["title"], "revid": page["revisions"][0]["revid"], "text": page["extract"]}


def main():
    DATA.mkdir(exist_ok=True)
    rides = {}
    for park_id in QUEUE_TIMES_PARKS:
        payload = get_json(f"https://queue-times.com/parks/{park_id}/queue_times.json")
        rides[park_id] = [(land["name"], r["name"]) for land in payload["lands"] for r in land["rides"]]
        rides[park_id] += [("", r["name"]) for r in payload.get("rides", [])]
    (DATA / "queue_times_rides.json").write_text(json.dumps(rides, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"queue-times: {sum(len(v) for v in rides.values())} rides")

    articles, sentences, seen = [], [], set()
    for split, titles in (("train", TRAIN), ("test", TEST)):
        for title in titles:
            article = wikipedia_article(title)
            time.sleep(0.5)
            if not article or article["title"] in seen:
                print(f"  skipped {title!r} ({'duplicate' if article else 'missing'})")
                continue
            seen.add(article["title"])
            url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(article["title"].replace(" ", "_"))
            articles.append({"title": article["title"], "revid": article["revid"], "url": url, "split": split})
            for i, text in enumerate(split_sentences(article["text"])):
                sentences.append({"id": f"{article['revid']}-{i}", "article": article["title"], "split": split, "text": text})
    (DATA / "articles.json").write_text(json.dumps(articles, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(DATA / "sentences.jsonl", "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    for split in ("train", "test"):
        n_articles = sum(a["split"] == split for a in articles)
        n_sentences = sum(s["split"] == split for s in sentences)
        print(f"{split}: {n_articles} articles, {n_sentences} sentences")


if __name__ == "__main__":
    main()
