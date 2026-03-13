import argparse
import hashlib
import io
import logging
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from html import unescape
from urllib.request import Request, urlopen
from typing import List

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("post_process")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
IMAGE_DOMAINS = ("i.redd.it", "i.imgur.com", "imgur.com", "preview.redd.it")

TOPIC_CATEGORIES = {
    "Technology":    {"ai", "software", "hardware", "computer", "app", "tech", "code",
                      "programming", "algorithm", "data", "machine", "learning", "robot",
                      "digital", "internet", "cyber", "cloud", "api", "device", "gpu",
                      "cpu", "neural", "model", "automation", "blockchain"},
    "Science":       {"research", "study", "scientist", "experiment", "discovery",
                      "physics", "chemistry", "biology", "space", "quantum", "genome",
                      "climate", "species", "molecule", "theory", "lab", "nasa", "dna"},
    "Politics":      {"government", "president", "election", "policy", "congress",
                      "democrat", "republican", "vote", "law", "legislation", "senate",
                      "political", "campaign", "regulation", "federal", "court", "rights"},
    "Finance":       {"stock", "market", "invest", "bank", "economy", "inflation",
                      "crypto", "bitcoin", "trading", "revenue", "profit", "financial",
                      "dollar", "fund", "debt", "gdp", "tax", "price", "earnings"},
    "Health":        {"health", "medical", "disease", "treatment", "vaccine", "doctor",
                      "patient", "hospital", "drug", "mental", "cancer", "therapy",
                      "symptom", "diagnosis", "clinical", "virus", "fda", "nutrition"},
    "Entertainment": {"movie", "game", "music", "show", "film", "series", "album",
                      "artist", "trailer", "release", "review", "stream", "concert",
                      "anime", "gaming", "play", "season", "episode", "actor"},
    "Education":     {"learn", "school", "university", "student", "course", "degree",
                      "education", "teacher", "college", "academic", "study", "exam",
                      "tutor", "curriculum", "lecture", "scholarship"},
    "World News":    {"war", "country", "international", "global", "conflict", "nation",
                      "treaty", "military", "diplomatic", "sanction", "crisis", "refugee",
                      "border", "foreign", "united", "eu", "nato", "trade", "peace"},
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"/r/\w+|/u/\w+|u/\w+", "", text)
    text = re.sub(r"\[removed\]|\[deleted\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mask_username(username: str) -> str:
    if not username or username in ("[deleted]", "[removed]", "AutoModerator"):
        return username
    return f"user_{hashlib.sha256(username.encode()).hexdigest()[:8]}"


def convert_timestamp(iso_timestamp: str) -> str:
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%B %d, %Y at %I:%M %p UTC")
    except (ValueError, TypeError):
        return iso_timestamp


STOP_WORDS = {
    "the", "and", "for", "that", "with", "this", "are", "was", "were", "been",
    "but", "not", "you", "your", "all", "can", "had", "her", "his", "him",
    "how", "its", "may", "new", "now", "old", "see", "way", "who", "did",
    "get", "has", "have", "just", "more", "also", "been", "from", "they",
    "will", "what", "when", "where", "which", "their", "there", "these",
    "those", "would", "about", "could", "other", "than", "then", "them",
    "into", "some", "such", "only", "over", "very", "after", "before",
    "being", "between", "both", "each", "because", "does", "during",
    "should", "while", "here", "most", "much", "many", "well", "back",
    "like", "make", "made", "know", "think", "still", "even", "take",
    "come", "want", "say", "said", "use", "used", "first", "going",
    "people", "thing", "things", "really", "good", "great", "right",
    "look", "long", "little", "big", "keep", "let", "put", "give",
    "tell", "need", "every", "own", "through", "our", "out", "any",
    "time", "day", "too", "don", "one", "two", "off", "got", "why",
}


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    filtered = [w for w in tokens if w not in STOP_WORDS]
    return [w for w, _ in Counter(filtered).most_common(top_n)]


def classify_topics(text: str, top_n: int = 3) -> List[str]:
    if not text:
        return ["General"]
    tokens = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    scores = {topic: len(tokens & kws) for topic, kws in TOPIC_CATEGORIES.items()}
    ranked = [t for t in sorted(scores, key=scores.get, reverse=True) if scores[t] > 0]
    return ranked[:top_n] if ranked else ["General"]


def extract_ocr_text(url: str, domain: str = "", is_self: bool = True) -> str:
    if not OCR_AVAILABLE or is_self or not url:
        return ""
    url_lower = url.lower()
    is_image = any(url_lower.endswith(ext) for ext in IMAGE_EXTENSIONS) or \
               any(d in (domain or "") for d in IMAGE_DOMAINS)
    if not is_image:
        return ""
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=15) as resp:
            img = Image.open(io.BytesIO(resp.read()))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return clean_text(pytesseract.image_to_string(img, timeout=30))
    except Exception as e:
        log.debug("OCR failed for %s: %s", url[:80], e)
        return ""


def post_process(db_path: str = "reddit_posts.db", skip_ocr: bool = False):
    if not os.path.exists(db_path):
        log.error("Database not found: %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    existing = {r[1] for r in conn.execute("PRAGMA table_info(posts)").fetchall()}
    for col, dtype in {
        "cleaned_title": "TEXT", "cleaned_selftext": "TEXT",
        "masked_author": "TEXT", "readable_date": "TEXT",
        "ocr_text": "TEXT DEFAULT ''",
        "keywords": "TEXT DEFAULT ''", "topics": "TEXT DEFAULT ''",
        "is_processed": "INTEGER DEFAULT 0",
    }.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE posts ADD COLUMN {col} {dtype}")
    conn.commit()

    conn.row_factory = sqlite3.Row
    posts = conn.execute(
        "SELECT post_id, title, selftext, author, created_utc, url, domain, is_self "
        "FROM posts WHERE is_processed = 0 OR is_processed IS NULL"
    ).fetchall()

    if not posts:
        log.info("No unprocessed posts found.")
        conn.close()
        return

    log.info("Processing %d posts ...", len(posts))
    ocr_count = 0

    for i, p in enumerate(posts, 1):
        cleaned_title    = clean_text(p["title"] or "")
        cleaned_selftext = clean_text(p["selftext"] or "")
        masked           = mask_username(p["author"] or "")
        readable_date    = convert_timestamp(p["created_utc"] or "")
        ocr_text = ""
        if not skip_ocr:
            ocr_text = extract_ocr_text(p["url"] or "", p["domain"] or "", bool(p["is_self"]))
            if ocr_text:
                ocr_count += 1

        combined = f"{cleaned_title} {cleaned_selftext} {ocr_text}".strip()
        keywords = extract_keywords(combined)
        topics   = classify_topics(combined)

        conn.execute(
            "UPDATE posts SET cleaned_title=?, cleaned_selftext=?, masked_author=?, "
            "readable_date=?, ocr_text=?, keywords=?, topics=?, is_processed=1 "
            "WHERE post_id=?",
            (cleaned_title, cleaned_selftext, masked, readable_date,
             ocr_text, ", ".join(keywords), ", ".join(topics), p["post_id"]),
        )

        if i % 100 == 0:
            conn.commit()
            log.info("  %d / %d done ...", i, len(posts))

    conn.commit()
    conn.close()

    log.info("=" * 50)
    log.info("DONE - %d posts processed, %d OCR extractions", len(posts), ocr_count)
    log.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process Reddit data.")
    parser.add_argument("--db", default="reddit_posts.db", help="SQLite database path")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip image OCR extraction")
    args = parser.parse_args()
    post_process(db_path=args.db, skip_ocr=args.skip_ocr)