import html
import re
from datetime import datetime, timezone
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def process_post(raw: dict) -> dict:
    created = datetime.fromtimestamp(raw.get("created_utc", 0), tz=timezone.utc)
    return {
        "post_id":       raw.get("id", ""),
        "subreddit":     raw.get("subreddit", ""),
        "title":         clean_text(raw.get("title")),
        "author":        raw.get("author", "[deleted]"),
        "score":         raw.get("score", 0),
        "upvote_ratio":  raw.get("upvote_ratio", 0.0),
        "num_comments":  raw.get("num_comments", 0),
        "created_utc":   created.isoformat(),
        "url":           raw.get("url", ""),
        "permalink":     f"https://www.reddit.com{raw.get('permalink', '')}",
        "selftext":      clean_text(raw.get("selftext")),
        "is_self":       int(raw.get("is_self", False)),
        "over_18":       int(raw.get("over_18", False)),
        "flair":         raw.get("link_flair_text", ""),
        "domain":        raw.get("domain", ""),
        "fetched_at":    datetime.now(tz=timezone.utc).isoformat(),
    }
