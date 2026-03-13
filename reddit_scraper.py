import argparse
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
import time
import html
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from sqlite_db import init_db, insert_posts
from process import process_post, clean_text
from summary import print_summary

from typing import Dict, List, Optional, Set, Tuple

with open("config.json", "r") as f:
    config = json.load(f)

DEFAULT_SUBREDDITS = config["DEFAULT_SUBREDDITS"]
SORT_OPTIONS = config["SORT_OPTIONS"]
TIME_FILTERS = config["TIME_FILTERS"]
MAX_PER_REQUEST, LISTING_CEILING = config["MAX_PER_REQUEST"], config["LISTING_CEILING"]
PER_REQUEST_TIMEOUT, DEFAULT_MAX_WALL_CLOCK = config["PER_REQUEST_TIMEOUT"], config["DEFAULT_MAX_WALL_CLOCK"]
REQUEST_DELAY_MIN, REQUEST_DELAY_MAX = config["REQUEST_DELAY_MIN"], config["REQUEST_DELAY_MAX"]
MAX_RETRIES, DB_FILE, USER_AGENT = config["MAX_RETRIES"], config["DB_FILE"], config["USER_AGENT"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reddit_scraper")


def fetch_json(url: str, retries: int = MAX_RETRIES) -> Optional[dict]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=PER_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                wait = min(2 ** attempt + random.random(), 60)
                log.warning("Rate-limited (429). Backing off %.1fs …", wait)
                time.sleep(wait)
            elif e.code >= 500:
                wait = 2 ** attempt + random.random()
                log.warning("Server error %d. Retry %d/%d in %.1fs",
                            e.code, attempt, retries, wait)
                time.sleep(wait)
            else:
                log.error("HTTP %d for %s — not retrying", e.code, url)
                return None
        except (URLError, TimeoutError, OSError) as e:
            wait = 2 ** attempt + random.random()
            log.warning("Network error: %s. Retry %d/%d in %.1fs",
                        e, attempt, retries, wait)
            time.sleep(wait)
    log.error("All %d retries exhausted for %s", retries, url)
    return None



def fetch_listing(
    subreddit: str,
    sort: str = "hot",
    time_filter: str = "all",
    limit: int = MAX_PER_REQUEST,
    after: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str]]:
    params = {"limit": min(limit, MAX_PER_REQUEST), "raw_json": 1}
    if after:
        params["after"] = after
    if sort == "top":
        params["t"] = time_filter

    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?{urlencode(params)}"
    data = fetch_json(url)
    if not data or "data" not in data:
        return [], None

    children = data["data"].get("children", [])
    posts = [c["data"] for c in children if c.get("kind") == "t3"]
    next_after = data["data"].get("after")
    return posts, next_after


def scrape_reddit(
    num_posts: int,
    subreddits: List[str],
    db_path: str = DB_FILE,
    max_wall_clock: int = DEFAULT_MAX_WALL_CLOCK,
) -> int:
    conn = init_db(db_path)
    seen_ids = set()  # type: Set[str]
    total_stored = 0
    start_time = time.monotonic()

    combos = []
    for sub in subreddits:
        for sort in SORT_OPTIONS:
            if sort == "top":
                for tf in TIME_FILTERS:
                    combos.append((sub, sort, tf))
            else:
                combos.append((sub, sort, "all"))
    random.shuffle(combos)

    combos_needed = math.ceil(num_posts / LISTING_CEILING)
    if len(combos) < combos_needed:
        log.warning(
            "Only %d listing combos available; may not reach %d posts. "
            "Add more subreddits for better coverage.",
            len(combos), num_posts,
        )

    log.info(
        "Starting scrape: target=%d posts, subreddits=%d, combos=%d, timeout=%ds",
        num_posts, len(subreddits), len(combos), max_wall_clock,
    )

    combo_idx = 0
    while total_stored < num_posts and combo_idx < len(combos):
        elapsed = time.monotonic() - start_time
        if elapsed >= max_wall_clock:
            log.warning(
                "Wall-clock limit (%ds) reached after %.0fs. "
                "Stopping with %d/%d posts.",
                max_wall_clock, elapsed, total_stored, num_posts,
            )
            break

        sub, sort, tf = combos[combo_idx]
        combo_idx += 1
        after_cursor = None
        pages_in_combo = 0

        log.info(" Combo %d/%d: r/%s sort=%s t=%s",
                 combo_idx, len(combos), sub, sort, tf)

        while total_stored < num_posts:
            elapsed = time.monotonic() - start_time
            if elapsed >= max_wall_clock:
                break

            remaining = num_posts - total_stored
            fetch_limit = min(remaining, MAX_PER_REQUEST)

            raw_posts, next_after = fetch_listing(
                sub, sort=sort, time_filter=tf,
                limit=fetch_limit, after=after_cursor,
            )

            if not raw_posts:
                log.info("  No more posts from this combo.")
                break

            new_posts = []
            for rp in raw_posts:
                pid = rp.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    new_posts.append(process_post(rp))

            if new_posts:
                batch_size = min(len(new_posts), num_posts - total_stored)
                inserted = insert_posts(conn, new_posts[:batch_size])
                total_stored += inserted
                pages_in_combo += 1

                log.info(
                    "  Page %d: fetched %d, new %d, stored %d (total %d/%d)",
                    pages_in_combo, len(raw_posts), len(new_posts),
                    inserted, total_stored, num_posts,
                )

            if not next_after:
                log.info("  Pagination exhausted for this combo.")
                break

            after_cursor = next_after

            delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
            time.sleep(delay)

            if pages_in_combo * MAX_PER_REQUEST >= LISTING_CEILING:
                log.info("  Hit ~1000-post listing ceiling. Moving to next combo.")
                break

    elapsed = time.monotonic() - start_time
    conn.close()

    log.info("DONE  %d unique posts stored in %s", total_stored, db_path)
    log.info("      Wall-clock: %.1fs  Combos used: %d/%d",
             elapsed, combo_idx, len(combos))

    return total_stored


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Reddit posts (no API key) and store in SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--posts", "-n", type=int, required=True,
        help="Number of posts to fetch (e.g. 5000)",
    )
    parser.add_argument(
        "--subreddits", "-s", type=str, default=None,
        help="Comma-separated list of subreddits (default: built-in list of 20)",
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=DEFAULT_MAX_WALL_CLOCK,
        help=f"Max wall-clock seconds (default: {DEFAULT_MAX_WALL_CLOCK})",
    )
    parser.add_argument(
        "--db", type=str, default=DB_FILE,
        help=f"SQLite database file (default: {DB_FILE})",
    )
    args = parser.parse_args()

    if args.posts <= 0:
        parser.error("--posts must be a positive integer")

    subreddits = (
        [s.strip() for s in args.subreddits.split(",") if s.strip()]
        if args.subreddits
        else DEFAULT_SUBREDDITS
    )

    total = scrape_reddit(
        num_posts=args.posts,
        subreddits=subreddits,
        db_path=args.db,
        max_wall_clock=args.timeout,
    )

    print_summary(args.db)


if __name__ == "__main__":
    main()