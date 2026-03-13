import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reddit_scraper import scrape_reddit, DEFAULT_SUBREDDITS
from post_process import post_process
from vector import vectorize_posts
from cluster import run_pipeline as run_clustering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("automate")

DEFAULT_DB = "reddit_posts.db"

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

def run_full_pipeline(db_path: str, num_posts: int, subreddits: List[str],
                      num_clusters: int):
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"\n  PIPELINE UPDATE — {timestamp}")

    try:
        print("\nFetching data from Reddit ")
        scrape_reddit(num_posts=num_posts, subreddits=subreddits, db_path=db_path)
        print("       Scraping complete.")
    except Exception as e:
        print(f"       Scraping failed: {e}")
        log.error("Scraping error: %s", e)

    try:
        print("\nProcessing and cleaning data ")
        post_process(db_path=db_path, skip_ocr=True)
        print("        Preprocessing complete.")
    except Exception as e:
        print(f"        Preprocessing failed: {e}")
        log.error("Preprocessing error: %s", e)

    try:
        print("\n Generating vector embeddings ")
        vectorize_posts(db_path=db_path, method="transformer")
        print("        Embedding complete.")
    except Exception as e:
        print(f"        Embedding failed: {e}")
        log.error("Embedding error: %s", e)

    try:
        print("\nClustering posts ...")
        run_clustering(db_path=db_path, k=num_clusters)
        print("        Clustering complete.")
    except Exception as e:
        print(f"        Clustering failed: {e}")
        log.error("Clustering error: %s", e)

    try:
        conn = sqlite3.connect(db_path)
        total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        embedded = conn.execute("SELECT COUNT(*) FROM posts WHERE embedding != '' AND embedding IS NOT NULL").fetchone()[0]
        clustered = conn.execute("SELECT COUNT(*) FROM posts WHERE cluster_id IS NOT NULL").fetchone()[0]
        conn.close()
        print(f"\n  Database updated: {total} total, {embedded} embedded, {clustered} clustered")
    except Exception:
        pass

    print("\n")

def find_matching_cluster(query: str, db_path: str):
    if not os.path.exists(db_path):
        print("  Database not found. Run pipeline first.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    check = conn.execute(
        "SELECT COUNT(*) FROM posts WHERE cluster_id IS NOT NULL"
    ).fetchone()[0]
    if check == 0:
        print("  No clusters found. Pipeline must run first.")
        conn.close()
        return

    best_cluster = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vec = model.encode([query])[0]

        clusters = conn.execute(
            "SELECT DISTINCT cluster_id FROM posts WHERE cluster_id IS NOT NULL"
        ).fetchall()

        best_score, best_cluster = -1, 0
        cluster_scores = {}

        for row in clusters:
            cid = row["cluster_id"]
            embeddings = conn.execute(
                "SELECT embedding FROM posts WHERE cluster_id = ? AND embedding != ''",
                (cid,)
            ).fetchall()
            if not embeddings:
                continue
            vecs = np.array([json.loads(e["embedding"]) for e in embeddings])
            centroid = vecs.mean(axis=0)
            sim = np.dot(query_vec, centroid) / (np.linalg.norm(query_vec) * np.linalg.norm(centroid) + 1e-10)
            cluster_scores[cid] = float(sim)
            if sim > best_score:
                best_score, best_cluster = sim, cid

        print(f"\n  Query: \"{query}\"")
        print(f"  Best match: Cluster {best_cluster} (similarity: {best_score:.4f})")
        print("\n  Cluster similarity scores:")
        for cid in sorted(cluster_scores, key=cluster_scores.get, reverse=True):
            bar = " " * int(cluster_scores[cid] * 40)
            print(f"    Cluster {cid}: {cluster_scores[cid]:.4f} {bar}")

    except ImportError:
        query_tokens = set(re.findall(r"\b[a-z]{3,}\b", query.lower())) - STOP_WORDS
        clusters = conn.execute(
            "SELECT DISTINCT cluster_id, cluster_keywords FROM posts WHERE cluster_id IS NOT NULL"
        ).fetchall()

        best_score, best_cluster = 0, 0
        cluster_scores = {}
        seen = set()
        for row in clusters:
            cid = row["cluster_id"]
            if cid in seen:
                continue
            seen.add(cid)
            kws = set(row["cluster_keywords"].lower().split(", ")) if row["cluster_keywords"] else set()
            overlap = len(query_tokens & kws)
            cluster_scores[cid] = overlap
            if overlap > best_score:
                best_score, best_cluster = overlap, cid

        print(f"\n  Query: \"{query}\"")
        print(f"  Best match: Cluster {best_cluster} (keyword overlap: {best_score})")

    posts = conn.execute(
        "SELECT cleaned_title, score, subreddit, keywords, cluster_keywords "
        "FROM posts WHERE cluster_id = ? ORDER BY score DESC",
        (best_cluster,)
    ).fetchall()
    conn.close()

    if not posts:
        print("  No posts found in this cluster.")
        return

    cluster_kws = posts[0]["cluster_keywords"] or ""

    print(f"\n   Cluster {best_cluster}    {len(posts)} posts    Keywords: {cluster_kws}")
    for p in posts[:15]:
        title = (p["cleaned_title"] or "(no title)")[:75]
        print(f"    [{p['score']:>5}] r/{p['subreddit']:<14} {title}")
    if len(posts) > 15:
        print(f"     and {len(posts) - 15} more posts")

    generate_cluster_visual(best_cluster, posts, cluster_kws, query)


def generate_cluster_visual(cluster_id: int, posts: list, cluster_kws: str, query: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    all_text = " ".join((p["cleaned_title"] or "") for p in posts).lower()
    tokens = [w for w in re.findall(r"\b[a-z]{3,}\b", all_text) if w not in STOP_WORDS]
    top_kw = Counter(tokens).most_common(10)

    if top_kw:
        words, counts = zip(*top_kw)
        colors = ["#e74c3c" if w in query.lower() else "#3498db" for w in words]
        ax.barh(list(reversed(words)), list(reversed(counts)), color=list(reversed(colors)))
        ax.set_xlabel("Frequency")
        ax.set_title(f"Cluster {cluster_id} — Top Keywords")
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="#e74c3c", lw=6, label="Query match"),
                plt.Line2D([0], [0], color="#3498db", lw=6, label="Cluster keyword"),
            ],
            loc="lower right",
        )

    fig.suptitle(f'Search: "{query}" → Cluster {cluster_id} ({len(posts)} posts)',
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_file = "cluster_search_result.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Visualization saved → {output_file}\n")


def scheduler_loop(interval_min: int, db_path: str, num_posts: int,
                   subreddits: List[str], num_clusters: int, stop_event: threading.Event):
    while not stop_event.is_set():
        for _ in range(interval_min * 60):
            if stop_event.is_set():
                return
            time.sleep(1)
        if not stop_event.is_set():
            run_full_pipeline(db_path, num_posts, subreddits, num_clusters)


def interactive_prompt(db_path: str, stop_event: threading.Event):
    print("\n  INTERACTIVE MODE")
    print("  Type keywords or a message to find the matching cluster.")
    print("  Commands: 'status', 'quit', 'help'")
    print("\n")

    while not stop_event.is_set():
        try:
            query = input("🔍 Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down ")
            stop_event.set()
            break

        if not query:
            continue
        elif query.lower() in ("quit", "exit", "q"):
            print("Shutting down ")
            stop_event.set()
            break
        elif query.lower() == "help":
            print("  Enter keywords to find matching clusters (e.g. 'cancer treatment')")
            print("  'status'  — show database stats")
            print("  'quit'    — exit the program\n")
        elif query.lower() == "status":
            try:
                conn = sqlite3.connect(db_path)
                total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
                clustered = conn.execute("SELECT COUNT(*) FROM posts WHERE cluster_id IS NOT NULL").fetchone()[0]
                clusters = conn.execute(
                    "SELECT cluster_id, COUNT(*) as cnt, cluster_keywords "
                    "FROM posts WHERE cluster_id IS NOT NULL "
                    "GROUP BY cluster_id ORDER BY cnt DESC"
                ).fetchall()
                conn.close()
                print(f"\n  Total posts: {total}, Clustered: {clustered}")
                for cid, cnt, kws in clusters:
                    print(f"    Cluster {cid}: {cnt:>5} posts │ {(kws or '')[:50]}")
                print()
            except Exception as e:
                print(f"  Error reading database: {e}\n")
        else:
            find_matching_cluster(query, db_path)

def main():
    parser = argparse.ArgumentParser(
        description="Automated Reddit pipeline with interactive cluster search.",
    )
    parser.add_argument(
        "interval", type=int,
        help="Update interval in minutes",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--posts", type=int, default=200, help="Posts to fetch per update (default: 200)")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument(
        "--subreddits", type=str, default=None,
        help="Comma-separated subreddits (default: built-in list)",
    )
    args = parser.parse_args()

    subreddits = (
        [s.strip() for s in args.subreddits.split(",") if s.strip()]
        if args.subreddits else DEFAULT_SUBREDDITS
    )

    print(f"  Interval : {args.interval} minutes")
    print(f"  Posts/run: {args.posts}")
    print(f"  Clusters : {args.clusters}")
    print(f"  Database : {args.db}")
    print(f"  Subreddits: {', '.join(subreddits[:5])}{'...' if len(subreddits) > 5 else ''}")

    run_full_pipeline(args.db, args.posts, subreddits, args.clusters)

    stop_event = threading.Event()
    scheduler = threading.Thread(
        target=scheduler_loop,
        args=(args.interval, args.db, args.posts, subreddits, args.clusters, stop_event),
        daemon=True,
    )
    scheduler.start()

    try:
        interactive_prompt(args.db, stop_event)
    except KeyboardInterrupt:
        print("\nShutting down ...")
        stop_event.set()

    scheduler.join(timeout=5)


if __name__ == "__main__":
    main()
