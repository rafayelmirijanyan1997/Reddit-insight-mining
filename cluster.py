import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("cluster")

DEFAULT_DB = "reddit_posts.db"
def load_embedded_posts(db_path: str) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT post_id, embedding, cleaned_title, cleaned_selftext, keywords "
        "FROM posts WHERE embedding IS NOT NULL AND embedding != ''"
    ).fetchall()
    conn.close()

    if not rows:
        log.error("No embedded posts found. Run vector.py first.")
        sys.exit(1)

    post_ids = [r["post_id"] for r in rows]
    embeddings = np.array([json.loads(r["embedding"]) for r in rows])
    texts = [f"{r['cleaned_title'] or ''} {r['cleaned_selftext'] or ''}".strip() for r in rows]
    titles = [r["cleaned_title"] or "(no title)" for r in rows]

    log.info("Loaded %d posts with %d-dim embeddings", len(post_ids), embeddings.shape[1])
    return post_ids, embeddings, texts, titles


def find_optimal_k(embeddings: np.ndarray, k_range: range = range(2, 11)) -> int:
    log.info("Finding optimal K (range %d–%d) ...", k_range.start, k_range.stop - 1)
    best_k, best_score = 2, -1

    scores = {}
    for k in k_range:
        if k >= len(embeddings):
            break
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
        scores[k] = score
        log.info("  K=%d → silhouette=%.4f", k, score)
        if score > best_score:
            best_k, best_score = k, score

    plt.figure(figsize=(8, 4))
    plt.plot(list(scores.keys()), list(scores.values()), "bo-")
    plt.axvline(x=best_k, color="r", linestyle="--", label=f"Best K={best_k}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Optimal K — Silhouette Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cluster_elbow.png", dpi=150)
    plt.close()
    log.info("Saved elbow chart → cluster_elbow.png")
    log.info("Optimal K = %d (silhouette = %.4f)", best_k, best_score)
    return best_k


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


def extract_cluster_keywords(texts: List[str], top_n: int = 10) -> List[str]:
    all_tokens = re.findall(r"\b[a-z]{3,}\b", " ".join(texts).lower())
    filtered = [w for w in all_tokens if w not in STOP_WORDS]
    return [w for w, _ in Counter(filtered).most_common(top_n)]


def cluster_posts(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, "KMeans"]:
    log.info("Clustering into %d groups ...", k)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
    log.info("Clustering done — silhouette score: %.4f", score)
    return labels, model


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, titles: List[str],
                       cluster_kw: Dict[int, List[str]]):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    k = len(set(labels))
    cmap = plt.cm.get_cmap("tab10", k)

    plt.figure(figsize=(14, 9))
    for c in range(k):
        mask = labels == c
        kw_label = ", ".join(cluster_kw.get(c, [])[:5])
        plt.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(c)], label=f"C{c}: {kw_label}",
                    alpha=0.6, s=20, edgecolors="none")

        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        plt.annotate(f"C{c}", (cx, cy), fontsize=11, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc=cmap(c), alpha=0.7))

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(f"Reddit Post Clusters (K={k}) — PCA Projection")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    sizes = [np.sum(labels == c) for c in range(k)]
    colors = [cmap(c) for c in range(k)]
    bar_labels = [f"C{c}\n({', '.join(cluster_kw.get(c, [])[:3])})" for c in range(k)]

    plt.figure(figsize=(max(8, k * 1.2), 5))
    plt.bar(bar_labels, sizes, color=colors, edgecolor="white")
    plt.ylabel("Number of Posts")
    plt.title("Cluster Sizes with Top Keywords")
    for i, v in enumerate(sizes):
        plt.text(i, v + max(sizes) * 0.01, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("cluster_sizes.png", dpi=150)
    plt.close()


def verify_clusters(labels: np.ndarray, titles: List[str], texts: List[str],
                    cluster_kw: Dict[int, List[str]], sample: int = 5):
    k = len(set(labels))
    print("\n")
    print("CLUSTER VERIFICATION — Sample posts per cluster")

    for c in range(k):
        mask = np.where(labels == c)[0]
        kws = ", ".join(cluster_kw.get(c, []))
        print(f"\n Cluster {c}    {len(mask)} posts   Keywords: {kws}")
        for idx in mask[:sample]:
            title = titles[idx][:80]
            print(f" {title}")
        if len(mask) > sample:
            print(f" and {len(mask) - sample} more")


def store_clusters(db_path: str, post_ids: List[str], labels: np.ndarray,
                   cluster_kw: Dict[int, List[str]]):
    conn = sqlite3.connect(db_path)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(posts)").fetchall()}
    for col, dtype in {"cluster_id": "INTEGER", "cluster_keywords": "TEXT DEFAULT ''"}.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE posts ADD COLUMN {col} {dtype}")

    for pid, label in zip(post_ids, labels):
        kws = ", ".join(cluster_kw.get(int(label), []))
        conn.execute("UPDATE posts SET cluster_id=?, cluster_keywords=? WHERE post_id=?",
                     (int(label), kws, pid))

    conn.commit()
    conn.close()
    log.info("Stored cluster_id and cluster_keywords in database.")


def run_pipeline(db_path: str = DEFAULT_DB, k: int = 0):
    post_ids, embeddings, texts, titles = load_embedded_posts(db_path)

    if k <= 0:
        k = find_optimal_k(embeddings)

    labels, model = cluster_posts(embeddings, k)

    cluster_kw = {}
    for c in range(k):
        mask = np.where(labels == c)[0]
        cluster_texts = [texts[i] for i in mask]
        cluster_kw[c] = extract_cluster_keywords(cluster_texts)

    visualize_clusters(embeddings, labels, titles, cluster_kw)

    verify_clusters(labels, titles, texts, cluster_kw)

    store_clusters(db_path, post_ids, labels, cluster_kw)
    print("\n")
    print(f"SUMMARY: {len(post_ids)} posts → {k} clusters")
    for c in range(k):
        count = int(np.sum(labels == c))
        kws = ", ".join(cluster_kw[c][:5])
        print(f"  Cluster {c}: {count:>5} posts │ {kws}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster Reddit posts by embedding similarity.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--clusters", "-k", default="auto",
                        help="Number of clusters: integer or 'auto' (default: auto)")
    args = parser.parse_args()
    k = 0 if args.clusters == "auto" else int(args.clusters)
    run_pipeline(db_path=args.db, k=k)