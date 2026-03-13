import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from typing import List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("vector")

DEFAULT_DB = "reddit_posts.db"


def embed_texts(texts: List[str], method: str = "transformer") -> List[List[float]]:
    if not texts:
        return []
    texts = [t if t.strip() else "empty" for t in texts]

    if method == "transformer":
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence transformers model")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        return [emb.tolist() for emb in embeddings]
    else:
        import gensim.downloader as api
        log.info("Loading Word2Vec model")
        model = api.load("word2vec-google-news-300")
        vectors = []
        for text in texts:
            tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text)
            word_vecs = [model[w] for w in tokens if w in model]
            vectors.append(np.mean(word_vecs, axis=0).tolist() if word_vecs
                           else np.zeros(model.vector_size).tolist())
        return vectors


def vectorize_posts(db_path: str = DEFAULT_DB, method: str = "transformer"):
    if not os.path.exists(db_path):
        log.error("Database not found: %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    existing = {r[1] for r in conn.execute("PRAGMA table_info(posts)").fetchall()}
    if "embedding" not in existing:
        conn.execute("ALTER TABLE posts ADD COLUMN embedding TEXT DEFAULT ''")
        conn.commit()

    conn.row_factory = sqlite3.Row
    posts = conn.execute(
        "SELECT post_id, cleaned_title, cleaned_selftext, ocr_text FROM posts"
    ).fetchall()

    if not posts:
        log.info("No posts found in database.")
        conn.close()
        return

    log.info("Embedding %d posts using '%s' method ...", len(posts), method)

    texts = [
        f"{p['cleaned_title'] or ''} {p['cleaned_selftext'] or ''} {p['ocr_text'] or ''}".strip()
        for p in posts
    ]
    post_ids = [p["post_id"] for p in posts]

    vectors = embed_texts(texts, method=method)
    dim_size = len(vectors[0]) if vectors else 0

    for pid, vec in zip(post_ids, vectors):
        conn.execute("UPDATE posts SET embedding = ? WHERE post_id = ?",
                     (json.dumps(vec), pid))

    conn.commit()
    conn.close()
    log.info("DONE %d posts embedded (%d-dim vectors)", len(vectors), dim_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed Reddit posts into vectors.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--method", choices=["transformer", "word2vec"], default="transformer",
                        help="Embedding method (default: transformer)")
    args = parser.parse_args()
    vectorize_posts(db_path=args.db, method=args.method)