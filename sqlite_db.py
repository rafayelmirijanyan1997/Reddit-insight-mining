import sqlite3
from typing import List, Dict

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id         TEXT PRIMARY KEY,
            subreddit       TEXT NOT NULL,
            title           TEXT,
            author          TEXT,
            score           INTEGER,
            upvote_ratio    REAL,
            num_comments    INTEGER,
            created_utc     TEXT,
            url             TEXT,
            permalink       TEXT,
            selftext        TEXT,
            is_self         INTEGER,
            over_18         INTEGER,
            flair           TEXT,
            domain          TEXT,
            fetched_at      TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_subreddit ON posts(subreddit)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_created ON posts(created_utc)
    """)
    conn.commit()
    return conn



def insert_posts(conn: sqlite3.Connection, posts: List[Dict]) -> int:
    if not posts:
        return 0
    inserted = 0
    for p in posts:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO posts
                   (post_id, subreddit, title, author, score, upvote_ratio,
                    num_comments, created_utc, url, permalink, selftext,
                    is_self, over_18, flair, domain, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    p["post_id"], p["subreddit"], p["title"], p["author"],
                    p["score"], p["upvote_ratio"], p["num_comments"],
                    p["created_utc"], p["url"], p["permalink"], p["selftext"],
                    p["is_self"], p["over_18"], p["flair"], p["domain"],
                    p["fetched_at"],
                ),
            )
            if conn.total_changes:
                inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return inserted
