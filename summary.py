import os
import sqlite3

def print_summary(db_path: str):
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM posts")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT subreddit, COUNT(*) as cnt
        FROM posts GROUP BY subreddit ORDER BY cnt DESC LIMIT 10
    """)
    by_sub = cur.fetchall()

    cur.execute("SELECT AVG(score), AVG(num_comments) FROM posts")
    avg_score, avg_comments = cur.fetchone()

    cur.execute("""
        SELECT title, subreddit, score FROM posts ORDER BY score DESC LIMIT 5
    """)
    top_posts = cur.fetchall()

    conn.close()

    print("\nDATABASE SUMMARY")
    print(f"Total posts: {total:<48}")
    print(f"Avg score:   {avg_score or 0:<48.1f}")
    print(f"Avg comments:{avg_comments or 0:<48.1f}")
    print("")
    print("Posts by subreddit (top 10):")
    for sub, cnt in by_sub:
        print(f"    r/{sub:<20} {cnt:>6} posts")
    print("")
    print("Top 5 posts by score:")
    for title, sub, score in top_posts:
        short = (title[:42] + "…") if len(title) > 43 else title
        print(f"    [{score:>6}] r/{sub:<14} {short:<28}")
    print("\n")