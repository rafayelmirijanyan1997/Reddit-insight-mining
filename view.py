import argparse
import sqlite3
import sys

DEFAULT_DB = "reddit_posts.db"


def run_query(db_path: str, sql: str):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)

        if cursor.description is None:
            conn.commit()
            print(f"Query executed successfully. Rows affected: {cursor.rowcount}")
            conn.close()
            return

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("No results found.")
            return

        widths = []
        for i, col in enumerate(columns):
            max_val = max((len(str(row[i])) for row in rows), default=0)
            widths.append(min(max(len(col), max_val), 40))

        header = "  ".join(col.ljust(w) for col, w in zip(columns, widths))
        print(header)

        for row in rows:
            values = []
            for val, w in zip(row, widths):
                s = str(val) if val is not None else "NULL"
                if len(s) > w:
                    s = s[: w - 1] + "…"
                values.append(s.ljust(w))
            print("  ".join(values))

        print(f"\n({len(rows)} rows)")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run SQL queries against a SQLite database.")
    parser.add_argument("sql", help='SQL query to execute, e.g. "SELECT * FROM posts LIMIT 10"')
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Path to SQLite database (default: {DEFAULT_DB})")
    args = parser.parse_args()

    run_query(args.db, args.sql)


if __name__ == "__main__":
    main()
