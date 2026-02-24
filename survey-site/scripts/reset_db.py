import sqlite3
from pathlib import Path


def main() -> int:
    db_path = Path(__file__).resolve().parents[1] / "data.sqlite"
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM events")
    cur.execute("DELETE FROM submissions")
    conn.commit()
    cur.execute("VACUUM")
    conn.commit()

    events_count = cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    submissions_count = cur.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    conn.close()

    print(f"Reset complete: {db_path}")
    print(f"events={events_count}, submissions={submissions_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
