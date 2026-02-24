import json
import sqlite3
from pathlib import Path


def main() -> int:
    db_path = Path(__file__).resolve().parents[1] / "data.sqlite"
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    events_count = cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    submissions_count = cur.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    print(f"db={db_path}")
    print(f"events={events_count}")
    print(f"submissions={submissions_count}")

    print("\nEvent types:")
    for event_type, count in cur.execute(
        "SELECT event_type, COUNT(*) FROM events GROUP BY event_type ORDER BY COUNT(*) DESC, event_type ASC"
    ).fetchall():
        print(f"- {event_type}: {count}")

    print("\nLatest submissions:")
    rows = cur.execute(
        """
        SELECT ts, session_id, text_answers, image_answers, text_attention_value, image_attention_choice, captcha_input
        FROM submissions
        ORDER BY ts DESC
        LIMIT 10
        """
    ).fetchall()
    for ts, sid, text_answers, image_answers, text_attention_value, image_attention_choice, captcha_input in rows:
        text_count = len(json.loads(text_answers or "{}"))
        image_count = len(json.loads(image_answers or "{}"))
        print(
            f"- ts={ts} session={sid} text_answers={text_count} image_answers={image_count} "
            f"text_attn='{text_attention_value}' image_attn='{image_attention_choice}' captcha='{captcha_input}'"
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
