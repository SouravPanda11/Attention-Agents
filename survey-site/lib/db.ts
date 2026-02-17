import Database from "better-sqlite3";
import path from "path";

const dbPath = path.join(process.cwd(), "data.sqlite");
export const db = new Database(dbPath);

// Create tables (idempotent)
db.exec(`
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  session_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  session_id TEXT NOT NULL,
  text_answers TEXT NOT NULL,
  text_attention_value TEXT NOT NULL,
  text_attention_ok INTEGER NOT NULL,
  image_answers TEXT NOT NULL,
  captcha_input TEXT NOT NULL,
  captcha_ok INTEGER NOT NULL,
  image_attention_choice TEXT NOT NULL,
  image_attention_ok INTEGER NOT NULL,
  overall_ok INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_submissions_session ON submissions(session_id);
`);
