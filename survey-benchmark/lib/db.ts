import Database from "better-sqlite3";
import path from "path";

let database: Database.Database | undefined;

export function getDatabase(): Database.Database {
  if (database) return database;
  database = new Database(path.join(process.cwd(), "benchmark.sqlite"));
  database.pragma("journal_mode = WAL");
  database.exec(`
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL,
      session_id TEXT NOT NULL,
      run_id TEXT NOT NULL,
      workflow_id TEXT NOT NULL,
      event_type TEXT NOT NULL,
      page_index INTEGER,
      payload TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS submissions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL,
      session_id TEXT NOT NULL,
      run_id TEXT NOT NULL,
      workflow_id TEXT NOT NULL,
      suite_version TEXT NOT NULL,
      profile TEXT NOT NULL,
      occurrence INTEGER NOT NULL,
      layout TEXT NOT NULL,
      order_id TEXT NOT NULL,
      content_version INTEGER NOT NULL,
      ordered_question_ids TEXT NOT NULL,
      answers TEXT NOT NULL,
      answer_count INTEGER NOT NULL,
      UNIQUE(run_id, workflow_id)
    );

    CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
    CREATE INDEX IF NOT EXISTS idx_events_workflow ON events(workflow_id);
    CREATE INDEX IF NOT EXISTS idx_submissions_workflow ON submissions(workflow_id);
  `);
  return database;
}
