import Database from "better-sqlite3";
import path from "path";

let database: Database.Database | undefined;

export function getDatabase(): Database.Database {
  if (database) return database;
  const configuredPath = process.env.SURVEY_DB_PATH?.trim();
  database = new Database(configuredPath || path.join(process.cwd(), "benchmark.sqlite"));
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
      attention_check_content_version INTEGER NOT NULL DEFAULT 1,
      ordered_question_ids TEXT NOT NULL,
      answers TEXT NOT NULL,
      answer_count INTEGER NOT NULL,
      valid_answer_count INTEGER NOT NULL DEFAULT 0,
      invalid_answer_count INTEGER NOT NULL DEFAULT 0,
      skipped_question_count INTEGER NOT NULL DEFAULT 0,
      attention_check_count INTEGER NOT NULL DEFAULT 0,
      attention_check_scored_count INTEGER NOT NULL DEFAULT 0,
      attention_check_unscored_count INTEGER NOT NULL DEFAULT 0,
      attention_check_attempted_count INTEGER NOT NULL DEFAULT 0,
      attention_check_pass_count INTEGER NOT NULL DEFAULT 0,
      attention_check_fail_count INTEGER NOT NULL DEFAULT 0,
      attention_check_skipped_count INTEGER NOT NULL DEFAULT 0,
      attention_check_results TEXT NOT NULL DEFAULT '[]',
      UNIQUE(run_id, workflow_id)
    );

    CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
    CREATE INDEX IF NOT EXISTS idx_events_workflow ON events(workflow_id);
    CREATE INDEX IF NOT EXISTS idx_submissions_workflow ON submissions(workflow_id);
  `);

  const submissionColumns = new Set(
    (database.prepare("PRAGMA table_info(submissions)").all() as { name: string }[]).map((column) => column.name)
  );
  const integerColumns = [
    "valid_answer_count",
    "invalid_answer_count",
    "skipped_question_count",
    "attention_check_count",
    "attention_check_scored_count",
    "attention_check_unscored_count",
    "attention_check_attempted_count",
    "attention_check_pass_count",
    "attention_check_fail_count",
    "attention_check_skipped_count",
    "attention_check_content_version",
  ] as const;
  for (const column of integerColumns) {
    if (!submissionColumns.has(column)) {
      const defaultValue = column === "attention_check_content_version" ? 1 : 0;
      database.exec(`ALTER TABLE submissions ADD COLUMN ${column} INTEGER NOT NULL DEFAULT ${defaultValue}`);
    }
  }
  if (!submissionColumns.has("attention_check_results")) {
    database.exec("ALTER TABLE submissions ADD COLUMN attention_check_results TEXT NOT NULL DEFAULT '[]'");
  }
  return database;
}
