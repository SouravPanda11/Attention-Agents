/* eslint-disable @typescript-eslint/no-require-imports */
require("../scripts/register-typescript.cjs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const Database = require("better-sqlite3");

test("additive migrations preserve existing v0 records", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "survey-benchmark-migration-"));
  const file = path.join(directory, "legacy.sqlite");
  const legacy = new Database(file);
  legacy.exec(`CREATE TABLE submissions (
    id INTEGER PRIMARY KEY, run_id TEXT, workflow_id TEXT, suite_version TEXT, order_id TEXT,
    answers TEXT, UNIQUE(run_id, workflow_id));
    INSERT INTO submissions VALUES (1, 'old-run', 'v0-workflow', 'v0', 'order01', '{"old":"answer"}');`);
  legacy.close();
  process.env.SURVEY_DB_PATH = file;
  let database;
  try {
    database = require("../lib/db.ts").getDatabase();
    const row = database.prepare("SELECT * FROM submissions").get();
    assert.equal(row.suite_version, "v0");
    assert.equal(row.answers, '{"old":"answer"}');
    assert.equal(row.sample_id, null);
    assert.equal(row.order_seed, null);
    assert.equal(row.repeat_index, null);
  } finally {
    database?.close();
    for (const name of fs.readdirSync(directory)) fs.unlinkSync(path.join(directory, name));
    fs.rmdirSync(directory);
  }
});
