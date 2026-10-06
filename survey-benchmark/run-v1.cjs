/* eslint-disable @typescript-eslint/no-require-imports */
// AC samples only: O1, then O2 navigation/item, ... through O8 navigation/item.
// Finish all samples and their orders/repeats within each layout.
const path = require("node:path");
if (!process.argv.slice(2).includes("--output")) {
  process.argv.push("--output", path.join(__dirname, "run-v1.jsonl"));
}
process.argv.push("--no-baselines");
require("./scripts/export-run-plan.cjs");
