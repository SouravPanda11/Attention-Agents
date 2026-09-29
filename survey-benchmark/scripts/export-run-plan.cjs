/* eslint-disable @typescript-eslint/no-require-imports */
require("./register-typescript.cjs");
const fs = require("node:fs");
const { getRunPlan } = require("../lib/benchmark/runPlan.ts");
const { isOccurrence, isLayoutMode } = require("../lib/benchmark/schema.ts");

const args = process.argv.slice(2);
const options = {};
let output;
for (let index = 0; index < args.length; index++) {
  const flag = args[index];
  if (flag === "--no-baselines") options.includeBaselines = false;
  else if (["--output", "--occurrence", "--repeats", "--layout"].includes(flag)) {
    const value = args[++index];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${flag}`);
    if (flag === "--output") output = value;
    else if (flag === "--occurrence") {
      const occurrence = Number(value);
      if (!isOccurrence(occurrence)) throw new Error("Occurrence must be 1–8.");
      options.occurrence = occurrence;
    } else if (flag === "--layout") {
      if (!isLayoutMode(value)) throw new Error("Layout must be navigation or item.");
      options.layout = value;
    } else options.repeats = Number(value);
  } else throw new Error(`Unknown argument: ${flag}`);
}
// Validate arguments before opening an output file. Exclusive creation avoids overwrites.
const plan = getRunPlan(options);
const first = plan.next();
const fd = output ? fs.openSync(output, "wx") : 1;
let count = 0;
try {
  if (!first.done) { fs.writeSync(fd, JSON.stringify(first.value) + "\n"); count++; }
  for (const row of plan) { fs.writeSync(fd, JSON.stringify(row) + "\n"); count++; }
} finally {
  if (output) fs.closeSync(fd);
}
process.stderr.write(`Exported ${count.toLocaleString("en-US")} planned executions per model.\n`);
