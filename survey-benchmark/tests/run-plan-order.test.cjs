/* eslint-disable @typescript-eslint/no-require-imports */
require("../scripts/register-typescript.cjs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { getRunPlan } = require("../lib/benchmark/runPlan.ts");

test("AC plan finishes navigation across all samples before item at each occurrence", () => {
  const groups = [];
  const ids = new Set();
  let previous;
  for (const row of getRunPlan({ repeats: 1, includeBaselines: false })) {
    const group = `${row.occurrence}:${row.layout}`;
    if (groups.at(-1) !== group) {
      assert.ok(!groups.includes(group), `Returned to completed group ${group}`);
      groups.push(group);
      previous = undefined;
    }
    assert.equal(row.suiteVersion, "v1");
    assert.equal(row.condition, "attention-horizon");
    assert.ok(!ids.has(row.planId));
    ids.add(row.planId);
    if (previous) {
      assert.ok(row.sampleId >= previous.sampleId);
      if (row.sampleId === previous.sampleId) assert.ok(row.orderId > previous.orderId);
    }
    previous = row;
  }
  assert.deepEqual(groups, ["1:navigation", ...Array.from({ length: 7 }, (_, i) =>
    [`${i + 2}:navigation`, `${i + 2}:item`]).flat()]);
  assert.equal(ids.size, 44778);
});
