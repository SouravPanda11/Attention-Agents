/* eslint-disable @typescript-eslint/no-require-imports */
require("../scripts/register-typescript.cjs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { getSurveySamples, getSurveySample, getSamplingSummary } = require("../lib/benchmark/sampling.ts");
const { buildWorkflow, getWorkflowManifest } = require("../lib/benchmark/buildWorkflow.ts");
const { getAttentionCheckInstances, summarizeAttentionChecks } = require("../lib/benchmark/attentionChecks.ts");
const { getThemeWorkflowManifest } = require("../lib/benchmark/themeWorkflow.ts");
const { getRunPlan } = require("../lib/benchmark/runPlan.ts");
const { ORDER_IDS, QUESTION_KINDS } = require("../lib/benchmark/schema.ts");

test("complete sample enumeration follows the agreed counts, theme selections and repetition limits", () => {
  assert.deepEqual(getSamplingSummary().map((row) => row.sampleCount), [224, 1960, 1568, 70, 1568, 1960, 224, 1]);
  const sampleIds = new Set();
  for (let occurrence = 1; occurrence <= 8; occurrence++) {
    const signatures = new Set();
    for (const sample of getSurveySamples(occurrence)) {
      assert.equal(getSurveySample(sample.id), sample);
      assert.equal(new Set(sample.themeIds).size, occurrence);
      assert.equal(sample.rotatingAttentionCheckIds.length, 2 * occurrence);
      const frequencies = new Map();
      for (const id of sample.rotatingAttentionCheckIds) frequencies.set(id, (frequencies.get(id) ?? 0) + 1);
      assert.equal(frequencies.size, Math.min(8, 2 * occurrence));
      assert.ok([...frequencies.values()].every((count) => count <= (occurrence <= 4 ? 1 : 2)));
      assert.equal([...frequencies.values()].filter((count) => count === 2).length, Math.max(0, 2 * occurrence - 8));
      signatures.add(JSON.stringify([sample.themeIds, sample.rotatingAttentionCheckIds]));
      sampleIds.add(sample.id);
    }
    assert.equal(signatures.size, getSurveySamples(occurrence).length);
  }
  assert.equal(sampleIds.size, 7575);
  for (const bad of ["o0-s0001", "o9-s0001", "o1-s0000", "o1-s0225", "o8-s0002", "o1-s1", "__proto__"])
    assert.equal(getSurveySample(bad), undefined);
});

test("all 44,778 variants preserve content, navigation quotas and exact item/navigation parity", () => {
  let checked = 0;
  const opaqueIdentities = new Map();
  for (let occurrence = 1; occurrence <= 8; occurrence++) {
    for (const sample of getSurveySamples(occurrence)) {
      const variants = ORDER_IDS.map((order) => buildWorkflow(sample.id, order));
      const content = [...variants[0].orderedQuestionIds].sort();
      assert.equal(new Set(variants.map((variant) => variant.orderedQuestionIds.join("|"))).size, 3);
      for (const workflow of variants) {
        assert.deepEqual([...workflow.orderedQuestionIds].sort(), content);
        assert.equal(workflow.pageCount, occurrence);
        assert.equal(workflow.questionCount, 13 * occurrence + 1);
        assert.equal(workflow.attentionCheckCount, 2 * occurrence + 1);
        assert.equal(new Set(workflow.orderedQuestionIds).size, workflow.questionCount);
        const checks = getAttentionCheckInstances(sample, workflow.orderId);
        const checkIds = new Set(checks.map((check) => check.question.id));
        const fixed = checks.find((check) => check.role === "fixed-penultimate");
        assert.equal(workflow.pages.at(-1).questions.at(-2).id, fixed.question.id);
        for (const check of checks) {
          const identity = `${sample.id}:${check.privateId}:${check.copy}`;
          assert.ok(!opaqueIdentities.has(check.question.id) || opaqueIdentities.get(check.question.id) === identity);
          opaqueIdentities.set(check.question.id, identity);
        }
        const normal = workflow.pages.flatMap((page) => page.questions).filter((q) => !checkIds.has(q.id));
        assert.deepEqual([...new Set(normal.map((q) => q.id.split("-")[1]))].sort(), [...sample.themeIds].sort());
        for (const theme of sample.themeIds) assert.equal(normal.filter((q) => q.id.startsWith(`main-${theme}-`)).length, 11);
        for (const kind of QUESTION_KINDS) assert.equal(normal.filter((q) => q.kind === kind).length, occurrence);
        for (const [index, page] of workflow.pages.entries()) {
          assert.equal(page.questions.length, index === occurrence - 1 ? 14 : 13);
          assert.equal(page.questions.filter((q) => !checkIds.has(q.id)).length, 11);
          assert.ok(page.questions.every((q) => q.block === index + 1));
          const ordinary = checks.filter((check) => check.role === "rotating" && check.question.block === index + 1);
          assert.equal(ordinary.length, 2);
          assert.notEqual(ordinary[0].privateId, ordinary[1].privateId);
          for (const q of page.questions) {
            assert.equal("expectedAnswer" in q, false);
            assert.equal("privateId" in q, false);
          }
        }
        if (occurrence > 1) {
          const item = buildWorkflow(sample.id, workflow.orderId, "item");
          assert.equal(item.pageCount, 1);
          assert.equal(item.layout, "item");
          assert.equal(item.questionCount, workflow.questionCount);
          assert.equal(item.attentionCheckCount, workflow.attentionCheckCount);
          assert.deepEqual(item.questionCountsPerPage, [workflow.questionCount]);
          assert.deepEqual(item.orderedQuestionIds, workflow.orderedQuestionIds);
          assert.deepEqual(item.pages[0].questions, workflow.pages.flatMap((page) => page.questions));
          assert.equal(item.orderSeed, workflow.orderSeed);
          assert.equal(item.sampleId, workflow.sampleId);
          assert.equal(item.pages[0].questions.at(-2).id, fixed.question.id);
          assert.notEqual(item.id, workflow.id);
          checked++;
        }
        checked++;
      }
    }
  }
  assert.equal(checked, 44778);
  assert.throws(() => buildWorkflow("o1-s0001", "order01", "item"), /Unsupported layout/);
});

test("layouts reproduce exactly, mix themes, and move questions across pages", () => {
  for (const sampleId of ["o1-s0001", "o2-s1960", "o5-s0100", "o8-s0001"]) {
    for (const order of ORDER_IDS) assert.deepEqual(buildWorkflow(sampleId, order), buildWorkflow(sampleId, order));
  }
  const variants = ORDER_IDS.map((order) => buildWorkflow("o8-s0001", order));
  for (const workflow of variants) assert.ok(workflow.pages.every((page) =>
    new Set(page.questions.filter((q) => q.id.startsWith("main-")).map((q) => q.id.split("-")[1])).size > 1));
  assert.notDeepEqual(variants[0].pages[0].questions.map((q) => q.id).sort(), variants[1].pages[0].questions.map((q) => q.id).sort());
});

test("repeated AC instances score independently and the fixed placeholder remains unscored", () => {
  const sample = getSurveySample("o8-s0001");
  const instances = getAttentionCheckInstances(sample, "order01");
  const answers = Object.fromEntries(instances.filter((check) => check.role === "rotating").map((check) => [check.question.id, check.expectedAnswer]));
  const first = instances.find((check) => check.role === "rotating");
  delete answers[first.question.id];
  const summary = summarizeAttentionChecks(sample, "order01", answers);
  assert.equal(summary.checkCount, 17);
  assert.equal(summary.scoredCount, 16);
  assert.equal(summary.unscoredCount, 1);
  assert.equal(summary.passCount, 15);
  assert.equal(summary.failCount, 1);
  assert.equal(summary.skippedCount, 2);
});

test("24 baseline layouts and paginated manifest stay separate from the AC suite", () => {
  const baselines = getThemeWorkflowManifest();
  assert.equal(baselines.length, 24);
  for (const baseline of baselines) {
    assert.equal(baseline.questionCount, 11);
    assert.equal(baseline.attentionCheckCount, 0);
    assert.equal(baseline.condition, "theme-baseline");
  }
  const first = getWorkflowManifest({ limit: 2 });
  assert.equal(first.pagination.total, 44778);
  assert.equal(first.pagination.nextOffset, 2);
  assert.equal(first.workflows.length, 2);
  const selected = getWorkflowManifest({ sampleId: "o8-s0001" });
  assert.equal(selected.workflows.length, 6);
  assert.equal(selected.pagination.nextOffset, null);
  assert.equal(getWorkflowManifest({ occurrence: 1 }).pagination.total, 672);
  assert.equal(getWorkflowManifest({ layout: "navigation" }).pagination.total, 22725);
  assert.equal(getWorkflowManifest({ layout: "item" }).pagination.total, 22053);
  assert.equal(getWorkflowManifest({ occurrence: 1, layout: "item" }).pagination.total, 0);
  const acrossBoundary = getWorkflowManifest({ sampleId: "o8-s0001", offset: 2, limit: 2 });
  assert.deepEqual(acrossBoundary.workflows.map((w) => [w.layout, w.orderId]), [["navigation", "order03"], ["item", "order01"]]);
  assert.equal(getWorkflowManifest({ offset: 44778 }).workflows.length, 0);
});

test("execution plans distinguish samples, three layouts and three repetitions", () => {
  const plan = [...getRunPlan()];
  assert.equal(plan.length, 134406);
  assert.equal(new Set(plan.map((row) => row.planId)).size, 134406);
  assert.equal(plan.filter((row) => row.condition === "theme-baseline").length, 72);
  assert.equal([...getRunPlan({ occurrence: 8, includeBaselines: false })].length, 18);
  assert.equal([...getRunPlan({ repeats: 1 })].length, 44802);
  assert.equal([...getRunPlan({ occurrence: 8, layout: "item", includeBaselines: false })].length, 9);
  assert.equal([...getRunPlan({ occurrence: 1, layout: "item", includeBaselines: false })].length, 0);
  assert.ok(plan.filter((row) => row.condition === "attention-horizon" && row.layout === "item").every((row) => row.pageCount === 1));
  assert.throws(() => [...getRunPlan({ repeats: 0 })]);
});
