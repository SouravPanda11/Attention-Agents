/* eslint-disable @typescript-eslint/no-require-imports */
require("../scripts/register-typescript.cjs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
process.env.SURVEY_DB_PATH = ":memory:";
require("../lib/session.ts").getOrCreateSessionId = async () => "test-session";
const { POST } = require("../app/api/submissions/route.ts");
const { GET } = require("../app/api/manifest/route.ts");
const { getDatabase } = require("../lib/db.ts");
const { buildWorkflow } = require("../lib/benchmark/buildWorkflow.ts");
const { buildThemeWorkflow } = require("../lib/benchmark/themeWorkflow.ts");
const { getAttentionCheckInstances } = require("../lib/benchmark/attentionChecks.ts");
const { getSurveySample } = require("../lib/benchmark/sampling.ts");

function payload(workflow, extra = {}) {
  return { runId: "api-test", workflowId: workflow.id, sampleId: workflow.sampleId,
    orderingVersion: workflow.orderingVersion, profile: workflow.profile,
    themeId: workflow.themeId, occurrence: workflow.occurrence, layout: workflow.layout,
    orderId: workflow.orderId, contentVersion: workflow.contentVersion,
    attentionCheckContentVersion: workflow.attentionCheckContentVersion,
    orderedQuestionIds: workflow.orderedQuestionIds, answers: {}, ...extra };
}
function submit(body) {
  return POST(new Request("http://localhost/api/submissions", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }));
}

test("manifest reports exact totals, follows pagination and rejects invalid filters", async () => {
  const response = await GET(new Request("http://localhost/api/manifest?occurrence=8&layout=item&limit=2"));
  const body = await response.json();
  assert.equal(body.contentSampleCount, 7575);
  assert.equal(body.totalContentSampleCount, 7583);
  assert.equal(body.workflowInstanceCount, 44778);
  assert.equal(body.totalWorkflowInstanceCount, 44802);
  assert.equal(body.defaultRunsPerModel, 134406);
  assert.equal(body.themeWorkflows.length, 24);
  assert.equal(body.pagination.total, 3);
  const next = await (await GET(new Request(`http://localhost${body.next}`))).json();
  assert.equal(next.workflows.length, 1);
  assert.equal(next.next, null);
  assert.equal(next.workflows[0].orderId, "order03");
  assert.equal(next.workflows[0].layout, "item");
  assert.equal(next.workflows[0].pageCount, 1);
  for (const query of ["layout=bad", "occurrence=9", "occurrence=1.5", "limit=0", "limit=301", "offset=-1", "order=bad", "sampleId=o8-s0002"]) {
    assert.equal((await GET(new Request(`http://localhost/api/manifest?${query}`))).status, 400);
  }
  assert.equal(JSON.stringify(body).includes("expectedAnswer"), false);
  assert.equal(JSON.stringify(body).includes("privateId"), false);
});

test("submissions persist sample identity, independent AC results and repeat metadata", async () => {
  const workflow = buildWorkflow("o8-s0001", "order02");
  const checks = getAttentionCheckInstances(getSurveySample(workflow.sampleId), workflow.orderId);
  const answered = checks.find((check) => check.role === "rotating");
  const data = payload(workflow, { repeatIndex: 2, answers: { [answered.question.id]: answered.expectedAnswer } });
  const response = await submit(data);
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.sampleId, "o8-s0001");
  assert.equal(result.questionResults.length, 105);
  assert.equal(result.substantiveSummary.skippedCount, 88);
  const saved = getDatabase().prepare("SELECT * FROM submissions WHERE run_id = ?").get("api-test");
  assert.equal(saved.sample_id, "o8-s0001");
  assert.equal(saved.suite_version, "v1");
  assert.equal(saved.repeat_index, 2);
  assert.equal(JSON.parse(saved.selected_theme_ids).length, 8);
  assert.equal(saved.attention_check_count, 17);
  assert.equal(saved.attention_check_scored_count, 16);
  assert.equal(saved.attention_check_unscored_count, 1);
  assert.equal(saved.attention_check_pass_count, 1);
  assert.equal(saved.attention_check_results.includes(answered.privateId), true);
  assert.equal(saved.order_seed, workflow.orderSeed);
  assert.equal((await submit(data)).status, 200);
  assert.equal(getDatabase().prepare("SELECT count(*) AS count FROM submissions WHERE run_id = ?").get("api-test").count, 1);
  assert.equal((await submit({ ...data, runId: "another-repeat", repeatIndex: 3 })).status, 200);
});

test("wrong sample, layout, version, question order and unexpected answers are rejected", async () => {
  const workflow = buildWorkflow("o1-s0001", "order01");
  for (const changes of [
    { sampleId: undefined }, { sampleId: "o1-s0002" }, { occurrence: 2 }, { layout: "item" },
    { workflowId: "v0-standard-o1-navigation-order01" }, { contentVersion: 99 }, { orderingVersion: 99 },
    { orderedQuestionIds: [...workflow.orderedQuestionIds].reverse() }, { answers: { unknown: "answer" } },
    { repeatIndex: 0 }, { runId: "" }, { themeId: "work" },
  ]) assert.equal((await submit(payload(workflow, changes))).status, 400, JSON.stringify(changes));
});

test("theme-only baseline submissions stay compatible and have no AC scores", async () => {
  const workflow = buildThemeWorkflow("work", "order01");
  const response = await submit(payload(workflow, { runId: "baseline-test", sampleId: undefined, orderingVersion: undefined }));
  assert.equal(response.status, 200);
  assert.equal((await response.json()).attentionCheckCount, 0);
  const saved = getDatabase().prepare("SELECT * FROM submissions WHERE run_id = ?").get("baseline-test");
  assert.equal(saved.sample_id, "theme-work");
  assert.equal(saved.theme_id, "work");
  assert.equal(saved.attention_check_count, 0);
  assert.equal(saved.skipped_question_count, 11);
});

test("item and navigation submissions stay separate and reject a swapped layout", async () => {
  const navigation = buildWorkflow("o2-s0001", "order01");
  const item = buildWorkflow("o2-s0001", "order01", "item");
  assert.deepEqual(item.orderedQuestionIds, navigation.orderedQuestionIds);
  for (const workflow of [navigation, item]) {
    const response = await submit(payload(workflow, { runId: "matched-layout-test" }));
    assert.equal(response.status, 200);
    assert.equal((await response.json()).questionResults.length, 27);
  }
  const rows = getDatabase().prepare("SELECT workflow_id, layout, sample_id, order_seed, ordered_question_ids FROM submissions WHERE run_id = ? ORDER BY layout").all("matched-layout-test");
  assert.deepEqual(rows.map((row) => row.layout), ["item", "navigation"]);
  assert.equal(rows[0].sample_id, rows[1].sample_id);
  assert.equal(rows[0].order_seed, rows[1].order_seed);
  assert.equal(rows[0].ordered_question_ids, rows[1].ordered_question_ids);
  assert.notEqual(rows[0].workflow_id, rows[1].workflow_id);
  const events = getDatabase().prepare("SELECT workflow_id, page_index FROM events WHERE run_id = ? ORDER BY workflow_id").all("matched-layout-test");
  assert.equal(events.find((event) => event.workflow_id === item.id).page_index, 0);
  assert.equal(events.find((event) => event.workflow_id === navigation.id).page_index, 1);
  assert.equal((await submit(payload(navigation, { layout: "item" }))).status, 400);
  assert.equal((await submit(payload(item, { layout: "navigation" }))).status, 400);
});
