import { NextResponse } from "next/server";
import { classifyAnswer, summarizeAnswers } from "@/lib/benchmark/answers";
import { summarizeAttentionChecks } from "@/lib/benchmark/attentionChecks";
import { buildWorkflow } from "@/lib/benchmark/buildWorkflow";
import { buildThemeWorkflow } from "@/lib/benchmark/themeWorkflow";
import { isThemeId } from "@/lib/benchmark/questions/themes/types";
import {
  isLayoutMode,
  isOccurrence,
  isOrderId,
  isPresentationProfile,
} from "@/lib/benchmark/schema";
import { getDatabase } from "@/lib/db";
import { getOrCreateSessionId } from "@/lib/session";

export const runtime = "nodejs";

type SubmissionBody = {
  runId?: unknown;
  workflowId?: unknown;
  themeId?: unknown;
  profile?: unknown;
  occurrence?: unknown;
  layout?: unknown;
  orderId?: unknown;
  contentVersion?: unknown;
  attentionCheckContentVersion?: unknown;
  orderedQuestionIds?: unknown;
  answers?: unknown;
};

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as SubmissionBody | null;
  if (
    !body ||
    typeof body.runId !== "string" ||
    body.runId.length > 128 ||
    typeof body.workflowId !== "string" ||
    typeof body.profile !== "string" ||
    typeof body.occurrence !== "number" ||
    typeof body.layout !== "string" ||
    typeof body.orderId !== "string" ||
    typeof body.contentVersion !== "number" ||
    typeof body.attentionCheckContentVersion !== "number" ||
    !Array.isArray(body.orderedQuestionIds) ||
    !body.answers ||
    typeof body.answers !== "object" ||
    Array.isArray(body.answers) ||
    !isPresentationProfile(body.profile) ||
    !isOccurrence(body.occurrence) ||
    !isLayoutMode(body.layout) ||
    !isOrderId(body.orderId)
  ) {
    return NextResponse.json({ ok: false, error: "invalid_submission" }, { status: 400 });
  }

  if (body.themeId !== undefined && (!isThemeId(body.themeId) || body.occurrence !== 1 || body.layout !== "item")) {
    return NextResponse.json({ ok: false, error: "invalid_theme_workflow" }, { status: 400 });
  }
  const workflow = isThemeId(body.themeId)
    ? buildThemeWorkflow(body.themeId, body.orderId)
    : buildWorkflow(body.profile, body.occurrence, body.layout, body.orderId);
  const submittedIds = body.orderedQuestionIds.map(String);
  if (
    body.workflowId !== workflow.id ||
    body.contentVersion !== workflow.contentVersion ||
    body.attentionCheckContentVersion !== workflow.attentionCheckContentVersion ||
    submittedIds.join("|") !== workflow.orderedQuestionIds.join("|")
  ) {
    return NextResponse.json({ ok: false, error: "workflow_mismatch" }, { status: 400 });
  }

  const answers = body.answers as Record<string, unknown>;
  const questions = workflow.pages.flatMap((page) => page.questions);
  const expectedQuestionIds = new Set(workflow.orderedQuestionIds);
  const unexpectedQuestionIds = Object.keys(answers).filter((questionId) => !expectedQuestionIds.has(questionId));
  if (unexpectedQuestionIds.length > 0) {
    return NextResponse.json({ ok: false, error: "unexpected_answers", unexpectedQuestionIds }, { status: 400 });
  }
  const summary = summarizeAnswers(questions, answers);
  const attentionSummary = workflow.themeId
    ? { checkCount: 0, scoredCount: 0, unscoredCount: 0, attemptedCount: 0, passCount: 0, failCount: 0, skippedCount: 0, results: [] }
    : summarizeAttentionChecks(workflow.occurrence, workflow.orderId, answers);

  const sessionId = await getOrCreateSessionId();
  const database = getDatabase();
  database
    .prepare(
      `INSERT INTO submissions (
        ts, session_id, run_id, workflow_id, suite_version, profile, occurrence, layout,
        order_id, content_version, attention_check_content_version, ordered_question_ids, answers, answer_count,
        valid_answer_count, invalid_answer_count, skipped_question_count,
        attention_check_count, attention_check_scored_count, attention_check_unscored_count,
        attention_check_attempted_count, attention_check_pass_count,
        attention_check_fail_count, attention_check_skipped_count, attention_check_results, theme_id
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(run_id, workflow_id) DO UPDATE SET
        ts = excluded.ts,
        content_version = excluded.content_version,
        attention_check_content_version = excluded.attention_check_content_version,
        ordered_question_ids = excluded.ordered_question_ids,
        answers = excluded.answers,
        answer_count = excluded.answer_count,
        valid_answer_count = excluded.valid_answer_count,
        invalid_answer_count = excluded.invalid_answer_count,
        skipped_question_count = excluded.skipped_question_count,
        attention_check_count = excluded.attention_check_count,
        attention_check_scored_count = excluded.attention_check_scored_count,
        attention_check_unscored_count = excluded.attention_check_unscored_count,
        attention_check_attempted_count = excluded.attention_check_attempted_count,
        attention_check_pass_count = excluded.attention_check_pass_count,
        attention_check_fail_count = excluded.attention_check_fail_count,
        attention_check_skipped_count = excluded.attention_check_skipped_count,
        attention_check_results = excluded.attention_check_results,
        theme_id = excluded.theme_id`
    )
    .run(
      new Date().toISOString(),
      sessionId,
      body.runId,
      workflow.id,
      workflow.suiteVersion,
      workflow.profile,
      workflow.occurrence,
      workflow.layout,
      workflow.orderId,
      workflow.contentVersion,
      workflow.attentionCheckContentVersion,
      JSON.stringify(workflow.orderedQuestionIds),
      JSON.stringify(answers),
      summary.attemptedCount,
      summary.validCount,
      summary.invalidCount,
      summary.skippedCount,
      attentionSummary.checkCount,
      attentionSummary.scoredCount,
      attentionSummary.unscoredCount,
      attentionSummary.attemptedCount,
      attentionSummary.passCount,
      attentionSummary.failCount,
      attentionSummary.skippedCount,
      JSON.stringify(attentionSummary.results),
      workflow.themeId ?? null
    );

  database
    .prepare(
      `INSERT INTO events (ts, session_id, run_id, workflow_id, event_type, page_index, payload)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      new Date().toISOString(),
      sessionId,
      body.runId,
      workflow.id,
      "workflow_submitted",
      workflow.pageCount - 1,
      JSON.stringify({
        attemptedCount: summary.attemptedCount,
        validCount: summary.validCount,
        invalidCount: summary.invalidCount,
        skippedCount: summary.skippedCount,
        skippedQuestionIds: summary.skippedQuestionIds,
        invalidQuestionIds: summary.invalidQuestionIds,
        attentionCheckCount: attentionSummary.checkCount,
        attentionCheckScoredCount: attentionSummary.scoredCount,
        attentionCheckUnscoredCount: attentionSummary.unscoredCount,
        attentionCheckAttemptedCount: attentionSummary.attemptedCount,
        attentionCheckPassCount: attentionSummary.passCount,
        attentionCheckFailCount: attentionSummary.failCount,
        attentionCheckSkippedCount: attentionSummary.skippedCount,
      })
    );

  return NextResponse.json({
    ok: true,
    accepted: true,
    attemptedCount: summary.attemptedCount,
    validCount: summary.validCount,
    invalidCount: summary.invalidCount,
    skippedCount: summary.skippedCount,
    themeId: workflow.themeId ?? null,
    attentionCheckCount: workflow.attentionCheckCount,
    questionResults: questions.map((question) => ({
      questionId: question.id,
      kind: question.kind,
      status: classifyAnswer(question, answers[question.id]),
    })),
  });
}
