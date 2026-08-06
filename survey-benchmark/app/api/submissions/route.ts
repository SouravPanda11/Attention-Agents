import { NextResponse } from "next/server";
import { isAnswerValid } from "@/lib/benchmark/answers";
import { buildWorkflow } from "@/lib/benchmark/buildWorkflow";
import {
  isLayoutMode,
  isOccurrence,
  isOrderId,
  isPresentationProfile,
  type SurveyAnswers,
} from "@/lib/benchmark/schema";
import { getDatabase } from "@/lib/db";
import { getOrCreateSessionId } from "@/lib/session";

export const runtime = "nodejs";

type SubmissionBody = {
  runId?: unknown;
  workflowId?: unknown;
  profile?: unknown;
  occurrence?: unknown;
  layout?: unknown;
  orderId?: unknown;
  contentVersion?: unknown;
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
    !Array.isArray(body.orderedQuestionIds) ||
    !body.answers ||
    typeof body.answers !== "object" ||
    !isPresentationProfile(body.profile) ||
    !isOccurrence(body.occurrence) ||
    !isLayoutMode(body.layout) ||
    !isOrderId(body.orderId)
  ) {
    return NextResponse.json({ ok: false, error: "invalid_submission" }, { status: 400 });
  }

  const workflow = buildWorkflow(body.profile, body.occurrence, body.layout, body.orderId);
  const submittedIds = body.orderedQuestionIds.map(String);
  if (
    body.workflowId !== workflow.id ||
    body.contentVersion !== workflow.contentVersion ||
    submittedIds.join("|") !== workflow.orderedQuestionIds.join("|")
  ) {
    return NextResponse.json({ ok: false, error: "workflow_mismatch" }, { status: 400 });
  }

  const answers = body.answers as SurveyAnswers;
  const questions = workflow.pages.flatMap((page) => page.questions);
  const invalidQuestionIds = questions
    .filter((question) => !isAnswerValid(question, answers[question.id]))
    .map((question) => question.id);
  if (invalidQuestionIds.length > 0) {
    return NextResponse.json(
      { ok: false, error: "incomplete_answers", invalidQuestionIds },
      { status: 400 }
    );
  }

  const sessionId = await getOrCreateSessionId();
  const database = getDatabase();
  database
    .prepare(
      `INSERT INTO submissions (
        ts, session_id, run_id, workflow_id, suite_version, profile, occurrence, layout,
        order_id, content_version, ordered_question_ids, answers, answer_count
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(run_id, workflow_id) DO UPDATE SET
        ts = excluded.ts,
        answers = excluded.answers,
        answer_count = excluded.answer_count`
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
      JSON.stringify(workflow.orderedQuestionIds),
      JSON.stringify(answers),
      questions.length
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
      JSON.stringify({ answerCount: questions.length })
    );

  return NextResponse.json({ ok: true, accepted: true, answerCount: questions.length });
}
