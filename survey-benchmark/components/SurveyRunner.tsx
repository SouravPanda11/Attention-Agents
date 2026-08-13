"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { summarizeAnswers } from "@/lib/benchmark/answers";
import type { AnswerValue, SurveyAnswers, Workflow } from "@/lib/benchmark/schema";
import { QuestionRenderer } from "@/components/questions/QuestionRenderer";
import { SurveyWelcome } from "@/components/SurveyWelcome";

type StoredProgress = {
  runId: string;
  answers: SurveyAnswers;
  pageIndex: number;
  started: boolean;
  completed: boolean;
};

function createRunId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `run-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function postJson(url: string, body: unknown) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

export function SurveyRunner({ workflow }: { workflow: Workflow }) {
  const storageKey = `survey-benchmark:${workflow.id}`;
  const [runId, setRunId] = useState("");
  const [answers, setAnswers] = useState<SurveyAnswers>({});
  const [pageIndex, setPageIndex] = useState(0);
  const [restored, setRestored] = useState(false);
  const [started, setStarted] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const pageStartedAt = useRef(0);
  const welcomeViewedAt = useRef(0);
  const loggedWelcome = useRef<string | null>(null);
  const loggedPage = useRef<string | null>(null);

  const currentPage = workflow.pages[pageIndex];
  const allQuestions = useMemo(() => workflow.pages.flatMap((page) => page.questions), [workflow.pages]);
  const questionOffset = useMemo(
    () => workflow.pages.slice(0, pageIndex).reduce((sum, page) => sum + page.questions.length, 0),
    [pageIndex, workflow.pages]
  );
  const responseSummary = useMemo(() => summarizeAnswers(allQuestions, answers), [allQuestions, answers]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      let stored: StoredProgress | null = null;
      try {
        const raw = sessionStorage.getItem(storageKey);
        stored = raw ? (JSON.parse(raw) as StoredProgress) : null;
      } catch {
        stored = null;
      }

      const nextRunId = stored?.runId || createRunId();
      setRunId(nextRunId);
      setAnswers(stored?.answers ?? {});
      setPageIndex(Math.min(Math.max(stored?.pageIndex ?? 0, 0), workflow.pageCount - 1));
      setStarted(
        Boolean(
          stored?.started ||
            stored?.completed ||
            (stored?.pageIndex ?? 0) > 0 ||
            Object.keys(stored?.answers ?? {}).length > 0
        )
      );
      setCompleted(Boolean(stored?.completed));
      setRestored(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [storageKey, workflow.pageCount]);

  useEffect(() => {
    if (!restored || !runId) return;
    const progress: StoredProgress = { runId, answers, pageIndex, started, completed };
    sessionStorage.setItem(storageKey, JSON.stringify(progress));
  }, [answers, completed, pageIndex, restored, runId, started, storageKey]);

  useEffect(() => {
    if (!restored || !runId || started || completed) return;
    if (loggedWelcome.current === runId) return;
    loggedWelcome.current = runId;
    welcomeViewedAt.current = Date.now();
    void postJson("/api/events", {
      runId,
      workflowId: workflow.id,
      eventType: "welcome_viewed",
      pageIndex: null,
      payload: { hasWelcomePage: true },
    }).catch(() => undefined);
  }, [completed, restored, runId, started, workflow.id]);

  useEffect(() => {
    if (!restored || !runId || !started || completed) return;
    const pageKey = `${runId}:${pageIndex}`;
    if (loggedPage.current === pageKey) return;
    loggedPage.current = pageKey;
    pageStartedAt.current = Date.now();
    void postJson("/api/events", {
      runId,
      workflowId: workflow.id,
      eventType: "page_viewed",
      pageIndex,
      payload: {
        pageCount: workflow.pageCount,
        questionIds: workflow.pages[pageIndex].questions.map((question) => question.id),
      },
    }).catch(() => undefined);
  }, [completed, pageIndex, restored, runId, started, workflow]);

  function startSurvey() {
    void postJson("/api/events", {
      runId,
      workflowId: workflow.id,
      eventType: "workflow_started",
      pageIndex: null,
      payload: {
        welcomeElapsedMs: Math.max(0, Date.now() - welcomeViewedAt.current),
        questionPageCount: workflow.pageCount,
        questionCount: workflow.questionCount,
      },
    }).catch(() => undefined);
    setStarted(true);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function setAnswer(questionId: string, value: AnswerValue) {
    setAnswers((current) => ({ ...current, [questionId]: value }));
  }

  function leavePage(direction: "previous" | "next" | "submit") {
    if (!currentPage) return;
    const pageSummary = summarizeAnswers(currentPage.questions, answers);
    void postJson("/api/events", {
      runId,
      workflowId: workflow.id,
      eventType: "page_left",
      pageIndex,
      payload: {
        direction,
        elapsedMs: Date.now() - pageStartedAt.current,
        questionIds: currentPage.questions.map((question) => question.id),
        attemptedCount: pageSummary.attemptedCount,
        validCount: pageSummary.validCount,
        invalidCount: pageSummary.invalidCount,
        skippedCount: pageSummary.skippedCount,
        skippedQuestionIds: pageSummary.skippedQuestionIds,
        invalidQuestionIds: pageSummary.invalidQuestionIds,
      },
    }).catch(() => undefined);
  }

  function previousPage() {
    if (pageIndex === 0 || submitting) return;
    leavePage("previous");
    setSubmitError("");
    setPageIndex((current) => current - 1);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  async function advance() {
    if (!currentPage || submitting) return;
    setSubmitError("");

    if (pageIndex < workflow.pageCount - 1) {
      leavePage("next");
      setPageIndex((current) => current + 1);
      window.scrollTo({ top: 0, behavior: "instant" });
      return;
    }

    leavePage("submit");
    setSubmitting(true);
    try {
      await postJson("/api/submissions", {
        runId,
        workflowId: workflow.id,
        profile: workflow.profile,
        occurrence: workflow.occurrence,
        layout: workflow.layout,
        orderId: workflow.orderId,
        contentVersion: workflow.contentVersion,
        attentionCheckContentVersion: workflow.attentionCheckContentVersion,
        orderedQuestionIds: workflow.orderedQuestionIds,
        answers,
      });
      setCompleted(true);
      window.scrollTo({ top: 0, behavior: "instant" });
    } catch {
      setSubmitError("The response could not be saved. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  function restart() {
    sessionStorage.removeItem(storageKey);
    window.location.reload();
  }

  if (!restored || !runId) {
    return (
      <main className="survey-shell">
        <p>Loading workflow…</p>
      </main>
    );
  }

  if (completed) {
    return (
      <main className="survey-shell">
        <section className="completion-card">
          <p className="eyebrow">Submission recorded</p>
          <h1>Workflow complete</h1>
          <p>
            You submitted {responseSummary.attemptedCount} response{responseSummary.attemptedCount === 1 ? "" : "s"}
            {responseSummary.skippedCount > 0
              ? ` and skipped ${responseSummary.skippedCount} question${responseSummary.skippedCount === 1 ? "" : "s"}`
              : ""}
            .
          </p>
          <p className="run-reference">Run ID: {runId}</p>
          <div className="button-row">
            <Link className="secondary-button" href="/">
              Return to launcher
            </Link>
            <button className="primary-button" type="button" onClick={restart}>
              Start a fresh run
            </button>
          </div>
        </section>
      </main>
    );
  }

  if (!started) {
    return <SurveyWelcome workflowId={workflow.id} onStart={startSurvey} />;
  }

  return (
    <main className="survey-shell" data-workflow-id={workflow.id} data-profile={workflow.profile}>
      <header className="survey-header">
        <div>
          <p className="eyebrow">Survey Benchmark · {workflow.suiteVersion} · Standard web</p>
          <h1>Survey workflow</h1>
          <p className="workflow-id">{workflow.id}</p>
        </div>
        <div className="progress-summary" aria-label="Workflow progress">
          <strong>
            Page {pageIndex + 1} of {workflow.pageCount}
          </strong>
          <span>
            {responseSummary.attemptedCount} of {workflow.renderedQuestionCount} responses entered
          </span>
        </div>
      </header>

      <section className="instruction-card">
        <h2>Instructions</h2>
        <p>Questions may be left unanswered. Use Previous or Next to move between pages and revise responses.</p>
        <p>
          This is the <strong>{workflow.layout === "item" ? "item-heavy" : "navigation-heavy"}</strong>{" "}
          layout with {workflow.renderedQuestionCount} displayed items.
        </p>
      </section>

      <div className="question-list">
        {currentPage.questions.map((question, index) => (
          <QuestionRenderer
            key={question.id}
            question={question}
            number={questionOffset + index + 1}
            value={answers[question.id]}
            onChange={(value) => setAnswer(question.id, value)}
          />
        ))}
      </div>

      {submitError ? (
        <div className="error-summary" role="alert">
          {submitError}
        </div>
      ) : null}

      <footer className="survey-footer">
        <span>
          Items {questionOffset + 1}–{questionOffset + currentPage.questions.length} of {workflow.renderedQuestionCount}
        </span>
        <div className="page-navigation-actions">
          {pageIndex > 0 ? (
            <button
              type="button"
              className="secondary-button"
              data-action="previous-page"
              disabled={submitting}
              onClick={previousPage}
            >
              Previous page
            </button>
          ) : null}
          <button
            type="button"
            className="primary-button"
            data-action={pageIndex === workflow.pageCount - 1 ? "submit-survey" : "next-page"}
            disabled={submitting}
            onClick={advance}
          >
            {submitting ? "Saving…" : pageIndex === workflow.pageCount - 1 ? "Submit survey" : "Next page"}
          </button>
        </div>
      </footer>
    </main>
  );
}
