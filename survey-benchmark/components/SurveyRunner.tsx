"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { isAnswerValid } from "@/lib/benchmark/answers";
import type { AnswerValue, SurveyAnswers, Workflow } from "@/lib/benchmark/schema";
import { QuestionRenderer } from "@/components/questions/QuestionRenderer";

type StoredProgress = {
  runId: string;
  answers: SurveyAnswers;
  pageIndex: number;
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
  const [invalidIds, setInvalidIds] = useState<string[]>([]);
  const [restored, setRestored] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const pageStartedAt = useRef(0);
  const loggedPage = useRef<string | null>(null);

  const currentPage = workflow.pages[pageIndex];
  const questionOffset = useMemo(
    () => workflow.pages.slice(0, pageIndex).reduce((sum, page) => sum + page.questions.length, 0),
    [pageIndex, workflow.pages]
  );
  const answeredCount = workflow.orderedQuestionIds.filter((questionId) => answers[questionId] !== undefined).length;

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
      setCompleted(Boolean(stored?.completed));
      setRestored(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [storageKey, workflow.pageCount]);

  useEffect(() => {
    if (!restored || !runId) return;
    const progress: StoredProgress = { runId, answers, pageIndex, completed };
    sessionStorage.setItem(storageKey, JSON.stringify(progress));
  }, [answers, completed, pageIndex, restored, runId, storageKey]);

  useEffect(() => {
    if (!restored || !runId || completed) return;
    const pageKey = `${runId}:${pageIndex}`;
    if (loggedPage.current === pageKey) return;
    loggedPage.current = pageKey;
    pageStartedAt.current = Date.now();
    void postJson("/api/events", {
      runId,
      workflowId: workflow.id,
      eventType: pageIndex === 0 ? "workflow_started" : "page_viewed",
      pageIndex,
      payload: {
        pageCount: workflow.pageCount,
        questionIds: workflow.pages[pageIndex].questions.map((question) => question.id),
      },
    }).catch(() => undefined);
  }, [completed, pageIndex, restored, runId, workflow]);

  function setAnswer(questionId: string, value: AnswerValue) {
    setAnswers((current) => ({ ...current, [questionId]: value }));
    setInvalidIds((current) => current.filter((id) => id !== questionId));
  }

  async function advance() {
    if (!currentPage || submitting) return;
    const invalid = currentPage.questions
      .filter((question) => !isAnswerValid(question, answers[question.id]))
      .map((question) => question.id);
    setInvalidIds(invalid);

    if (invalid.length > 0) {
      void postJson("/api/events", {
        runId,
        workflowId: workflow.id,
        eventType: "page_validation_failed",
        pageIndex,
        payload: { invalidQuestionIds: invalid },
      }).catch(() => undefined);
      document.querySelector<HTMLElement>(`[data-question-id="${invalid[0]}"]`)?.focus();
      document.querySelector<HTMLElement>(`[data-question-id="${invalid[0]}"]`)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
      return;
    }

    setSubmitError("");
    setSubmitting(true);
    try {
      await postJson("/api/events", {
        runId,
        workflowId: workflow.id,
        eventType: "page_completed",
        pageIndex,
        payload: {
          elapsedMs: Date.now() - pageStartedAt.current,
          questionIds: currentPage.questions.map((question) => question.id),
        },
      });

      if (pageIndex < workflow.pageCount - 1) {
        setPageIndex((current) => current + 1);
        setInvalidIds([]);
        window.scrollTo({ top: 0, behavior: "instant" });
      } else {
        await postJson("/api/submissions", {
          runId,
          workflowId: workflow.id,
          profile: workflow.profile,
          occurrence: workflow.occurrence,
          layout: workflow.layout,
          orderId: workflow.orderId,
          contentVersion: workflow.contentVersion,
          orderedQuestionIds: workflow.orderedQuestionIds,
          answers,
        });
        setCompleted(true);
        window.scrollTo({ top: 0, behavior: "instant" });
      }
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
          <p>You completed all {workflow.questionCount} questions in {workflow.id}.</p>
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
            {answeredCount} of {workflow.questionCount} questions answered
          </span>
        </div>
      </header>

      <section className="instruction-card">
        <h2>Instructions</h2>
        <p>Complete every question on this page, then use the button at the bottom to continue.</p>
        <p>
          This is the <strong>{workflow.layout === "item" ? "item-heavy" : "navigation-heavy"}</strong>{" "}
          layout with {workflow.questionCount} questions.
        </p>
      </section>

      <div className="question-list">
        {currentPage.questions.map((question, index) => (
          <QuestionRenderer
            key={question.id}
            question={question}
            number={questionOffset + index + 1}
            value={answers[question.id]}
            invalid={invalidIds.includes(question.id)}
            onChange={(value) => setAnswer(question.id, value)}
          />
        ))}
      </div>

      {invalidIds.length > 0 ? (
        <div className="error-summary" role="alert">
          Complete the {invalidIds.length} highlighted question{invalidIds.length === 1 ? "" : "s"} before continuing.
        </div>
      ) : null}
      {submitError ? (
        <div className="error-summary" role="alert">
          {submitError}
        </div>
      ) : null}

      <footer className="survey-footer">
        <span>
          Questions {questionOffset + 1}–{questionOffset + currentPage.questions.length} of {workflow.questionCount}
        </span>
        <button
          type="button"
          className="primary-button"
          data-action={pageIndex === workflow.pageCount - 1 ? "submit-survey" : "next-page"}
          disabled={submitting}
          onClick={advance}
        >
          {submitting ? "Saving…" : pageIndex === workflow.pageCount - 1 ? "Submit survey" : "Next page"}
        </button>
      </footer>
    </main>
  );
}
