"use client";

import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { QuestionRenderer, type Question } from "@/app/survey/questionTypes";
import type { TextAnswerValue } from "@/lib/surveys/types";

type TextPayload = {
  session_id: string;
  text: {
    questions: Question[];
    attention_insert_after: number;
    attention: {
      id: string;
      label: string;
      options: { value: string; label: string }[];
    };
  };
};

type TextState = Record<string, TextAnswerValue>;

async function logEvent(eventType: string, payload: Record<string, unknown>) {
  await fetch("/api/log", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ event_type: eventType, payload }),
  });
}

function fieldStyle(): CSSProperties {
  return {
    width: "100%",
    padding: 10,
    borderRadius: 8,
    border: "1px solid #cccccc",
    background: "#fff",
  };
}

// Toggle required-enforcement mode by uncommenting exactly one line.
const ENFORCE_REQUIRED_TEXT_PAGE = true;
// const ENFORCE_REQUIRED_TEXT_PAGE = false;

function isAnswered(value: TextAnswerValue): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (typeof value === "number") return Number.isFinite(value);
  return false;
}

export default function TextSurveyPage() {
  const [payload, setPayload] = useState<TextPayload | null>(null);
  const [answers, setAnswers] = useState<TextState>({});
  const [attentionValue, setAttentionValue] = useState("");
  const [validationError, setValidationError] = useState("");

  useEffect(() => {
    (async () => {
      const res = await fetch("/api/survey");
      const json = (await res.json()) as TextPayload;
      setPayload(json);
      await logEvent("text_page_view", { page: "/survey/text" });
    })();
  }, []);

  const splitIndex = useMemo(() => payload?.text.attention_insert_after ?? 4, [payload]);

  if (!payload) {
    return (
      <main style={{ padding: 40 }}>
        <p>Loading...</p>
      </main>
    );
  }

  const firstHalf = payload.text.questions.slice(0, splitIndex);
  const secondHalf = payload.text.questions.slice(splitIndex);

  async function onNext() {
    if (!payload) return;
    if (ENFORCE_REQUIRED_TEXT_PAGE) {
      const missingQuestionIds = payload.text.questions
        .filter((q) => !isAnswered(answers[q.id]))
        .map((q) => q.id);
      const missingAttention = attentionValue.trim() === "";
      if (missingQuestionIds.length > 0 || missingAttention) {
        setValidationError("Please answer all required questions before moving to the next page.");
        await logEvent("text_required_block", {
          missing_question_ids: missingQuestionIds,
          missing_attention: missingAttention,
        });
        return;
      }
    }
    setValidationError("");
    const data = { answers, attention_value: attentionValue };
    sessionStorage.setItem("survey_text_answers", JSON.stringify(data));
    await logEvent("text_page_saved", {
      answer_count: Object.keys(answers).length,
      has_attention_value: Boolean(attentionValue),
    });
    window.location.href = "/survey/image";
  }

  return (
    <main style={{ padding: 40 }}>
      <div style={{ maxWidth: 760, margin: "0 auto" }}>
        <div style={{ border: "1px solid #e2e2e2", borderRadius: 12, padding: 28, background: "#fafafa" }}>
          <h2 style={{ margin: 0, fontSize: 22, textAlign: "center" }}>Olympics Survey</h2>

          <hr style={{ margin: "18px 0", borderColor: "#e8e8e8" }} />

          <div style={{ marginBottom: 18 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Instructions:</div>
            <p style={{ color: "#555", margin: "0 0 8px 0" }}>
              This page has 10 questions. Complete them all before moving to the next page.
            </p>
            <ul
              style={{
                margin: 0,
                paddingLeft: 20,
                color: "#444",
                lineHeight: 1.5,
                listStyleType: "disc",
                listStylePosition: "outside",
              }}
            >
              <li>Dropdown: choose exactly one option.</li>
              <li>Slider: select a value from 1 to 10.</li>
              <li>Text: write a short response in your own words.</li>
              <li>Submit this page and move to the next page.</li>
            </ul>
          </div>

          <hr style={{ margin: "18px 0", borderColor: "#e8e8e8" }} />

          {firstHalf.map((q) => (
            <QuestionRenderer
              key={q.id}
              q={q}
              value={answers[q.id]}
              required={ENFORCE_REQUIRED_TEXT_PAGE}
              setValue={(v) => {
                setValidationError("");
                setAnswers((prev) => ({ ...prev, [q.id]: v }));
              }}
            />
          ))}

          <div style={{ marginBottom: 16 }}>
            <label
              htmlFor={payload.text.attention.id}
              style={{ display: "block", fontWeight: 600, marginBottom: 6 }}
            >
              {payload.text.attention.label}
            </label>
            <select
              id={payload.text.attention.id}
              name={payload.text.attention.id}
              data-question-id={payload.text.attention.id}
              data-question-label={payload.text.attention.label}
              data-question-type="choice"
              required={ENFORCE_REQUIRED_TEXT_PAGE}
              style={fieldStyle()}
              value={attentionValue}
              onChange={(e) => {
                setValidationError("");
                setAttentionValue(e.target.value);
              }}
            >
              <option value="" disabled>
                Select...
              </option>
              {payload.text.attention.options.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {secondHalf.map((q) => (
            <QuestionRenderer
              key={q.id}
              q={q}
              value={answers[q.id]}
              required={ENFORCE_REQUIRED_TEXT_PAGE}
              setValue={(v) => {
                setValidationError("");
                setAnswers((prev) => ({ ...prev, [q.id]: v }));
              }}
            />
          ))}

          {validationError ? (
            <div style={{ margin: "0 0 12px 0", color: "#b00020", fontWeight: 600 }}>{validationError}</div>
          ) : null}

          <div style={{ textAlign: "center" }}>
            <button
              onClick={onNext}
              style={{
                padding: "10px 16px",
                borderRadius: 8,
                border: "0",
                background: "#2c6bed",
                color: "white",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
