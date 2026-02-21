"use client";

import React from "react";
import type { TextAnswerValue, TextQuestion } from "@/lib/surveys/types";

export type Question = TextQuestion;
export type QuestionValue = TextAnswerValue;

export function QuestionRenderer({
  q,
  value,
  setValue,
  required = false,
}: {
  q: Question;
  value: QuestionValue;
  setValue: (v: QuestionValue) => void;
  required?: boolean;
}) {
  const controlId = `q_${q.id}`;
  return (
    <div
      style={{ marginBottom: 16 }}
      data-question-id={q.id}
      data-question-label={q.label}
      data-question-type={q.type}
    >
      {q.type === "likert" ? (
        <div style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>{q.label}</div>
      ) : (
        <label htmlFor={controlId} style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>
          {q.label}
        </label>
      )}
      <QuestionField q={q} value={value} setValue={setValue} required={required} controlId={controlId} />
    </div>
  );
}

function baseFieldStyle(): React.CSSProperties {
  return {
    width: "100%",
    padding: 10,
    borderRadius: 8,
    border: "1px solid #cccccc",
    background: "#fff",
  };
}

function QuestionField({
  q,
  value,
  setValue,
  required = false,
  controlId,
}: {
  q: Question;
  value: QuestionValue;
  setValue: (v: QuestionValue) => void;
  required?: boolean;
  controlId: string;
}) {
  if (q.type === "number") {
    return (
      <input
        id={controlId}
        type="number"
        name={q.id}
        min={q.min}
        max={q.max}
        required={required}
        style={baseFieldStyle()}
        value={value ?? ""}
        onChange={(e) => setValue(e.target.value ? Number(e.target.value) : undefined)}
      />
    );
  }

  if (q.type === "choice") {
    return (
      <select
        id={controlId}
        name={q.id}
        required={required}
        style={baseFieldStyle()}
        value={value ?? ""}
        onChange={(e) => setValue(e.target.value)}
      >
        <option value="" disabled>
          Select...
        </option>
        {q.options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    );
  }

  if (q.type === "text") {
    return (
      <textarea
        id={controlId}
        name={q.id}
        rows={3}
        maxLength={q.maxLen}
        required={required}
        style={baseFieldStyle()}
        value={value ?? ""}
        onChange={(e) => setValue(e.target.value)}
      />
    );
  }

  if (q.type === "slider") {
    const v = typeof value === "number" ? value : q.min;
    return (
      <div>
        <input
          id={controlId}
          type="range"
          name={q.id}
          min={q.min}
          max={q.max}
          step={q.step ?? 1}
          required={required}
          value={v}
          onChange={(e) => setValue(Number(e.target.value))}
          style={{ width: "100%" }}
        />
        <div style={{ color: "#4a4a4a", fontSize: 18, fontWeight: 500, marginTop: 8 }}>Value: {v}</div>
      </div>
    );
  }

  if (q.type === "likert") {
    return (
      <fieldset id={controlId} style={{ border: "none", padding: 0, margin: 0 }}>
        <legend style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}>
          {q.label}
        </legend>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {q.options.map((opt, idx) => (
          <label key={opt} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="radio"
              name={q.id}
              value={opt}
              required={required && idx === 0}
              checked={value === opt}
              onChange={() => setValue(opt)}
            />
            <span>{opt}</span>
          </label>
        ))}
        </div>
      </fieldset>
    );
  }

  return null;
}
