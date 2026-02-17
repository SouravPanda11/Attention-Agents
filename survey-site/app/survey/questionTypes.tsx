"use client";

import React from "react";
import type { TextAnswerValue, TextQuestion } from "@/lib/surveys/types";

export type Question = TextQuestion;
export type QuestionValue = TextAnswerValue;

export function QuestionRenderer({
  q,
  value,
  setValue,
}: {
  q: Question;
  value: QuestionValue;
  setValue: (v: QuestionValue) => void;
}) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>{q.label}</label>
      <QuestionField q={q} value={value} setValue={setValue} />
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
}: {
  q: Question;
  value: QuestionValue;
  setValue: (v: QuestionValue) => void;
}) {
  if (q.type === "number") {
    return (
      <input
        type="number"
        min={q.min}
        max={q.max}
        style={baseFieldStyle()}
        value={value ?? ""}
        onChange={(e) => setValue(e.target.value ? Number(e.target.value) : undefined)}
      />
    );
  }

  if (q.type === "choice") {
    return (
      <select style={baseFieldStyle()} value={value ?? ""} onChange={(e) => setValue(e.target.value)}>
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
        rows={3}
        maxLength={q.maxLen}
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
          type="range"
          min={q.min}
          max={q.max}
          step={q.step ?? 1}
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
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {q.options.map((opt) => (
          <label key={opt} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="radio" name={q.id} checked={value === opt} onChange={() => setValue(opt)} />
            <span>{opt}</span>
          </label>
        ))}
      </div>
    );
  }

  return null;
}
