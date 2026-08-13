"use client";

import Image from "next/image";
import type { AnswerValue, SurveyQuestion } from "@/lib/benchmark/schema";
import { RankingInput } from "@/components/questions/RankingInput";

function selectedValues(value: AnswerValue): string[] {
  return Array.isArray(value) ? value : [];
}

export function QuestionRenderer({
  question,
  number,
  value,
  onChange,
}: {
  question: SurveyQuestion;
  number: number;
  value: AnswerValue;
  onChange: (value: AnswerValue) => void;
}) {
  const controlId = `question-${question.id}`;
  const promptId = `${controlId}-prompt`;

  function toggleCheckbox(optionValue: string, exactlyOne: boolean) {
    const current = selectedValues(value);
    if (exactlyOne) {
      onChange(current.includes(optionValue) ? [] : [optionValue]);
      return;
    }
    onChange(
      current.includes(optionValue)
        ? current.filter((candidate) => candidate !== optionValue)
        : [...current, optionValue]
    );
  }

  const prompt = (
    <div id={promptId} className="question-prompt">
      <span className="question-number">{number}.</span>
      <span>{question.prompt}</span>
    </div>
  );

  let field: React.ReactNode;

  if (question.kind === "single-radio" || question.kind === "likert") {
    field = (
      <fieldset className="option-fieldset" aria-labelledby={promptId}>
        <legend className="sr-only">{question.prompt}</legend>
        <div className={question.kind === "likert" ? "option-row" : "option-column"}>
          {question.options.map((option) => (
            <label className="option-label" key={option.value}>
              <input
                type="radio"
                name={question.id}
                value={option.value}
                checked={value === option.value}
                onChange={() => onChange(option.value)}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
      </fieldset>
    );
  } else if (question.kind === "image-single-select") {
    field = (
      <fieldset className="option-fieldset" aria-labelledby={promptId}>
        <legend className="sr-only">{question.prompt}</legend>
        <div className="image-option-grid">
          {question.options.map((option) => (
            <label
              className={`image-option-label${value === option.value ? " image-option-selected" : ""}`}
              key={option.value}
            >
              <input
                type="radio"
                name={question.id}
                value={option.value}
                checked={value === option.value}
                onChange={() => onChange(option.value)}
              />
              <Image
                className="image-option-media"
                src={option.imageSrc}
                alt={option.imageAlt}
                width={480}
                height={320}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
      </fieldset>
    );
  } else if (question.kind === "single-dropdown") {
    field = (
      <select
        id={controlId}
        name={question.id}
        value={typeof value === "string" ? value : ""}
        aria-labelledby={promptId}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="" disabled>
          Select one option
        </option>
        {question.options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    );
  } else if (question.kind === "single-checkbox" || question.kind === "multiple-checkbox") {
    const selected = selectedValues(value);
    field = (
      <fieldset className="option-fieldset" aria-labelledby={promptId}>
        <legend className="sr-only">{question.prompt}</legend>
        <div className="option-column">
          {question.options.map((option) => (
            <label className="option-label" key={option.value}>
              <input
                type="checkbox"
                name={question.id}
                value={option.value}
                checked={selected.includes(option.value)}
                onChange={() => toggleCheckbox(option.value, question.kind === "single-checkbox")}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
      </fieldset>
    );
  } else if (question.kind === "slider") {
    const midpoint = question.min + Math.round((question.max - question.min) / (2 * question.step)) * question.step;
    const sliderValue = typeof value === "number" ? value : midpoint;
    field = (
      <div className="slider-field">
        <input
          id={controlId}
          type="range"
          name={question.id}
          min={question.min}
          max={question.max}
          step={question.step}
          value={sliderValue}
          aria-labelledby={promptId}
          data-interacted={typeof value === "number" ? "true" : "false"}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <output htmlFor={controlId} className="slider-output">
          {typeof value === "number" ? `Selected value: ${value}` : "Move the slider to select a value"}
        </output>
      </div>
    );
  } else if (question.kind === "ranking") {
    field = (
      <RankingInput
        question={question}
        value={Array.isArray(value) ? value : undefined}
        onChange={onChange}
      />
    );
  } else if (question.kind === "numeric") {
    field = (
      <input
        id={controlId}
        type="number"
        name={question.id}
        min={question.min}
        max={question.max}
        step={question.step}
        value={typeof value === "number" ? value : ""}
        aria-labelledby={promptId}
        onChange={(event) => onChange(event.target.value === "" ? undefined : Number(event.target.value))}
      />
    );
  } else if (question.kind === "short-text") {
    field = (
      <input
        id={controlId}
        type="text"
        name={question.id}
        minLength={question.minLength}
        maxLength={question.maxLength}
        value={typeof value === "string" ? value : ""}
        aria-labelledby={promptId}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  } else {
    field = (
      <textarea
        id={controlId}
        name={question.id}
        rows={question.rows}
        minLength={question.minLength}
        maxLength={question.maxLength}
        value={typeof value === "string" ? value : ""}
        aria-labelledby={promptId}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  }

  return (
    <section
      className="question-card"
      tabIndex={-1}
      data-question-id={question.id}
      data-question-type={question.kind}
      data-question-block={question.block}
      data-question-required="false"
    >
      {prompt}
      {question.helpText ? <p className="question-help">{question.helpText}</p> : null}
      {field}
    </section>
  );
}
