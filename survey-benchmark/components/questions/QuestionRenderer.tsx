"use client";

import type { AnswerValue, SurveyQuestion } from "@/lib/benchmark/schema";
import { answerRequirement } from "@/lib/benchmark/answers";
import { RankingInput } from "@/components/questions/RankingInput";

function selectedValues(value: AnswerValue): string[] {
  return Array.isArray(value) ? value : [];
}

export function QuestionRenderer({
  question,
  number,
  value,
  invalid,
  onChange,
}: {
  question: SurveyQuestion;
  number: number;
  value: AnswerValue;
  invalid: boolean;
  onChange: (value: AnswerValue) => void;
}) {
  const controlId = `question-${question.id}`;
  const promptId = `${controlId}-prompt`;
  const errorId = `${controlId}-error`;
  const describedBy = invalid ? errorId : undefined;

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
      <span className="required-mark" aria-label="required">
        *
      </span>
    </div>
  );

  let field: React.ReactNode;

  if (question.kind === "single-radio" || question.kind === "likert") {
    field = (
      <fieldset className="option-fieldset" aria-labelledby={promptId} aria-describedby={describedBy}>
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
  } else if (question.kind === "single-dropdown") {
    field = (
      <select
        id={controlId}
        name={question.id}
        value={typeof value === "string" ? value : ""}
        aria-labelledby={promptId}
        aria-describedby={describedBy}
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
      <fieldset className="option-fieldset" aria-labelledby={promptId} aria-describedby={describedBy}>
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
          aria-describedby={describedBy}
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
        aria-describedby={describedBy}
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
        aria-describedby={describedBy}
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
        aria-describedby={describedBy}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  }

  return (
    <section
      className={`question-card${invalid ? " question-card-invalid" : ""}`}
      tabIndex={-1}
      data-question-id={question.id}
      data-question-type={question.kind}
      data-question-block={question.block}
      data-question-required="true"
    >
      {prompt}
      {question.helpText ? <p className="question-help">{question.helpText}</p> : null}
      {field}
      {invalid ? (
        <p id={errorId} className="field-error" role="alert">
          {answerRequirement(question)}
        </p>
      ) : null}
    </section>
  );
}
