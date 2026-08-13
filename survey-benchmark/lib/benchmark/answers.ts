import type { AnswerValue, SurveyQuestion } from "@/lib/benchmark/schema";

export type AnswerStatus = "skipped" | "valid" | "invalid";

export function isAnswerAttempted(value: unknown): boolean {
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value === "string") return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  return false;
}

function isSelection(value: AnswerValue): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

export function isAnswerValid(question: SurveyQuestion, value: AnswerValue): boolean {
  switch (question.kind) {
    case "single-radio":
    case "single-dropdown":
    case "likert":
    case "image-single-select":
      return typeof value === "string" && question.options.some((option) => option.value === value);
    case "single-checkbox":
    case "multiple-checkbox":
      return (
        isSelection(value) &&
        value.length >= question.minSelections &&
        value.length <= question.maxSelections &&
        value.every((selected) => question.options.some((option) => option.value === selected))
      );
    case "slider":
      return (
        typeof value === "number" &&
        Number.isFinite(value) &&
        value >= question.min &&
        value <= question.max
      );
    case "ranking":
      return (
        isSelection(value) &&
        value.length === question.items.length &&
        new Set(value).size === value.length &&
        value.every((selected) => question.items.some((item) => item.value === selected))
      );
    case "numeric":
      return (
        typeof value === "number" &&
        Number.isFinite(value) &&
        (question.min === undefined || value >= question.min) &&
        (question.max === undefined || value <= question.max)
      );
    case "short-text":
    case "long-text": {
      if (typeof value !== "string") return false;
      const length = value.trim().length;
      return length >= question.minLength && length <= question.maxLength;
    }
  }
}

export function classifyAnswer(question: SurveyQuestion, value: unknown): AnswerStatus {
  if (!isAnswerAttempted(value)) return "skipped";
  return isAnswerValid(question, value as AnswerValue) ? "valid" : "invalid";
}

export function summarizeAnswers(
  questions: readonly SurveyQuestion[],
  answers: Readonly<Record<string, unknown>>
) {
  const validQuestionIds: string[] = [];
  const invalidQuestionIds: string[] = [];
  const skippedQuestionIds: string[] = [];

  for (const question of questions) {
    const status = classifyAnswer(question, answers[question.id]);
    if (status === "valid") validQuestionIds.push(question.id);
    else if (status === "invalid") invalidQuestionIds.push(question.id);
    else skippedQuestionIds.push(question.id);
  }

  return {
    attemptedCount: validQuestionIds.length + invalidQuestionIds.length,
    validCount: validQuestionIds.length,
    invalidCount: invalidQuestionIds.length,
    skippedCount: skippedQuestionIds.length,
    validQuestionIds,
    invalidQuestionIds,
    skippedQuestionIds,
  };
}
