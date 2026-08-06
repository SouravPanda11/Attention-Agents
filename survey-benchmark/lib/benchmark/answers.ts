import type { AnswerValue, SurveyQuestion } from "@/lib/benchmark/schema";

function isSelection(value: AnswerValue): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

export function isAnswerValid(question: SurveyQuestion, value: AnswerValue): boolean {
  switch (question.kind) {
    case "single-radio":
    case "single-dropdown":
    case "likert":
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

export function answerRequirement(question: SurveyQuestion): string {
  switch (question.kind) {
    case "single-checkbox":
      return "Select exactly one option.";
    case "multiple-checkbox":
      return question.minSelections === question.maxSelections
        ? `Select exactly ${question.minSelections} options.`
        : `Select ${question.minSelections} to ${question.maxSelections} options.`;
    case "slider":
      return "Move the slider to record a response.";
    case "ranking":
      return "Change the order using drag-and-drop or the move buttons.";
    case "short-text":
    case "long-text":
      return `Enter between ${question.minLength} and ${question.maxLength} characters.`;
    default:
      return "A response is required.";
  }
}
