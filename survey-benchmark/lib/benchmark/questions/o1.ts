import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionBank } from "@/lib/benchmark/schema";

// Edit the 10 question entries below. Keep every id unique and stable after data collection begins.
export const o1Bank = {
  id: "o1",
  occurrence: 1,
  contentVersion: 1,
  questions: [
    // Occurrence block 1: one question of each interaction type.
    q.radio("o01-b01-radio", 1, "Placeholder block 1: choose one option using radio buttons."),
    q.dropdown("o01-b01-dropdown", 1, "Placeholder block 1: choose one option from the dropdown."),
    q.singleCheckbox("o01-b01-single-checkbox", 1, "Placeholder block 1: select exactly one checkbox option."),
    q.multipleCheckbox("o01-b01-multiple-checkbox", 1, "Placeholder block 1: select exactly two checkbox options."),
    q.likert("o01-b01-likert", 1, "Placeholder block 1: indicate how strongly you agree with this statement."),
    q.slider("o01-b01-slider", 1, "Placeholder block 1: choose a value on the slider."),
    q.ranking("o01-b01-ranking", 1, "Placeholder block 1: rank the four items from highest to lowest priority."),
    q.numeric("o01-b01-numeric", 1, "Placeholder block 1: enter a whole number from 0 to 100."),
    q.shortText("o01-b01-short-text", 1, "Placeholder block 1: provide a short text response."),
    q.longText("o01-b01-long-text", 1, "Placeholder block 1: explain your response in one or two sentences."),
  ],
} satisfies QuestionBank;

