import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionBank } from "@/lib/benchmark/schema";

// Edit the 20 question entries below. Keep every id unique and stable after data collection begins.
export const o2Bank = {
  id: "o2",
  occurrence: 2,
  contentVersion: 1,
  questions: [
    // Occurrence block 1: one question of each interaction type.
    q.radio("o02-b01-radio", 1, "Placeholder block 1: choose one option using radio buttons."),
    q.dropdown("o02-b01-dropdown", 1, "Placeholder block 1: choose one option from the dropdown."),
    q.singleCheckbox("o02-b01-single-checkbox", 1, "Placeholder block 1: select exactly one checkbox option."),
    q.multipleCheckbox("o02-b01-multiple-checkbox", 1, "Placeholder block 1: select exactly two checkbox options."),
    q.likert("o02-b01-likert", 1, "Placeholder block 1: indicate how strongly you agree with this statement."),
    q.slider("o02-b01-slider", 1, "Placeholder block 1: choose a value on the slider."),
    q.ranking("o02-b01-ranking", 1, "Placeholder block 1: rank the four items from highest to lowest priority."),
    q.numeric("o02-b01-numeric", 1, "Placeholder block 1: enter a whole number from 0 to 100."),
    q.shortText("o02-b01-short-text", 1, "Placeholder block 1: provide a short text response."),
    q.longText("o02-b01-long-text", 1, "Placeholder block 1: explain your response in one or two sentences."),

    // Occurrence block 2: one question of each interaction type.
    q.radio("o02-b02-radio", 2, "Placeholder block 2: choose one option using radio buttons."),
    q.dropdown("o02-b02-dropdown", 2, "Placeholder block 2: choose one option from the dropdown."),
    q.singleCheckbox("o02-b02-single-checkbox", 2, "Placeholder block 2: select exactly one checkbox option."),
    q.multipleCheckbox("o02-b02-multiple-checkbox", 2, "Placeholder block 2: select exactly two checkbox options."),
    q.likert("o02-b02-likert", 2, "Placeholder block 2: indicate how strongly you agree with this statement."),
    q.slider("o02-b02-slider", 2, "Placeholder block 2: choose a value on the slider."),
    q.ranking("o02-b02-ranking", 2, "Placeholder block 2: rank the four items from highest to lowest priority."),
    q.numeric("o02-b02-numeric", 2, "Placeholder block 2: enter a whole number from 0 to 100."),
    q.shortText("o02-b02-short-text", 2, "Placeholder block 2: provide a short text response."),
    q.longText("o02-b02-long-text", 2, "Placeholder block 2: explain your response in one or two sentences."),
  ],
} satisfies QuestionBank;

