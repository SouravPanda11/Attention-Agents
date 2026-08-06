import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionBank } from "@/lib/benchmark/schema";

// Edit the 30 question entries below. Keep every id unique and stable after data collection begins.
export const o3Bank = {
  id: "o3",
  occurrence: 3,
  contentVersion: 1,
  questions: [
    // Occurrence block 1: one question of each interaction type.
    q.radio("o03-b01-radio", 1, "Placeholder block 1: choose one option using radio buttons."),
    q.dropdown("o03-b01-dropdown", 1, "Placeholder block 1: choose one option from the dropdown."),
    q.singleCheckbox("o03-b01-single-checkbox", 1, "Placeholder block 1: select exactly one checkbox option."),
    q.multipleCheckbox("o03-b01-multiple-checkbox", 1, "Placeholder block 1: select exactly two checkbox options."),
    q.likert("o03-b01-likert", 1, "Placeholder block 1: indicate how strongly you agree with this statement."),
    q.slider("o03-b01-slider", 1, "Placeholder block 1: choose a value on the slider."),
    q.ranking("o03-b01-ranking", 1, "Placeholder block 1: rank the four items from highest to lowest priority."),
    q.numeric("o03-b01-numeric", 1, "Placeholder block 1: enter a whole number from 0 to 100."),
    q.shortText("o03-b01-short-text", 1, "Placeholder block 1: provide a short text response."),
    q.longText("o03-b01-long-text", 1, "Placeholder block 1: explain your response in one or two sentences."),

    // Occurrence block 2: one question of each interaction type.
    q.radio("o03-b02-radio", 2, "Placeholder block 2: choose one option using radio buttons."),
    q.dropdown("o03-b02-dropdown", 2, "Placeholder block 2: choose one option from the dropdown."),
    q.singleCheckbox("o03-b02-single-checkbox", 2, "Placeholder block 2: select exactly one checkbox option."),
    q.multipleCheckbox("o03-b02-multiple-checkbox", 2, "Placeholder block 2: select exactly two checkbox options."),
    q.likert("o03-b02-likert", 2, "Placeholder block 2: indicate how strongly you agree with this statement."),
    q.slider("o03-b02-slider", 2, "Placeholder block 2: choose a value on the slider."),
    q.ranking("o03-b02-ranking", 2, "Placeholder block 2: rank the four items from highest to lowest priority."),
    q.numeric("o03-b02-numeric", 2, "Placeholder block 2: enter a whole number from 0 to 100."),
    q.shortText("o03-b02-short-text", 2, "Placeholder block 2: provide a short text response."),
    q.longText("o03-b02-long-text", 2, "Placeholder block 2: explain your response in one or two sentences."),

    // Occurrence block 3: one question of each interaction type.
    q.radio("o03-b03-radio", 3, "Placeholder block 3: choose one option using radio buttons."),
    q.dropdown("o03-b03-dropdown", 3, "Placeholder block 3: choose one option from the dropdown."),
    q.singleCheckbox("o03-b03-single-checkbox", 3, "Placeholder block 3: select exactly one checkbox option."),
    q.multipleCheckbox("o03-b03-multiple-checkbox", 3, "Placeholder block 3: select exactly two checkbox options."),
    q.likert("o03-b03-likert", 3, "Placeholder block 3: indicate how strongly you agree with this statement."),
    q.slider("o03-b03-slider", 3, "Placeholder block 3: choose a value on the slider."),
    q.ranking("o03-b03-ranking", 3, "Placeholder block 3: rank the four items from highest to lowest priority."),
    q.numeric("o03-b03-numeric", 3, "Placeholder block 3: enter a whole number from 0 to 100."),
    q.shortText("o03-b03-short-text", 3, "Placeholder block 3: provide a short text response."),
    q.longText("o03-b03-long-text", 3, "Placeholder block 3: explain your response in one or two sentences."),
  ],
} satisfies QuestionBank;

