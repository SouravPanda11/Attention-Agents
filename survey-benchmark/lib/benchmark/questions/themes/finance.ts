import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionKind } from "@/lib/benchmark/schema";

import { defineTheme, mainQuestionId } from "./types";

const THEME_ID = "finance" as const;

function id(kind: QuestionKind) {
  return mainQuestionId(THEME_ID, kind);
}

export const financeTheme = defineTheme({
  id: THEME_ID,
  label: "Financial habits and economic perceptions",
  questions: {
    "single-radio": q.radio(
      id("single-radio"),
      1,
      "Which method do you primarily use to track a budget?",
      ["Mobile app", "Spreadsheet", "Paper notes", "I do not track one"]
    ),
    "single-dropdown": q.dropdown(
      id("single-dropdown"),
      1,
      "How often do you review your bank or spending records?",
      ["Never", "Monthly or less", "Weekly", "Daily"]
    ),
    "single-checkbox": q.singleCheckbox(
      id("single-checkbox"),
      1,
      "Select the one spending category you monitor most closely.",
      ["Housing", "Food", "Transport", "Leisure"]
    ),
    "multiple-checkbox": q.multipleCheckbox(
      id("multiple-checkbox"),
      1,
      "Select exactly two actions you use when planning expenses.",
      ["Set a limit", "Compare prices", "Track receipts", "Delay a purchase"],
      2,
      2
    ),
    likert: q.likert(
      id("likert"),
      1,
      "I feel confident managing routine expenses."
    ),
    slider: q.slider(
      id("slider"),
      1,
      "From 0 to 10, how confident are you about your routine budgeting?"
    ),
    ranking: q.ranking(
      id("ranking"),
      1,
      "Rank these financial goals from most to least important.",
      ["Paying bills", "Saving", "Reducing debt", "Leisure spending"]
    ),
    numeric: q.numeric(
      id("numeric"),
      1,
      "How many times did you review your spending in the past month?",
      { min: 0, max: 168, step: 1 }
    ),
    "short-text": q.shortText(
      id("short-text"),
      1,
      "Name one method you use to monitor spending."
    ),
    "long-text": q.longText(
      id("long-text"),
      1,
      "In one or two sentences, describe how you plan for a routine expense."
    ),
    "image-single-select": q.imageSingleSelect(
      id("image-single-select"),
      1,
      "Which visual design would you prefer for a budgeting tool?"
    ),
  },
});
