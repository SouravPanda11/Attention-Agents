import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionKind } from "@/lib/benchmark/schema";

import { defineTheme, mainQuestionId } from "./types";

const THEME_ID = "work" as const;

function id(kind: QuestionKind) {
  return mainQuestionId(THEME_ID, kind);
}

export const workTheme = defineTheme({
  id: THEME_ID,
  label: "Employment and workplace experience",
  questions: {
    "single-radio": q.radio(
      id("single-radio"),
      1,
      "Which work arrangement would you generally prefer?",
      ["On-site", "Hybrid", "Remote", "Flexible by task"]
    ),
    "single-dropdown": q.dropdown(
      id("single-dropdown"),
      1,
      "How often does your team hold a scheduled meeting?",
      ["Never", "Monthly or less", "Weekly", "Several times a week"]
    ),
    "single-checkbox": q.singleCheckbox(
      id("single-checkbox"),
      1,
      "Select the one tool you use most for workplace coordination.",
      ["Email", "Chat", "Video meetings", "Project board"]
    ),
    "multiple-checkbox": q.multipleCheckbox(
      id("multiple-checkbox"),
      1,
      "Select exactly two factors you value in a workplace.",
      ["Flexible hours", "Stable schedule", "Team support", "Career growth"],
      2,
      2
    ),
    likert: q.likert(
      id("likert"),
      1,
      "I can organize my work tasks effectively."
    ),
    slider: q.slider(
      id("slider"),
      1,
      "From 0 to 10, how manageable is your current workload?"
    ),
    ranking: q.ranking(
      id("ranking"),
      1,
      "Rank these job features from most to least important.",
      ["Pay", "Flexibility", "Stability", "Growth"]
    ),
    numeric: q.numeric(
      id("numeric"),
      1,
      "About how many scheduled meetings did you attend in the past week?",
      { min: 0, max: 168, step: 1 }
    ),
    "short-text": q.shortText(
      id("short-text"),
      1,
      "Name one tool that helps you organize work."
    ),
    "long-text": q.longText(
      id("long-text"),
      1,
      "In one or two sentences, describe how you organize a typical work task."
    ),
    "image-single-select": q.imageSingleSelect(
      id("image-single-select"),
      1,
      "Which visual design would you prefer for a task-board marker?"
    ),
  },
});
