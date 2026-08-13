import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionKind } from "@/lib/benchmark/schema";

import { defineTheme, mainQuestionId } from "./types";

const THEME_ID = "civic" as const;

function id(kind: QuestionKind) {
  return mainQuestionId(THEME_ID, kind);
}

export const civicTheme = defineTheme({
  id: THEME_ID,
  label: "Civic engagement and public services",
  questions: {
    "single-radio": q.radio(
      id("single-radio"),
      1,
      "How would you prefer to receive updates from your local government?",
      ["Email", "Official website", "Social media", "Text message"]
    ),
    "single-dropdown": q.dropdown(
      id("single-dropdown"),
      1,
      "How often do you use an online public-service portal?",
      ["Never", "Less than yearly", "A few times a year", "Monthly or more"]
    ),
    "single-checkbox": q.singleCheckbox(
      id("single-checkbox"),
      1,
      "Select the one local service you use most often.",
      ["Library", "Parks", "Public transport", "Administrative services"]
    ),
    "multiple-checkbox": q.multipleCheckbox(
      id("multiple-checkbox"),
      1,
      "Select exactly two ways you learn about local issues.",
      ["Local news", "Official notices", "Community groups", "Neighbors"],
      2,
      2
    ),
    likert: q.likert(
      id("likert"),
      1,
      "It is easy to find information about local public services."
    ),
    slider: q.slider(
      id("slider"),
      1,
      "From 0 to 10, how satisfied are you with access to local services?"
    ),
    ranking: q.ranking(
      id("ranking"),
      1,
      "Rank these public-service qualities from most to least important.",
      ["Accessibility", "Speed", "Clarity", "Reliability"]
    ),
    numeric: q.numeric(
      id("numeric"),
      1,
      "How many local public services did you use in the past year?",
      { min: 0, max: 168, step: 1 }
    ),
    "short-text": q.shortText(
      id("short-text"),
      1,
      "Name one local public service you know about."
    ),
    "long-text": q.longText(
      id("long-text"),
      1,
      "In one or two sentences, describe how you find information about a public service."
    ),
    "image-single-select": q.imageSingleSelect(
      id("image-single-select"),
      1,
      "Which visual design would you prefer for a public-service portal?"
    ),
  },
});
