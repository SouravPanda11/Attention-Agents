import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionKind } from "@/lib/benchmark/schema";

import { defineTheme, mainQuestionId } from "./types";

const THEME_ID = "lifestyle" as const;

function id(kind: QuestionKind) {
  return mainQuestionId(THEME_ID, kind);
}

export const lifestyleTheme = defineTheme({
  id: THEME_ID,
  label: "Lifestyle, environment, and community",
  questions: {
    "single-radio": q.radio(
      id("single-radio"),
      1,
      "Which mode of transport do you use most often for local trips?",
      ["Walking", "Bicycle", "Public transport", "Car"]
    ),
    "single-dropdown": q.dropdown(
      id("single-dropdown"),
      1,
      "How often do you attend a local community event?",
      ["Never", "Once a year or less", "A few times a year", "Monthly or more"]
    ),
    "single-checkbox": q.singleCheckbox(
      id("single-checkbox"),
      1,
      "Select the one feature you value most in your neighborhood.",
      ["Quiet surroundings", "Nearby shops", "Green space", "Transport access"]
    ),
    "multiple-checkbox": q.multipleCheckbox(
      id("multiple-checkbox"),
      1,
      "Select exactly two activities you enjoy during free time.",
      ["Outdoor activity", "Arts or crafts", "Games", "Social gatherings"],
      2,
      2
    ),
    likert: q.likert(
      id("likert"),
      1,
      "I feel connected to my local community."
    ),
    slider: q.slider(
      id("slider"),
      1,
      "From 0 to 10, how satisfied are you with your current leisure time?"
    ),
    ranking: q.ranking(
      id("ranking"),
      1,
      "Rank these neighborhood features from most to least important.",
      ["Safety", "Green space", "Services", "Transport"]
    ),
    numeric: q.numeric(
      id("numeric"),
      1,
      "How many community or leisure activities did you attend in the past month?",
      { min: 0, max: 168, step: 1 }
    ),
    "short-text": q.shortText(
      id("short-text"),
      1,
      "Name one activity you enjoy in your community."
    ),
    "long-text": q.longText(
      id("long-text"),
      1,
      "In one or two sentences, describe an activity you value in your community."
    ),
    "image-single-select": q.imageSingleSelect(
      id("image-single-select"),
      1,
      "Which visual design would you prefer for a community-events icon?"
    ),
  },
});
