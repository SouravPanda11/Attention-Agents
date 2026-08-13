import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionKind } from "@/lib/benchmark/schema";
import { defineTheme, mainQuestionId } from "./types";

const themeId = "wellbeing" as const;
const id = (kind: QuestionKind) => mainQuestionId(themeId, kind);

export const wellbeingTheme = defineTheme({
  id: themeId,
  label: "Health and everyday wellbeing",
  questions: {
    "single-radio": q.radio(
      id("single-radio"),
      1,
      "Which activity most often helps you unwind?",
      ["Walking", "Listening to music", "Reading", "Talking with others"]
    ),
    "single-dropdown": q.dropdown(
      id("single-dropdown"),
      1,
      "On how many days in a typical week are you physically active?",
      ["0 days", "1–2 days", "3–4 days", "5–7 days"]
    ),
    "single-checkbox": q.singleCheckbox(
      id("single-checkbox"),
      1,
      "Select the one wellbeing habit you do most consistently.",
      ["Regular sleep schedule", "Physical activity", "Meal planning", "Relaxation time"]
    ),
    "multiple-checkbox": q.multipleCheckbox(
      id("multiple-checkbox"),
      1,
      "Select exactly two habits you use to support your wellbeing.",
      ["Exercise", "Sleep routine", "Time outdoors", "Social activities"],
      2,
      2
    ),
    likert: q.likert(
      id("likert"),
      1,
      "I am satisfied with my current daily routine."
    ),
    slider: q.slider(
      id("slider"),
      1,
      "From 0 to 10, how would you rate your energy today?"
    ),
    ranking: q.ranking(
      id("ranking"),
      1,
      "Rank these routine goals from most to least important to you.",
      ["Sleep", "Activity", "Relaxation", "Social time"]
    ),
    numeric: q.numeric(
      id("numeric"),
      1,
      "About how many hours did you sleep last night?",
      { min: 0, max: 168, step: 1 }
    ),
    "short-text": q.shortText(
      id("short-text"),
      1,
      "Name one activity that helps you relax."
    ),
    "long-text": q.longText(
      id("long-text"),
      1,
      "In one or two sentences, describe a routine that supports your wellbeing."
    ),
    "image-single-select": q.imageSingleSelect(
      id("image-single-select"),
      1,
      "Which visual design would you prefer for a wellbeing reminder?"
    ),
  },
});
