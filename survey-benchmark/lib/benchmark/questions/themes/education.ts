import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionKind } from "@/lib/benchmark/schema";
import { defineTheme, mainQuestionId } from "./types";

const themeId = "education" as const;
const id = (kind: QuestionKind) => mainQuestionId(themeId, kind);

export const educationTheme = defineTheme({
  id: themeId,
  label: "Education and learning",
  questions: {
    "single-radio": q.radio(
      id("single-radio"),
      1,
      "Which learning format do you generally prefer?",
      ["Live class", "Recorded lesson", "Self-paced exercise", "Written material"]
    ),
    "single-dropdown": q.dropdown(
      id("single-dropdown"),
      1,
      "How often do you use an online resource to learn something new?",
      ["Never", "Monthly or less", "Weekly", "Daily"]
    ),
    "single-checkbox": q.singleCheckbox(
      id("single-checkbox"),
      1,
      "Select the one study aid you use most often.",
      ["Notes", "Practice questions", "Videos", "Study group"]
    ),
    "multiple-checkbox": q.multipleCheckbox(
      id("multiple-checkbox"),
      1,
      "Select exactly two resources you commonly use while learning.",
      ["Textbooks", "Videos", "Practice tasks", "Discussion forums"],
      2,
      2
    ),
    likert: q.likert(
      id("likert"),
      1,
      "I feel comfortable learning through online materials."
    ),
    slider: q.slider(
      id("slider"),
      1,
      "From 0 to 10, how motivated are you to learn a new skill?"
    ),
    ranking: q.ranking(
      id("ranking"),
      1,
      "Rank these learning features from most to least useful.",
      ["Clear explanations", "Practice", "Feedback", "Flexible timing"]
    ),
    numeric: q.numeric(
      id("numeric"),
      1,
      "About how many hours did you spend learning in the past week?",
      { min: 0, max: 168, step: 1 }
    ),
    "short-text": q.shortText(
      id("short-text"),
      1,
      "Name one skill you would like to learn."
    ),
    "long-text": q.longText(
      id("long-text"),
      1,
      "In one or two sentences, describe how you prefer to learn something new."
    ),
    "image-single-select": q.imageSingleSelect(
      id("image-single-select"),
      1,
      "Which visual design would you prefer for a learning tool?"
    ),
  },
});
