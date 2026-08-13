import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionKind } from "@/lib/benchmark/schema";
import { defineTheme, mainQuestionId } from "./types";

const themeId = "consumer" as const;
const id = (kind: QuestionKind) => mainQuestionId(themeId, kind);

export const consumerTheme = defineTheme({
  id: themeId,
  label: "Consumer behavior and purchasing",
  questions: {
    "single-radio": q.radio(
      id("single-radio"),
      1,
      "Which factor matters most when you buy a household product?",
      ["Price", "Quality", "Brand", "Customer reviews"]
    ),
    "single-dropdown": q.dropdown(
      id("single-dropdown"),
      1,
      "How often do you shop online?",
      ["Never", "Less than monthly", "Monthly", "Weekly or more"]
    ),
    "single-checkbox": q.singleCheckbox(
      id("single-checkbox"),
      1,
      "Select the one statement that best describes your usual shopping plan.",
      ["I make a list", "I compare several stores", "I decide while shopping", "I rarely shop"]
    ),
    "multiple-checkbox": q.multipleCheckbox(
      id("multiple-checkbox"),
      1,
      "Select exactly two sources you commonly consult before a purchase.",
      ["Product reviews", "Friends or family", "Store information", "Social media"],
      2,
      2
    ),
    likert: q.likert(
      id("likert"),
      1,
      "I usually compare alternatives before making a purchase."
    ),
    slider: q.slider(
      id("slider"),
      1,
      "From 0 to 10, how satisfied are you with your recent purchases?"
    ),
    ranking: q.ranking(
      id("ranking"),
      1,
      "Rank these purchase factors from most to least important.",
      ["Price", "Quality", "Convenience", "Reviews"]
    ),
    numeric: q.numeric(
      id("numeric"),
      1,
      "About how many online purchases did you make in the past 30 days?",
      { min: 0, max: 168, step: 1 }
    ),
    "short-text": q.shortText(
      id("short-text"),
      1,
      "Name one product category you purchase regularly."
    ),
    "long-text": q.longText(
      id("long-text"),
      1,
      "In one or two sentences, describe how you make a routine purchase decision."
    ),
    "image-single-select": q.imageSingleSelect(
      id("image-single-select"),
      1,
      "Which visual design would you prefer on a shopping-list icon?"
    ),
  },
});
