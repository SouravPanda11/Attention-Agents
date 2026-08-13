import { q } from "@/lib/benchmark/questionFactories";
import type { QuestionKind } from "@/lib/benchmark/schema";
import { defineTheme, mainQuestionId } from "./types";

const themeId = "digital" as const;
const id = (kind: QuestionKind) => mainQuestionId(themeId, kind);

export const digitalTheme = defineTheme({
  id: themeId,
  label: "Digital technology and media use",
  questions: {
    "single-radio": q.radio(
      id("single-radio"),
      1,
      "Which device do you use most often to access the internet?",
      ["Smartphone", "Laptop", "Tablet", "Desktop computer"]
    ),
    "single-dropdown": q.dropdown(
      id("single-dropdown"),
      1,
      "How often do you check online news or media?",
      ["Never", "A few times a month", "A few times a week", "Daily"]
    ),
    "single-checkbox": q.singleCheckbox(
      id("single-checkbox"),
      1,
      "Select the one type of online activity you use most.",
      ["Communication", "Entertainment", "Work or study", "Shopping or services"]
    ),
    "multiple-checkbox": q.multipleCheckbox(
      id("multiple-checkbox"),
      1,
      "Select exactly two online activities you do regularly.",
      ["Messaging", "Streaming", "Reading news", "Creating content"],
      2,
      2
    ),
    likert: q.likert(
      id("likert"),
      1,
      "Digital tools make my everyday tasks easier."
    ),
    slider: q.slider(
      id("slider"),
      1,
      "From 0 to 10, how comfortable are you using new digital tools?"
    ),
    ranking: q.ranking(
      id("ranking"),
      1,
      "Rank these digital-tool qualities from most to least important.",
      ["Ease of use", "Privacy", "Speed", "Features"]
    ),
    numeric: q.numeric(
      id("numeric"),
      1,
      "About how many hours did you spend online yesterday?",
      { min: 0, max: 168, step: 1 }
    ),
    "short-text": q.shortText(
      id("short-text"),
      1,
      "Name one digital tool you use regularly."
    ),
    "long-text": q.longText(
      id("long-text"),
      1,
      "In one or two sentences, describe how digital technology fits into your day."
    ),
    "image-single-select": q.imageSingleSelect(
      id("image-single-select"),
      1,
      "Which visual design would you prefer for an app shortcut?"
    ),
  },
});
