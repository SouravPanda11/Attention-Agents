import type { TextAttentionCheck, TextQuestion } from "@/lib/surveys/types";

export function getTextSurveyQuestions(): TextQuestion[] {
  return [
    { id: "age", type: "number", label: "1) Your age", min: 18, max: 99 },
    {
      id: "follow_by_age",
      type: "choice",
      label: "2) Compared with people in your age group, how closely did you follow the most recent Winter Olympics?",
      options: [
        { value: "very_closely", label: "Very closely" },
        { value: "somewhat", label: "Somewhat" },
        { value: "highlights_only", label: "Only highlights" },
        { value: "not_much", label: "Not much" },
      ],
    },
    {
      id: "employment_status",
      type: "choice",
      label: "3) Current employment status",
      options: [
        { value: "student", label: "Student" },
        { value: "full_time", label: "Employed full-time" },
        { value: "part_time", label: "Employed part-time" },
        { value: "self_employed", label: "Self-employed" },
        { value: "not_working", label: "Not currently working" },
      ],
    },
    {
      id: "watch_time",
      type: "slider",
      label: "4) During the Olympics period, how much time did you spend daily on Olympics content?",
      min: 1,
      max: 10,
      step: 1,
    },
    {
      id: "trust_coverage",
      type: "likert",
      label: "5) I trust the Olympic coverage I usually consume.",
      options: ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
    },
    {
      id: "country_satisfaction",
      type: "slider",
      label: "6) How satisfied were you with your country's Olympic performance?",
      min: 1,
      max: 10,
      step: 1,
    },
    {
      id: "future_interest",
      type: "choice",
      label: "7) How likely are you to follow the next Olympics?",
      options: [
        { value: "very_likely", label: "Very likely" },
        { value: "likely", label: "Likely" },
        { value: "unsure", label: "Unsure" },
        { value: "unlikely", label: "Unlikely" },
      ],
    },
    {
      id: "favorite_moment",
      type: "text",
      label: "8) Briefly describe one memorable Olympic moment for you.",
      maxLen: 240,
    },
    {
      id: "mood_impact",
      type: "likert",
      label: "9) The Olympics positively affected my day-to-day mood.",
      options: ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
    },
  ];
}

export const textAttentionCheck: TextAttentionCheck = {
  id: "attention_text_mid",
  label: "Please select Somewhat for this item.",
  type: "choice",
  options: [
    { value: "very_closely", label: "Very closely" },
    { value: "somewhat", label: "Somewhat" },
    { value: "highlights_only", label: "Only highlights" },
    { value: "not_much", label: "Not much" },
  ],
  expectedValue: "somewhat",
};
