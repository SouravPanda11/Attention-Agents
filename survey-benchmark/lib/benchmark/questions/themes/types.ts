import type {
  QuestionKind,
  SurveyQuestion,
} from "@/lib/benchmark/schema";

export const THEME_IDS = [
  "consumer",
  "digital",
  "wellbeing",
  "education",
  "work",
  "finance",
  "civic",
  "lifestyle",
] as const;

export type ThemeId = (typeof THEME_IDS)[number];

export type ThemeQuestionSet = {
  [Kind in QuestionKind]: Extract<SurveyQuestion, { kind: Kind }>;
};

export type ThemeDefinition<Id extends ThemeId = ThemeId> = {
  id: Id;
  label: string;
  questions: ThemeQuestionSet;
};

/** Gives each substantive question its stable canonical benchmark id. */
export function mainQuestionId(
  themeId: ThemeId,
  kind: QuestionKind
): `main-${ThemeId}-${QuestionKind}` {
  return `main-${themeId}-${kind}`;
}

/** Adds compile-time checks while leaving a theme file easy to edit. */
export function defineTheme<const Id extends ThemeId>(
  definition: ThemeDefinition<Id>
): ThemeDefinition<Id> {
  return definition;
}
