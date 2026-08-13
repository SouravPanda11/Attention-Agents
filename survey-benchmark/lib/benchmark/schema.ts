export const SUITE_VERSION = "v0" as const;
export const PRESENTATION_PROFILES = ["standard"] as const;
export const OCCURRENCES = [1, 2, 3, 4, 5, 6, 7, 8] as const;
export const LAYOUT_MODES = ["item", "navigation"] as const;
export const ORDER_IDS = ["order01", "order02", "order03"] as const;

export const QUESTION_KINDS = [
  "single-radio",
  "single-dropdown",
  "single-checkbox",
  "multiple-checkbox",
  "likert",
  "slider",
  "ranking",
  "numeric",
  "short-text",
  "long-text",
  "image-single-select",
] as const;

export const SUBSTANTIVE_QUESTIONS_PER_BLOCK = QUESTION_KINDS.length;
export const ATTENTION_CHECKS_PER_BLOCK = 2 as const;
export const RENDERED_QUESTIONS_PER_BLOCK = 13 as const;

export type PresentationProfile = (typeof PRESENTATION_PROFILES)[number];
export type Occurrence = (typeof OCCURRENCES)[number];
export type OccurrenceId = `o${Occurrence}`;
export type LayoutMode = (typeof LAYOUT_MODES)[number];
export type OrderId = (typeof ORDER_IDS)[number];
export type QuestionKind = (typeof QUESTION_KINDS)[number];

export type ChoiceOption = {
  value: string;
  label: string;
};

export type ImageChoiceOption = ChoiceOption & {
  imageSrc: string;
  imageAlt: string;
};

type BaseQuestion<K extends QuestionKind> = {
  id: string;
  kind: K;
  block: number;
  prompt: string;
  required: false;
  helpText?: string;
  dependsOn?: readonly string[];
};

export type SingleRadioQuestion = BaseQuestion<"single-radio"> & {
  options: readonly ChoiceOption[];
};

export type SingleDropdownQuestion = BaseQuestion<"single-dropdown"> & {
  options: readonly ChoiceOption[];
};

export type SingleCheckboxQuestion = BaseQuestion<"single-checkbox"> & {
  options: readonly ChoiceOption[];
  minSelections: 1;
  maxSelections: 1;
};

export type MultipleCheckboxQuestion = BaseQuestion<"multiple-checkbox"> & {
  options: readonly ChoiceOption[];
  minSelections: number;
  maxSelections: number;
};

export type LikertQuestion = BaseQuestion<"likert"> & {
  options: readonly ChoiceOption[];
};

export type SliderQuestion = BaseQuestion<"slider"> & {
  min: number;
  max: number;
  step: number;
  requireInteraction: true;
};

export type RankingQuestion = BaseQuestion<"ranking"> & {
  items: readonly ChoiceOption[];
  requireInteraction: true;
};

export type NumericQuestion = BaseQuestion<"numeric"> & {
  min?: number;
  max?: number;
  step?: number;
};

export type ShortTextQuestion = BaseQuestion<"short-text"> & {
  minLength: number;
  maxLength: number;
};

export type LongTextQuestion = BaseQuestion<"long-text"> & {
  minLength: number;
  maxLength: number;
  rows: number;
};

export type ImageSingleSelectQuestion = BaseQuestion<"image-single-select"> & {
  options: readonly ImageChoiceOption[];
};

export type SurveyQuestion =
  | SingleRadioQuestion
  | SingleDropdownQuestion
  | SingleCheckboxQuestion
  | MultipleCheckboxQuestion
  | LikertQuestion
  | SliderQuestion
  | RankingQuestion
  | NumericQuestion
  | ShortTextQuestion
  | LongTextQuestion
  | ImageSingleSelectQuestion;

export type QuestionBank = {
  id: OccurrenceId;
  occurrence: Occurrence;
  contentVersion: number;
  questions: readonly SurveyQuestion[];
};

export type AnswerValue = string | number | string[] | undefined;
export type SurveyAnswers = Record<string, AnswerValue>;

export type WorkflowPage = {
  id: string;
  index: number;
  questions: readonly SurveyQuestion[];
};

export type Workflow = {
  id: string;
  suiteVersion: typeof SUITE_VERSION;
  hasWelcomePage: true;
  profile: PresentationProfile;
  occurrence: Occurrence;
  occurrenceId: OccurrenceId;
  layout: LayoutMode;
  orderId: OrderId;
  contentVersion: number;
  attentionCheckContentVersion: number;
  substantiveQuestionCount: number;
  attentionCheckCount: number;
  renderedQuestionCount: number;
  questionCount: number;
  pageCount: number;
  questionsPerNavigationPage: typeof RENDERED_QUESTIONS_PER_BLOCK;
  orderedQuestionIds: readonly string[];
  pages: readonly WorkflowPage[];
};

export function isOccurrence(value: number): value is Occurrence {
  return OCCURRENCES.includes(value as Occurrence);
}

export function isLayoutMode(value: string): value is LayoutMode {
  return LAYOUT_MODES.includes(value as LayoutMode);
}

export function isOrderId(value: string): value is OrderId {
  return ORDER_IDS.includes(value as OrderId);
}

export function isPresentationProfile(value: string): value is PresentationProfile {
  return PRESENTATION_PROFILES.includes(value as PresentationProfile);
}
