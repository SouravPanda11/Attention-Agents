export type ChoiceOpt = { value: string; label: string };

export type TextQuestion =
  | { id: string; type: "number"; label: string; min?: number; max?: number }
  | { id: string; type: "choice"; label: string; options: ChoiceOpt[] }
  | { id: string; type: "text"; label: string; maxLen?: number }
  | { id: string; type: "slider"; label: string; min: number; max: number; step?: number }
  | { id: string; type: "likert"; label: string; options: string[] };

export type TextAnswerValue = string | number | undefined;

export type TextAttentionCheck = {
  id: string;
  label: string;
  type: "choice";
  options: ChoiceOpt[];
  expectedValue: string;
};

export type ImageOption = {
  id: string;
  label: string;
  imageUrl: string;
  alt: string;
};

export type ImageQuestion = {
  id: string;
  label: string;
  options: [ImageOption, ImageOption];
};

export type ImageAttentionCheck = {
  id: string;
  label: string;
  options: [ImageOption, ImageOption];
  expectedOptionId: string;
};

