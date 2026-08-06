import type {
  ChoiceOption,
  LikertQuestion,
  LongTextQuestion,
  MultipleCheckboxQuestion,
  NumericQuestion,
  RankingQuestion,
  ShortTextQuestion,
  SingleCheckboxQuestion,
  SingleDropdownQuestion,
  SingleRadioQuestion,
  SliderQuestion,
} from "@/lib/benchmark/schema";

const DEFAULT_CHOICES = ["Option A", "Option B", "Option C", "Option D"] as const;
const DEFAULT_LIKERT = [
  "Strongly disagree",
  "Disagree",
  "Neutral",
  "Agree",
  "Strongly agree",
] as const;
const DEFAULT_RANKING = ["Item A", "Item B", "Item C", "Item D"] as const;

function options(labels: readonly string[]): ChoiceOption[] {
  return labels.map((label, index) => ({ value: `option_${index + 1}`, label }));
}

export const q = {
  radio(id: string, block: number, prompt: string, labels = DEFAULT_CHOICES): SingleRadioQuestion {
    return { id, kind: "single-radio", block, prompt, required: true, options: options(labels) };
  },

  dropdown(id: string, block: number, prompt: string, labels = DEFAULT_CHOICES): SingleDropdownQuestion {
    return { id, kind: "single-dropdown", block, prompt, required: true, options: options(labels) };
  },

  singleCheckbox(
    id: string,
    block: number,
    prompt: string,
    labels = DEFAULT_CHOICES
  ): SingleCheckboxQuestion {
    return {
      id,
      kind: "single-checkbox",
      block,
      prompt,
      required: true,
      options: options(labels),
      minSelections: 1,
      maxSelections: 1,
    };
  },

  multipleCheckbox(
    id: string,
    block: number,
    prompt: string,
    labels = DEFAULT_CHOICES,
    minSelections = 2,
    maxSelections = 2
  ): MultipleCheckboxQuestion {
    return {
      id,
      kind: "multiple-checkbox",
      block,
      prompt,
      required: true,
      options: options(labels),
      minSelections,
      maxSelections,
    };
  },

  likert(id: string, block: number, prompt: string, labels = DEFAULT_LIKERT): LikertQuestion {
    return { id, kind: "likert", block, prompt, required: true, options: options(labels) };
  },

  slider(
    id: string,
    block: number,
    prompt: string,
    range: { min: number; max: number; step: number } = { min: 0, max: 10, step: 1 }
  ): SliderQuestion {
    return {
      id,
      kind: "slider",
      block,
      prompt,
      required: true,
      requireInteraction: true,
      ...range,
    };
  },

  ranking(id: string, block: number, prompt: string, labels = DEFAULT_RANKING): RankingQuestion {
    return {
      id,
      kind: "ranking",
      block,
      prompt,
      required: true,
      requireInteraction: true,
      items: options(labels),
    };
  },

  numeric(
    id: string,
    block: number,
    prompt: string,
    range: { min?: number; max?: number; step?: number } = { min: 0, max: 100, step: 1 }
  ): NumericQuestion {
    return { id, kind: "numeric", block, prompt, required: true, ...range };
  },

  shortText(
    id: string,
    block: number,
    prompt: string,
    limits: { minLength: number; maxLength: number } = { minLength: 1, maxLength: 160 }
  ): ShortTextQuestion {
    return { id, kind: "short-text", block, prompt, required: true, ...limits };
  },

  longText(
    id: string,
    block: number,
    prompt: string,
    limits: { minLength: number; maxLength: number; rows: number } = {
      minLength: 1,
      maxLength: 1000,
      rows: 6,
    }
  ): LongTextQuestion {
    return { id, kind: "long-text", block, prompt, required: true, ...limits };
  },
};
