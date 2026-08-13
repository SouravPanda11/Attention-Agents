import type {
  ChoiceOption,
  ImageChoiceOption,
  ImageSingleSelectQuestion,
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
const DEFAULT_IMAGE_CHOICES = [
  {
    label: "Circular design",
    imageSrc: "/images/circular-design.svg",
    imageAlt: "A blue circular design",
  },
  {
    label: "Square design",
    imageSrc: "/images/square-design.svg",
    imageAlt: "An orange square design",
  },
] as const;

function options(labels: readonly string[]): ChoiceOption[] {
  return labels.map((label, index) => ({ value: `option_${index + 1}`, label }));
}

function imageOptions(
  values: readonly { label: string; imageSrc: string; imageAlt: string }[]
): ImageChoiceOption[] {
  return values.map((value, index) => ({ value: `option_${index + 1}`, ...value }));
}

export const q = {
  radio(id: string, block: number, prompt: string, labels: readonly string[] = DEFAULT_CHOICES): SingleRadioQuestion {
    return { id, kind: "single-radio", block, prompt, required: false, options: options(labels) };
  },

  dropdown(id: string, block: number, prompt: string, labels: readonly string[] = DEFAULT_CHOICES): SingleDropdownQuestion {
    return { id, kind: "single-dropdown", block, prompt, required: false, options: options(labels) };
  },

  singleCheckbox(
    id: string,
    block: number,
    prompt: string,
    labels: readonly string[] = DEFAULT_CHOICES
  ): SingleCheckboxQuestion {
    return {
      id,
      kind: "single-checkbox",
      block,
      prompt,
      required: false,
      options: options(labels),
      minSelections: 1,
      maxSelections: 1,
    };
  },

  multipleCheckbox(
    id: string,
    block: number,
    prompt: string,
    labels: readonly string[] = DEFAULT_CHOICES,
    minSelections = 2,
    maxSelections = 2
  ): MultipleCheckboxQuestion {
    return {
      id,
      kind: "multiple-checkbox",
      block,
      prompt,
      required: false,
      options: options(labels),
      minSelections,
      maxSelections,
    };
  },

  likert(id: string, block: number, prompt: string, labels: readonly string[] = DEFAULT_LIKERT): LikertQuestion {
    return { id, kind: "likert", block, prompt, required: false, options: options(labels) };
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
      required: false,
      requireInteraction: true,
      ...range,
    };
  },

  ranking(id: string, block: number, prompt: string, labels: readonly string[] = DEFAULT_RANKING): RankingQuestion {
    return {
      id,
      kind: "ranking",
      block,
      prompt,
      required: false,
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
    return { id, kind: "numeric", block, prompt, required: false, ...range };
  },

  shortText(
    id: string,
    block: number,
    prompt: string,
    limits: { minLength: number; maxLength: number } = { minLength: 1, maxLength: 160 }
  ): ShortTextQuestion {
    return { id, kind: "short-text", block, prompt, required: false, ...limits };
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
    return { id, kind: "long-text", block, prompt, required: false, ...limits };
  },

  imageSingleSelect(
    id: string,
    block: number,
    prompt: string,
    values: readonly { label: string; imageSrc: string; imageAlt: string }[] = DEFAULT_IMAGE_CHOICES
  ): ImageSingleSelectQuestion {
    return {
      id,
      kind: "image-single-select",
      block,
      prompt,
      required: false,
      options: imageOptions(values),
    };
  },
};
