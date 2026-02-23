import type { ImageAttentionCheck, ImageQuestion } from "@/lib/surveys/types";

type ImageLayoutTraceItem = {
  question_id: string;
  option_order: string[];
};

type ImageAttentionLayoutTrace = {
  question_id: string;
  option_order: string[];
};

type RandomizedImageSurvey = {
  questions: ImageQuestion[];
  image_attention: ImageAttentionCheck;
  layout_trace: {
    questions: ImageLayoutTraceItem[];
    image_attention: ImageAttentionLayoutTrace;
  };
};

function maybeSwapTwo<T>(pair: [T, T]): [T, T] {
  if (Math.random() < 0.5) return [pair[1], pair[0]];
  return pair;
}

const IMAGE_QUESTIONS_BASE_V1: ImageQuestion[] = [
  {
    id: "image_q1",
    label: "1) Which image is more related to basketball?",
    options: [
      {
        id: "q1_a",
        label: "A",
        imageUrl: "/survey_v1/q1_a.jpg",
        alt: "option_a",
      },
      {
        id: "q1_b",
        label: "B",
        imageUrl: "/survey_v1/q1_b.jpg",
        alt: "option_b",
      },
    ],
  },
  {
    id: "image_q2",
    label: "2) Which image is more related to gymnastics?",
    options: [
      {
        id: "q2_a",
        label: "A",
        imageUrl: "/survey_v1/q2_a.webp",
        alt: "option_a",
      },
      {
        id: "q2_b",
        label: "B",
        imageUrl: "/survey_v1/q2_b.jpg",
        alt: "option_b",
      },
    ],
  },
  {
    id: "image_q3",
    label: "3) Which image is more related to badminton?",
    options: [
      {
        id: "q3_a",
        label: "A",
        imageUrl: "/survey_v1/q3_a.jpg",
        alt: "option_a",
      },
      {
        id: "q3_b",
        label: "B",
        imageUrl: "/survey_v1/q3_b.jpg",
        alt: "option_b",
      },
    ],
  },
  {
    id: "image_q4",
    label: "4) Which image is more related to rowing?",
    options: [
      {
        id: "q4_a",
        label: "A",
        imageUrl: "/survey_v1/q4_a.webp",
        alt: "option_a",
      },
      {
        id: "q4_b",
        label: "B",
        imageUrl: "/survey_v1/q4_b.jpg",
        alt: "option_b",
      },
    ],
  },
];

export const imageAttentionCheckV1: ImageAttentionCheck = {
  id: "attention_image_mid",
  label: "Select the soccer ball.",
  options: [
    {
      id: "attn_a",
      label: "A",
      imageUrl: "/survey_v1/attn_a.jpg",
      alt: "option_a",
    },
    {
      id: "attn_b",
      label: "B",
      imageUrl: "/survey_v1/attn_b.jpeg",
      alt: "option_b",
    },
  ],
};

export function getImageSurveyQuestionsV1(): ImageQuestion[] {
  return IMAGE_QUESTIONS_BASE_V1;
}

export function getRandomizedImageSurveyV1(): RandomizedImageSurvey {
  const randomizedQuestions: ImageQuestion[] = IMAGE_QUESTIONS_BASE_V1.map((q) => {
    const swapped = maybeSwapTwo([q.options[0], q.options[1]]);
    return { ...q, options: swapped };
  });

  const randomizedAttentionOptions = maybeSwapTwo([
    imageAttentionCheckV1.options[0],
    imageAttentionCheckV1.options[1],
  ]);
  const randomizedAttention: ImageAttentionCheck = {
    ...imageAttentionCheckV1,
    options: randomizedAttentionOptions,
  };

  const layoutTraceQuestions: ImageLayoutTraceItem[] = randomizedQuestions.map((q) => ({
    question_id: q.id,
    option_order: q.options.map((opt) => opt.id),
  }));

  const attentionTrace: ImageAttentionLayoutTrace = {
    question_id: randomizedAttention.id,
    option_order: randomizedAttention.options.map((opt) => opt.id),
  };

  return {
    questions: randomizedQuestions,
    image_attention: randomizedAttention,
    layout_trace: {
      questions: layoutTraceQuestions,
      image_attention: attentionTrace,
    },
  };
}
