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

const IMAGE_QUESTIONS_BASE: ImageQuestion[] = [
    {
      id: "image_q1",
      label: "1) Which image is more related to basketball?",
      options: [
        {
          id: "a",
          label: "Basketball",
          imageUrl: "/Basketball.jpg",
          alt: "Basketball",
        },
        {
          id: "b",
          label: "Football",
          imageUrl: "/Football.jpg",
          alt: "Football",
        },
      ],
    },
    {
      id: "image_q2",
      label: "2) Which image is more related to gymnastics?",
      options: [
        {
          id: "a",
          label: "Gymnastics",
          imageUrl: "/Gymnastics.webp",
          alt: "gymnastics",
        },
        {
          id: "b",
          label: "Cycling",
          imageUrl: "/Cycling.jpg",
          alt: "cycling",
        },
      ],
    },
    {
      id: "image_q3",
      label: "3) Which image is more related to badminton?",
      options: [
        {
          id: "a",
          label: "Badminton",
          imageUrl: "/Badminton.jpg",
          alt: "Badminton",
        },
        {
          id: "b",
          label: "Tennis",
          imageUrl: "/Tennis.jpg",
          alt: "Tennis",
        },
      ],
    },
    {
      id: "image_q4",
      label: "4) Which image is more related to rowing?",
      options: [
        {
          id: "a",
          label: "Rowing",
          imageUrl: "/Rowing.webp",
          alt: "Rowing",
        },
        {
          id: "b",
          label: "Boat",
          imageUrl: "/Boat.jpg",
          alt: "Boat",
        },
      ],
    },
  ];

export const imageAttentionCheck: ImageAttentionCheck = {
  id: "attention_image_mid",
  label: "Select the soccer ball.",
  options: [
    {
      id: "a",
      label: "Soccer",
      imageUrl: "/Soccer.jpg",
      alt: "Soccer",
    },
    {
      id: "b",
      label: "Golf",
      imageUrl: "/golf.jpeg",
      alt: "Golf",
    },
  ],
};

export function getImageSurveyQuestions(): ImageQuestion[] {
  return IMAGE_QUESTIONS_BASE;
}

export function getRandomizedImageSurvey(): RandomizedImageSurvey {
  const randomizedQuestions: ImageQuestion[] = IMAGE_QUESTIONS_BASE.map((q) => {
    const swapped = maybeSwapTwo([q.options[0], q.options[1]]);
    return { ...q, options: swapped };
  });

  const randomizedAttentionOptions = maybeSwapTwo([
    imageAttentionCheck.options[0],
    imageAttentionCheck.options[1],
  ]);
  const randomizedAttention: ImageAttentionCheck = {
    ...imageAttentionCheck,
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
