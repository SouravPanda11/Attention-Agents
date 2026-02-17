import type { ImageAttentionCheck, ImageQuestion } from "@/lib/surveys/types";

export function getImageSurveyQuestions(): ImageQuestion[] {
  return [
    {
      id: "image_q1",
      label: "1) Which image is more related to basketball?",
      options: [
        {
          id: "a",
          label: "Basketball game",
          imageUrl: "/Basketball.jpg",
          alt: "Basketball close-up",
        },
        {
          id: "b",
          label: "Swimming race",
          imageUrl: "/Football.jpg",
          alt: "Swimmer in a pool",
        },
      ],
    },
    {
      id: "image_q2",
      label: "2) Which image is more related to gymnastics?",
      options: [
        {
          id: "a",
          label: "Gymnastics rings",
          imageUrl: "/Gymnastics.webp",
          alt: "Athlete on gymnastics rings",
        },
        {
          id: "b",
          label: "Cycling sprint",
          imageUrl: "/Cycling.jpg",
          alt: "Track cyclist racing",
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
          alt: "Badminton player",
        },
        {
          id: "b",
          label: "Tennis",
          imageUrl: "/Tennis.jpg",
          alt: "Tennis player",
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
          alt: "Rowing team racing",
        },
        {
          id: "b",
          label: "Boat",
          imageUrl: "/Boat.jpg",
          alt: "Boat on water",
        },
      ],
    },
  ];
}

export const imageAttentionCheck: ImageAttentionCheck = {
  id: "attention_image_mid",
  label: "Select the soccer ball.",
  expectedOptionId: "a",
  options: [
    {
      id: "a",
      label: "Soccer ball",
      imageUrl: "/Soccer.jpg",
      alt: "Soccer ball",
    },
    {
      id: "b",
      label: "Golf",
      imageUrl: "/golf.jpeg",
      alt: "Golf scene",
    },
  ],
};
