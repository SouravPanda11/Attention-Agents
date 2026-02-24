import { NextResponse } from "next/server";
import { getOrCreateSessionId } from "@/lib/session";
import { createCaptchaCode } from "@/lib/survey";
import { getTextSurveyQuestions, textAttentionCheck } from "@/lib/surveys/textSurvey";
import { getRandomizedImageSurveyV1 } from "@/lib/surveys/imageSurvey_v1";
import { db } from "@/lib/db";

export async function GET() {
  const sid = await getOrCreateSessionId();
  const captchaCode = createCaptchaCode();
  const randomizedImage = getRandomizedImageSurveyV1();
  const imageAttentionPublic = {
    id: randomizedImage.image_attention.id,
    label: randomizedImage.image_attention.label,
    options: randomizedImage.image_attention.options,
  };

  const response = NextResponse.json({
    session_id: sid,
    text: {
      questions: getTextSurveyQuestions(),
      attention_insert_after: 4,
      attention: {
        id: textAttentionCheck.id,
        label: textAttentionCheck.label,
        options: textAttentionCheck.options,
      },
    },
    image: {
      questions: randomizedImage.questions,
      captcha_insert_after: 2,
      image_attention_insert_after: 3,
      captcha: {
        id: "captcha_mid",
        label: "Type the code in captcha as shown.",
        display_code: captchaCode,
      },
      image_attention: imageAttentionPublic,
      layout_trace: randomizedImage.layout_trace,
    },
  });

  response.cookies.set("captcha_code_v1", captchaCode, {
    httpOnly: true,
    sameSite: "lax",
    secure: false,
    path: "/",
    maxAge: 60 * 30,
  });

  db.prepare("INSERT INTO events (ts, session_id, event_type, payload) VALUES (?, ?, ?, ?)").run(
    new Date().toISOString(),
    sid,
    "captcha_issued_v1",
    JSON.stringify({ captcha_code: captchaCode, survey_version: "survey_v1" })
  );
  db.prepare("INSERT INTO events (ts, session_id, event_type, payload) VALUES (?, ?, ?, ?)").run(
    new Date().toISOString(),
    sid,
    "image_layout_issued_v1",
    JSON.stringify({
      survey_version: "survey_v1",
      layout_trace: randomizedImage.layout_trace,
    })
  );

  return response;
}
