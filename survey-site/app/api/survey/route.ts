import { NextResponse } from "next/server";
import { getOrCreateSessionId } from "@/lib/session";
import { createCaptchaCode } from "@/lib/survey";
import { getTextSurveyQuestions, textAttentionCheck } from "@/lib/surveys/textSurvey";
import { getImageSurveyQuestions, imageAttentionCheck } from "@/lib/surveys/imageSurvey";

export async function GET() {
  const sid = await getOrCreateSessionId();
  const captchaCode = createCaptchaCode();

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
      questions: getImageSurveyQuestions(),
      captcha_insert_after: 2,
      image_attention_insert_after: 3,
      captcha: {
        id: "captcha_mid",
        label: "Type the code in captcha as shown.",
        display_code: captchaCode,
      },
      image_attention: {
        id: imageAttentionCheck.id,
        label: imageAttentionCheck.label,
        options: imageAttentionCheck.options,
      },
    },
  });

  response.cookies.set("captcha_code", captchaCode, {
    httpOnly: true,
    sameSite: "lax",
    secure: false,
    path: "/",
    maxAge: 60 * 30,
  });

  return response;
}
