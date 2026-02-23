import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { db } from "@/lib/db";
import { getOrCreateSessionId } from "@/lib/session";

const TEXT_ATTENTION_EXPECTED_VALUE = "somewhat";
const IMAGE_ATTENTION_EXPECTED_OPTION_ID = "a";

type SubmitBody = {
  text_answers?: Record<string, string | number | undefined>;
  text_attention_value?: string;
  image_answers?: Record<string, string>;
  captcha_input?: string;
  image_attention_choice?: string;
};

function logEvent(sessionId: string, eventType: string, payload: unknown) {
  db.prepare("INSERT INTO events (ts, session_id, event_type, payload) VALUES (?, ?, ?, ?)").run(
    new Date().toISOString(),
    sessionId,
    eventType,
    JSON.stringify(payload ?? {})
  );
}

export async function POST(req: Request) {
  const sid = await getOrCreateSessionId();
  const jar = await cookies();
  const body = (await req.json().catch(() => null)) as SubmitBody | null;

  if (!body) {
    logEvent(sid, "submit_error", { reason: "invalid_json" });
    return NextResponse.json({ ok: false, error: "invalid_json" }, { status: 400 });
  }

  const textAnswers = body.text_answers ?? {};
  const imageAnswers = body.image_answers ?? {};
  const textAttentionValue = String(body.text_attention_value ?? "");
  const captchaInput = String(body.captcha_input ?? "");
  const imageAttentionChoice = String(body.image_attention_choice ?? "");
  const expectedCaptcha = String(jar.get("captcha_code")?.value ?? "");

  const textAttentionOk = textAttentionValue === TEXT_ATTENTION_EXPECTED_VALUE;
  const captchaOk = Boolean(expectedCaptcha) && captchaInput.trim().toUpperCase() === expectedCaptcha;
  const imageAttentionOk = imageAttentionChoice === IMAGE_ATTENTION_EXPECTED_OPTION_ID;

  const ok = Boolean(textAttentionOk && captchaOk && imageAttentionOk);

  db.prepare(
    `INSERT INTO submissions (
      ts,
      session_id,
      text_answers,
      text_attention_value,
      text_attention_ok,
      image_answers,
      captcha_input,
      captcha_ok,
      image_attention_choice,
      image_attention_ok,
      overall_ok
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    new Date().toISOString(),
    sid,
    JSON.stringify(textAnswers),
    textAttentionValue,
    textAttentionOk ? 1 : 0,
    JSON.stringify(imageAnswers),
    captchaInput,
    captchaOk ? 1 : 0,
    imageAttentionChoice,
    imageAttentionOk ? 1 : 0,
    ok ? 1 : 0
  );

  logEvent(sid, "submit", {
    ok,
    textAttentionOk,
    captchaOk,
    imageAttentionOk,
    text_answer_count: Object.keys(textAnswers).length,
    image_answer_count: Object.keys(imageAnswers).length,
  });

  return NextResponse.json({ ok, textAttentionOk, captchaOk, imageAttentionOk });
}
