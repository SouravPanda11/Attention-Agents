import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getOrCreateSessionId } from "@/lib/session";

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
    0,
    JSON.stringify(imageAnswers),
    captchaInput,
    0,
    imageAttentionChoice,
    0,
    0
  );

  logEvent(sid, "submit", {
    accepted: true,
    text_answer_count: Object.keys(textAnswers).length,
    image_answer_count: Object.keys(imageAnswers).length,
  });

  return NextResponse.json({ ok: true, accepted: true });
}
