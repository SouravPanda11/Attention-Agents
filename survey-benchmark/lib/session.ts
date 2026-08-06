import crypto from "crypto";
import { cookies } from "next/headers";

const SESSION_COOKIE = "survey_benchmark_sid";

export async function getOrCreateSessionId(): Promise<string> {
  const jar = await cookies();
  const existing = jar.get(SESSION_COOKIE)?.value;
  if (existing) return existing;

  const sessionId = crypto.randomBytes(16).toString("hex");
  jar.set(SESSION_COOKIE, sessionId, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.SURVEY_COOKIE_SECURE === "true",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  return sessionId;
}
