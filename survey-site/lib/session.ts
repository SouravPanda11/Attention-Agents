import { cookies } from "next/headers";
import crypto from "crypto";

export const SESSION_COOKIE = "sid";

export async function getOrCreateSessionId(): Promise<string> {
  const jar = await cookies(); // ✅ await in Next 16
  const existing = jar.get(SESSION_COOKIE)?.value;
  if (existing) return existing;

  const sid = crypto.randomBytes(16).toString("hex");

  jar.set(SESSION_COOKIE, sid, {
    httpOnly: true,
    sameSite: "lax",
    secure: false, // local dev
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });

  return sid;
}
