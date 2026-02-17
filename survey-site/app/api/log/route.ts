import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getOrCreateSessionId } from "@/lib/session";

export async function POST(req: Request) {
  const sid = await getOrCreateSessionId(); // ✅
  const body = await req.json().catch(() => ({}));

  const ts = new Date().toISOString();
  const eventType = String(body?.event_type ?? "unknown");
  const payload = JSON.stringify(body?.payload ?? {});

  db.prepare(
    "INSERT INTO events (ts, session_id, event_type, payload) VALUES (?, ?, ?, ?)"
  ).run(ts, sid, eventType, payload);

  return NextResponse.json({ ok: true, session_id: sid });
}
