import { NextResponse } from "next/server";
import { getDatabase } from "@/lib/db";
import { getOrCreateSessionId } from "@/lib/session";

export const runtime = "nodejs";

type EventBody = {
  runId?: unknown;
  workflowId?: unknown;
  eventType?: unknown;
  pageIndex?: unknown;
  payload?: unknown;
};

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as EventBody | null;
  if (
    !body ||
    typeof body.runId !== "string" ||
    typeof body.workflowId !== "string" ||
    typeof body.eventType !== "string" ||
    body.runId.length > 128 ||
    body.workflowId.length > 160 ||
    body.eventType.length > 80
  ) {
    return NextResponse.json({ ok: false, error: "invalid_event" }, { status: 400 });
  }

  const pageIndex = typeof body.pageIndex === "number" && Number.isInteger(body.pageIndex) ? body.pageIndex : null;
  const sessionId = await getOrCreateSessionId();
  getDatabase()
    .prepare(
      `INSERT INTO events (ts, session_id, run_id, workflow_id, event_type, page_index, payload)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      new Date().toISOString(),
      sessionId,
      body.runId,
      body.workflowId,
      body.eventType,
      pageIndex,
      JSON.stringify(body.payload ?? {})
    );

  return NextResponse.json({ ok: true, sessionId });
}
