import { NextResponse } from "next/server";
import { getWorkflowManifest } from "@/lib/benchmark/buildWorkflow";
import { getThemeWorkflowManifest } from "@/lib/benchmark/themeWorkflow";
import { getSamplingSummary, getSurveySample } from "@/lib/benchmark/sampling";
import { THEMES } from "@/lib/benchmark/questions/mainQuestionBank";
import { DEFAULT_RUNS_PER_ORDER, ORDERING_VERSION, SUITE_VERSION, isOccurrence, isOrderId, isLayoutMode } from "@/lib/benchmark/schema";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;
  const occurrenceValue = params.get("occurrence");
  const occurrence = occurrenceValue === null ? undefined : Number(occurrenceValue);
  const orderId = params.get("order") ?? undefined;
  const sampleId = params.get("sampleId") ?? undefined;
  const layout = params.get("layout") ?? undefined;
  const offset = Number(params.get("offset") ?? 0);
  const limit = Number(params.get("limit") ?? 100);
  if ((occurrence !== undefined && !isOccurrence(occurrence)) ||
      (orderId !== undefined && !isOrderId(orderId)) ||
      (sampleId !== undefined && !getSurveySample(sampleId)) ||
      (layout !== undefined && !isLayoutMode(layout)) ||
      !Number.isSafeInteger(offset) || offset < 0 ||
      !Number.isSafeInteger(limit) || limit < 1 || limit > 300) {
    return NextResponse.json({ error: "invalid_manifest_filters" }, { status: 400 });
  }
  const manifest = getWorkflowManifest({
    occurrence: occurrence !== undefined && isOccurrence(occurrence) ? occurrence : undefined,
    orderId: orderId && isOrderId(orderId) ? orderId : undefined,
    layout: layout && isLayoutMode(layout) ? layout : undefined,
    sampleId, offset, limit,
  });
  const sampling = getSamplingSummary();
  const sampleCount = sampling.reduce((sum, entry) => sum + entry.sampleCount, 0);
  const workflowCount = sampling.reduce((sum, entry) => sum + entry.workflowInstanceCount, 0);
  const themeWorkflows = getThemeWorkflowManifest();
  const nextParams = new URLSearchParams(params);
  if (manifest.pagination.nextOffset !== null) nextParams.set("offset", String(manifest.pagination.nextOffset));
  return NextResponse.json({
    suiteVersion: SUITE_VERSION,
    orderingVersion: ORDERING_VERSION,
    presentationProfiles: ["standard"],
    layouts: ["navigation", "item"],
    contentSampleCount: sampleCount,
    baselineSampleCount: THEMES.length,
    totalContentSampleCount: sampleCount + THEMES.length,
    fixedOrderCount: 3,
    defaultRunsPerOrder: DEFAULT_RUNS_PER_ORDER,
    workflowInstanceCount: workflowCount,
    baselineWorkflowInstanceCount: themeWorkflows.length,
    totalWorkflowInstanceCount: workflowCount + themeWorkflows.length,
    defaultRunsPerModel: (workflowCount + themeWorkflows.length) * DEFAULT_RUNS_PER_ORDER,
    everyWorkflowHasWelcomePage: true,
    fixedAttentionChecksPerSurvey: 1,
    navigation: { substantiveQuestionsPerPage: 11, ordinaryAttentionChecksPerPage: 2,
      regularPageSize: 13, finalPageSize: 14 },
    item: { minimumOccurrence: 2, pageCount: 1, questionCountFormula: "13 * occurrence + 1" },
    sampling,
    ...manifest,
    next: manifest.pagination.nextOffset === null ? null : `/api/manifest?${nextParams}`,
    themes: THEMES,
    themeWorkflows,
  });
}
