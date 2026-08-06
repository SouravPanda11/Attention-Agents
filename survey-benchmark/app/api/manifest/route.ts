import { NextResponse } from "next/server";
import { getWorkflowManifest } from "@/lib/benchmark/buildWorkflow";
import { SUITE_VERSION } from "@/lib/benchmark/schema";

export async function GET() {
  const workflows = getWorkflowManifest();
  return NextResponse.json({
    suiteVersion: SUITE_VERSION,
    presentationProfiles: ["standard"],
    workflowConditionCount: 16,
    fixedOrderCount: 5,
    workflowInstanceCount: workflows.length,
    navigationPageSize: 10,
    workflows,
  });
}
