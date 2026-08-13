import { NextResponse } from "next/server";
import { NAVIGATION_PAGE_SIZE, getWorkflowManifest } from "@/lib/benchmark/buildWorkflow";
import {
  ATTENTION_CHECKS_PER_BLOCK,
  SUBSTANTIVE_QUESTIONS_PER_BLOCK,
  SUITE_VERSION,
} from "@/lib/benchmark/schema";

export async function GET() {
  const workflows = getWorkflowManifest();
  return NextResponse.json({
    suiteVersion: SUITE_VERSION,
    presentationProfiles: ["standard"],
    workflowConditionCount: 16,
    fixedOrderCount: 3,
    workflowInstanceCount: workflows.length,
    everyWorkflowHasWelcomePage: true,
    substantiveQuestionsPerBlock: SUBSTANTIVE_QUESTIONS_PER_BLOCK,
    attentionChecksPerBlock: ATTENTION_CHECKS_PER_BLOCK,
    renderedQuestionsPerBlock: NAVIGATION_PAGE_SIZE,
    navigationPageSize: NAVIGATION_PAGE_SIZE,
    workflows,
  });
}
