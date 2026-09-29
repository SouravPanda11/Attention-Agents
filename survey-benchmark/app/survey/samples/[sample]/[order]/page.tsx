import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SurveyRunner } from "@/components/SurveyRunner";
import { buildWorkflow } from "@/lib/benchmark/buildWorkflow";
import { getSurveySample } from "@/lib/benchmark/sampling";
import { isOrderId, SUITE_VERSION } from "@/lib/benchmark/schema";

// Generate on demand instead of pre-rendering the full suite at build time.
export const dynamic = "force-dynamic";
type RouteParams = { sample: string; order: string };

export async function generateMetadata({ params }: { params: Promise<RouteParams> }): Promise<Metadata> {
  const { sample, order } = await params;
  return { title: `${SUITE_VERSION} ${sample} ${order} · Survey Benchmark` };
}

export default async function WorkflowPage({ params }: { params: Promise<RouteParams> }) {
  const { sample, order } = await params;
  if (!getSurveySample(sample) || !isOrderId(order)) notFound();
  return <SurveyRunner workflow={buildWorkflow(sample, order)} />;
}
