import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SurveyRunner } from "@/components/SurveyRunner";
import { buildWorkflow, getAllWorkflowParams } from "@/lib/benchmark/buildWorkflow";
import {
  isLayoutMode,
  isOccurrence,
  isOrderId,
  isPresentationProfile,
} from "@/lib/benchmark/schema";

type RouteParams = {
  profile: string;
  occurrence: string;
  layout: string;
  order: string;
};

function parseParams(params: RouteParams) {
  const occurrenceMatch = /^o([1-8])$/.exec(params.occurrence);
  const occurrence = occurrenceMatch ? Number(occurrenceMatch[1]) : Number.NaN;
  if (
    !isPresentationProfile(params.profile) ||
    !isOccurrence(occurrence) ||
    !isLayoutMode(params.layout) ||
    !isOrderId(params.order)
  ) {
    return null;
  }
  return { profile: params.profile, occurrence, layout: params.layout, orderId: params.order };
}

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllWorkflowParams();
}

export async function generateMetadata({ params }: { params: Promise<RouteParams> }): Promise<Metadata> {
  const parsed = parseParams(await params);
  return parsed ? { title: `v0 ${parsed.occurrence} ${parsed.layout} ${parsed.orderId} · Survey Benchmark` } : {};
}

export default async function WorkflowPage({ params }: { params: Promise<RouteParams> }) {
  const parsed = parseParams(await params);
  if (!parsed) notFound();
  const workflow = buildWorkflow(parsed.profile, parsed.occurrence, parsed.layout, parsed.orderId);
  return <SurveyRunner workflow={workflow} />;
}
