import "server-only";
import { describeWorkflow } from "@/lib/benchmark/buildWorkflow";
import { getSurveySamples } from "@/lib/benchmark/sampling";
import { getThemeWorkflowManifest } from "@/lib/benchmark/themeWorkflow";
import { DEFAULT_RUNS_PER_ORDER, OCCURRENCES, ORDER_IDS, SUITE_VERSION, getSampleLayouts, type LayoutMode, type Occurrence } from "@/lib/benchmark/schema";

/** Plan executions; never contact a model or create study records here. */
export function* getRunPlan(options: { repeats?: number; occurrence?: Occurrence; layout?: LayoutMode; includeBaselines?: boolean } = {}) {
  const repeats = options.repeats ?? DEFAULT_RUNS_PER_ORDER;
  if (!Number.isSafeInteger(repeats) || repeats < 1) throw new Error("Repeats must be a positive integer.");
  const baselines = options.includeBaselines === false ? [] : getThemeWorkflowManifest();
  function* expand(workflow: { id: string; sampleId: string; url: string; orderId: string; occurrence: number;
    condition: string; layout: string; orderSeed: string; questionCount: number; pageCount: number }) {
    for (let repeatIndex = 1; repeatIndex <= repeats; repeatIndex++) {
      yield { suiteVersion: SUITE_VERSION, planId: `${workflow.id}-repeat${String(repeatIndex).padStart(3, "0")}`,
        workflowId: workflow.id, sampleId: workflow.sampleId, condition: workflow.condition,
        occurrence: workflow.occurrence, layout: workflow.layout, orderId: workflow.orderId, orderSeed: workflow.orderSeed,
        repeatIndex, questionCount: workflow.questionCount, pageCount: workflow.pageCount, url: workflow.url };
    }
  }
  for (const baseline of baselines) yield* expand(baseline);
  for (const occurrence of options.occurrence ? [options.occurrence] : OCCURRENCES) {
    for (const sample of getSurveySamples(occurrence)) {
      for (const layout of getSampleLayouts(occurrence)) {
        if (options.layout && options.layout !== layout) continue;
        for (const orderId of ORDER_IDS) yield* expand(describeWorkflow(sample, orderId, layout));
      }
    }
  }
}
