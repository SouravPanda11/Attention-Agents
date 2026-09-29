import { getQuestionBank } from "@/lib/benchmark/questions";
import { ATTENTION_CHECK_CONTENT_VERSION, getAttentionCheckInstances } from "@/lib/benchmark/attentionChecks";
import { getOrderSeed, orderQuestions, seededShuffle } from "@/lib/benchmark/ordering";
import { getSurveySample, getSurveySamples, type SurveySample } from "@/lib/benchmark/sampling";
import { MAIN_QUESTION_CONTENT_VERSION } from "@/lib/benchmark/questions/mainQuestionBank";
import {
  OCCURRENCES, ORDER_IDS, ORDERING_VERSION, SUITE_VERSION, getSampleLayouts,
  type LayoutMode, type Occurrence, type OrderId, type Workflow,
} from "@/lib/benchmark/schema";

export function workflowId(sampleId: string, orderId: OrderId, layout: LayoutMode = "navigation") {
  return `${SUITE_VERSION}-standard-${sampleId}${layout === "item" ? "-item" : ""}-${orderId}`;
}

export function workflowUrl(sampleId: string, orderId: OrderId, layout: LayoutMode = "navigation") {
  return `/survey/samples/${sampleId}/${orderId}${layout === "item" ? "/item" : ""}`;
}

export function buildWorkflow(sampleId: string, orderId: OrderId, layout: LayoutMode = "navigation"): Workflow {
  const sample = getSurveySample(sampleId);
  if (!sample) throw new Error(`Unknown survey sample: ${sampleId}`);
  if (!getSampleLayouts(sample.occurrence).includes(layout)) throw new Error(`Unsupported layout for ${sampleId}: ${layout}`);
  const bank = getQuestionBank(sample);
  const substantive = orderQuestions(bank, orderId, sample.id);
  const checks = getAttentionCheckInstances(sample, orderId);
  const seed = getOrderSeed(orderId, sample.id);
  const logicalPages = Array.from({ length: sample.occurrence }, (_, index) => {
    const block = index + 1;
    const ordinary = checks.filter((check) => check.question.block === block && check.role === "rotating");
    const questions = seededShuffle([
      ...substantive.slice(index * 11, (index + 1) * 11),
      ...ordinary.map((check) => check.question),
    ].map((question) => question.id), `${seed}:page-${block}`);
    // Randomize AC positions without disturbing any substantive dependencies.
    const ordinaryIds = new Set(ordinary.map((check) => check.question.id));
    const byId = new Map(ordinary.map((check) => [check.question.id, check.question]));
    let nextSubstantive = index * 11;
    const composed = questions.map((id) => ordinaryIds.has(id)
      ? byId.get(id)!
      : substantive[nextSubstantive++]);
    if (block === sample.occurrence) {
      const fixed = checks.find((check) => check.role === "fixed-penultimate")!;
      composed.splice(composed.length - 1, 0, fixed.question);
    }
    if (ordinary.length !== 2 || ordinary[0].privateId === ordinary[1].privateId ||
        composed.length !== (block === sample.occurrence ? 14 : 13)) {
      throw new Error(`Invalid page allocation for ${sample.id}/${orderId}/${block}.`);
    }
    return { id: `page-${String(block).padStart(2, "0")}`, index, questions: composed };
  });
  const ordered = logicalPages.flatMap((page) => page.questions);
  // Layout only changes page boundaries. Seeds, IDs, content, and positions stay matched.
  const pages = layout === "item" ? [{ id: "page-01", index: 0, questions: ordered }] : logicalPages;
  if (new Set(ordered.map((question) => question.id)).size !== ordered.length) {
    throw new Error(`Repeated question identity in ${sample.id}/${orderId}.`);
  }
  return {
    id: workflowId(sample.id, orderId, layout),
    sampleId: sample.id,
    condition: "attention-horizon",
    selectedThemeIds: sample.themeIds,
    suiteVersion: SUITE_VERSION,
    orderingVersion: ORDERING_VERSION,
    orderSeed: seed,
    hasWelcomePage: true,
    profile: "standard",
    occurrence: sample.occurrence,
    occurrenceId: `o${sample.occurrence}`,
    layout,
    orderId,
    contentVersion: bank.contentVersion,
    attentionCheckContentVersion: ATTENTION_CHECK_CONTENT_VERSION,
    substantiveQuestionCount: substantive.length,
    attentionCheckCount: checks.length,
    renderedQuestionCount: ordered.length,
    questionCount: ordered.length,
    pageCount: pages.length,
    questionCountsPerPage: pages.map((page) => page.questions.length),
    orderedQuestionIds: ordered.map((question) => question.id),
    pages,
  };
}

export function describeWorkflow(sample: SurveySample, orderId: OrderId, layout: LayoutMode = "navigation") {
  return {
    id: workflowId(sample.id, orderId, layout), sampleId: sample.id,
    condition: "attention-horizon" as const,
    url: workflowUrl(sample.id, orderId, layout),
    suiteVersion: SUITE_VERSION, profile: "standard", layout,
    occurrence: sample.occurrence, occurrenceId: `o${sample.occurrence}`,
    selectedThemeIds: sample.themeIds, orderId,
    orderSeed: getOrderSeed(orderId, sample.id), orderingVersion: ORDERING_VERSION,
    hasWelcomePage: true,
    contentVersion: MAIN_QUESTION_CONTENT_VERSION,
    attentionCheckContentVersion: ATTENTION_CHECK_CONTENT_VERSION,
    substantiveQuestionCount: 11 * sample.occurrence,
    attentionCheckCount: 2 * sample.occurrence + 1,
    renderedQuestionCount: 13 * sample.occurrence + 1,
    questionCount: 13 * sample.occurrence + 1,
    pageCount: layout === "item" ? 1 : sample.occurrence,
    questionCountsPerPage: layout === "item" ? [13 * sample.occurrence + 1]
      : Array.from({ length: sample.occurrence }, (_, index) => index + 1 === sample.occurrence ? 14 : 13),
  };
}

export type ManifestOptions = {
  occurrence?: Occurrence;
  sampleId?: string;
  orderId?: OrderId;
  layout?: LayoutMode;
  offset?: number;
  limit?: number;
};

/** Paginate descriptors before materializing layouts; never build the full suite per request. */
export function getWorkflowManifest(options: ManifestOptions = {}) {
  const selectedSample = options.sampleId ? getSurveySample(options.sampleId) : undefined;
  const samples = options.sampleId
    ? selectedSample && (!options.occurrence || selectedSample.occurrence === options.occurrence) ? [selectedSample] : []
    : (options.occurrence ? [options.occurrence] : OCCURRENCES).flatMap((occurrence) => [...getSurveySamples(occurrence)]);
  const orders = options.orderId ? [options.orderId] : ORDER_IDS;
  const entries = samples.flatMap((sample) => getSampleLayouts(sample.occurrence)
    .filter((layout) => !options.layout || options.layout === layout)
    .flatMap((layout) => orders.map((orderId) => ({ sample, layout, orderId }))));
  const total = entries.length;
  const offset = options.offset ?? 0;
  const limit = options.limit ?? 100;
  const workflows = Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) => {
    const { sample, layout, orderId } = entries[offset + index];
    const workflow = buildWorkflow(sample.id, orderId, layout);
    return { ...describeWorkflow(sample, orderId, layout), orderedQuestionIds: workflow.orderedQuestionIds,
      pageQuestionIds: workflow.pages.map((page) => page.questions.map((question) => question.id)) };
  });
  return { workflows, pagination: { offset, limit, total,
    nextOffset: offset + workflows.length < total ? offset + workflows.length : null } };
}
