import { getQuestionBank } from "@/lib/benchmark/questions";
import { getOrderSeed, orderQuestions } from "@/lib/benchmark/ordering";
import {
  LAYOUT_MODES,
  OCCURRENCES,
  ORDER_IDS,
  PRESENTATION_PROFILES,
  SUITE_VERSION,
  type LayoutMode,
  type Occurrence,
  type OrderId,
  type PresentationProfile,
  type SurveyQuestion,
  type Workflow,
} from "@/lib/benchmark/schema";

export const NAVIGATION_PAGE_SIZE = 10 as const;

function chunkQuestions(questions: readonly SurveyQuestion[], size: number): SurveyQuestion[][] {
  const chunks: SurveyQuestion[][] = [];
  for (let index = 0; index < questions.length; index += size) {
    chunks.push(questions.slice(index, index + size));
  }
  return chunks;
}

export function buildWorkflow(
  profile: PresentationProfile,
  occurrence: Occurrence,
  layout: LayoutMode,
  orderId: OrderId
): Workflow {
  const bank = getQuestionBank(occurrence);
  const ordered = orderQuestions(bank, orderId);
  const pageQuestions = layout === "item" ? [ordered] : chunkQuestions(ordered, NAVIGATION_PAGE_SIZE);
  const pages = pageQuestions.map((questions, index) => ({
    id: `page-${String(index + 1).padStart(2, "0")}`,
    index,
    questions,
  }));

  const workflow: Workflow = {
    id: `${SUITE_VERSION}-${profile}-o${occurrence}-${layout}-${orderId}`,
    suiteVersion: SUITE_VERSION,
    profile,
    occurrence,
    occurrenceId: `o${occurrence}`,
    layout,
    orderId,
    contentVersion: bank.contentVersion,
    questionCount: ordered.length,
    pageCount: pages.length,
    questionsPerNavigationPage: NAVIGATION_PAGE_SIZE,
    orderedQuestionIds: ordered.map((question) => question.id),
    pages,
  };

  const flattened = workflow.pages.flatMap((page) => page.questions.map((question) => question.id));
  if (flattened.join("|") !== workflow.orderedQuestionIds.join("|")) {
    throw new Error(`[workflow validation] Page layout changed the order for ${workflow.id}.`);
  }
  if (layout === "item" && workflow.pageCount !== 1) {
    throw new Error(`[workflow validation] Item workflow ${workflow.id} must contain one page.`);
  }
  if (layout === "navigation" && workflow.pages.some((page) => page.questions.length !== NAVIGATION_PAGE_SIZE)) {
    throw new Error(`[workflow validation] Navigation workflow ${workflow.id} must contain ten questions per page.`);
  }

  return workflow;
}

export function getAllWorkflowParams() {
  return PRESENTATION_PROFILES.flatMap((profile) =>
    OCCURRENCES.flatMap((occurrence) =>
      LAYOUT_MODES.flatMap((layout) =>
        ORDER_IDS.map((orderId) => ({
          profile,
          occurrence: `o${occurrence}`,
          layout,
          order: orderId,
        }))
      )
    )
  );
}

export function getWorkflowManifest() {
  return PRESENTATION_PROFILES.flatMap((profile) =>
    OCCURRENCES.flatMap((occurrence) =>
      LAYOUT_MODES.flatMap((layout) =>
        ORDER_IDS.map((orderId) => {
          const workflow = buildWorkflow(profile, occurrence, layout, orderId);
          return {
            id: workflow.id,
            url: `/survey/${profile}/o${occurrence}/${layout}/${orderId}`,
            profile,
            occurrence,
            layout,
            orderId,
            orderSeed: getOrderSeed(orderId),
            contentVersion: workflow.contentVersion,
            questionCount: workflow.questionCount,
            pageCount: workflow.pageCount,
            orderedQuestionIds: workflow.orderedQuestionIds,
          };
        })
      )
    )
  );
}
