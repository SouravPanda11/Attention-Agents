import { getQuestionBank } from "@/lib/benchmark/questions";
import {
  ATTENTION_CHECK_CONTENT_VERSION,
  getAttentionCheckInstances,
  type AttentionCheckInstance,
} from "@/lib/benchmark/attentionChecks";
import { getOrderSeed, orderQuestions, seededShuffle } from "@/lib/benchmark/ordering";
import {
  LAYOUT_MODES,
  OCCURRENCES,
  ORDER_IDS,
  PRESENTATION_PROFILES,
  ATTENTION_CHECKS_PER_BLOCK,
  RENDERED_QUESTIONS_PER_BLOCK,
  SUBSTANTIVE_QUESTIONS_PER_BLOCK,
  SUITE_VERSION,
  type LayoutMode,
  type Occurrence,
  type OrderId,
  type PresentationProfile,
  type SurveyQuestion,
  type Workflow,
} from "@/lib/benchmark/schema";

export const NAVIGATION_PAGE_SIZE = RENDERED_QUESTIONS_PER_BLOCK;

function rotatingPositions(orderId: OrderId, block: number, count: number): number[] {
  const candidates = Array.from({ length: NAVIGATION_PAGE_SIZE }, (_, index) => index);
  return seededShuffle(
    candidates,
    `${getOrderSeed(orderId)}:attention-positions:block-${block}`
  )
    .slice(0, count)
    .sort((left, right) => left - right);
}

function placeBlockQuestions(
  substantive: readonly SurveyQuestion[],
  checks: readonly AttentionCheckInstance[],
  orderId: OrderId,
  block: number,
  isFinalBlock: boolean
): SurveyQuestion[] {
  const fixed = checks.find((check) => check.placement === "fixed-penultimate");
  const rotating = checks.filter((check) => check.placement === "seeded");
  const positions = new Map<number, SurveyQuestion>();

  if (isFinalBlock) {
    if (!fixed || rotating.length !== 1) {
      throw new Error(
        `[workflow validation] Final block ${block} must contain one rotating and one fixed attention check.`
      );
    }
    // Reserve index 11: item 12 of the 13-item final block. Keep item 13 substantive.
    const rotatingPosition = seededShuffle(
      Array.from({ length: 11 }, (_, index) => index),
      `${getOrderSeed(orderId)}:attention-positions:block-${block}:terminal`
    )[0];
    positions.set(rotatingPosition, rotating[0].question);
    positions.set(NAVIGATION_PAGE_SIZE - 2, fixed.question);
  } else {
    if (fixed || rotating.length !== 2) {
      throw new Error(
        `[workflow validation] Non-final block ${block} must contain two rotating attention checks.`
      );
    }
    rotatingPositions(orderId, block, rotating.length).forEach((position, index) => {
      positions.set(position, rotating[index].question);
    });
  }

  let substantiveIndex = 0;
  const composed = Array.from({ length: NAVIGATION_PAGE_SIZE }, (_, position) => {
    const attentionCheck = positions.get(position);
    if (attentionCheck) return attentionCheck;
    const question = substantive[substantiveIndex];
    substantiveIndex += 1;
    if (!question) throw new Error(`[workflow validation] Block ${block} ran out of substantive questions.`);
    return question;
  });

  if (substantiveIndex !== substantive.length) {
    throw new Error(`[workflow validation] Block ${block} did not render every substantive question.`);
  }
  return composed;
}

function composeBlocks(
  orderedSubstantive: readonly SurveyQuestion[],
  occurrence: Occurrence,
  orderId: OrderId
): SurveyQuestion[][] {
  const attentionChecks = getAttentionCheckInstances(occurrence, orderId);
  return Array.from({ length: occurrence }, (_, index) => {
    const block = index + 1;
    const substantive = orderedSubstantive.filter((question) => question.block === block);
    if (substantive.length !== SUBSTANTIVE_QUESTIONS_PER_BLOCK) {
      throw new Error(
        `[workflow validation] Block ${block} must contain ${SUBSTANTIVE_QUESTIONS_PER_BLOCK} substantive questions.`
      );
    }
    const blockChecks = attentionChecks.filter((instance) => instance.question.block === block);
    if (blockChecks.length !== ATTENTION_CHECKS_PER_BLOCK) {
      throw new Error(
        `[workflow validation] Block ${block} must contain ${ATTENTION_CHECKS_PER_BLOCK} attention checks.`
      );
    }
    return placeBlockQuestions(substantive, blockChecks, orderId, block, block === occurrence);
  });
}

export function buildWorkflow(
  profile: PresentationProfile,
  occurrence: Occurrence,
  layout: LayoutMode,
  orderId: OrderId
): Workflow {
  const bank = getQuestionBank(occurrence, orderId);
  const orderedSubstantive = orderQuestions(bank, orderId);
  const logicalBlocks = composeBlocks(orderedSubstantive, occurrence, orderId);
  const ordered = logicalBlocks.flat();
  const pageQuestions = layout === "item" ? [ordered] : logicalBlocks;
  const pages = pageQuestions.map((questions, index) => ({
    id: `page-${String(index + 1).padStart(2, "0")}`,
    index,
    questions,
  }));

  const workflow: Workflow = {
    id: `${SUITE_VERSION}-${profile}-o${occurrence}-${layout}-${orderId}`,
    suiteVersion: SUITE_VERSION,
    hasWelcomePage: true,
    profile,
    occurrence,
    occurrenceId: `o${occurrence}`,
    layout,
    orderId,
    contentVersion: bank.contentVersion,
    attentionCheckContentVersion: ATTENTION_CHECK_CONTENT_VERSION,
    substantiveQuestionCount: orderedSubstantive.length,
    attentionCheckCount: occurrence * ATTENTION_CHECKS_PER_BLOCK,
    renderedQuestionCount: ordered.length,
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
    throw new Error(
      `[workflow validation] Navigation workflow ${workflow.id} must contain ${NAVIGATION_PAGE_SIZE} questions per page.`
    );
  }
  const finalBlock = logicalBlocks.at(-1);
  const fixedInstance = getAttentionCheckInstances(occurrence, orderId).find(
    (instance) => instance.placement === "fixed-penultimate"
  );
  if (!finalBlock || !fixedInstance || finalBlock.at(-2)?.id !== fixedInstance.question.id) {
    throw new Error(`[workflow validation] ${workflow.id} must place the delayed-recall placeholder penultimate.`);
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
            hasWelcomePage: workflow.hasWelcomePage,
            profile,
            occurrence,
            layout,
            orderId,
            orderSeed: getOrderSeed(orderId),
            contentVersion: workflow.contentVersion,
            attentionCheckContentVersion: workflow.attentionCheckContentVersion,
            substantiveQuestionCount: workflow.substantiveQuestionCount,
            attentionCheckCount: workflow.attentionCheckCount,
            renderedQuestionCount: workflow.renderedQuestionCount,
            questionCount: workflow.questionCount,
            pageCount: workflow.pageCount,
            orderedQuestionIds: workflow.orderedQuestionIds,
          };
        })
      )
    )
  );
}
