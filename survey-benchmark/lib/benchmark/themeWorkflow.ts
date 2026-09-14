import { getThemeDiagnosticQuestionBank, THEMES } from "@/lib/benchmark/questions/mainQuestionBank";
import type { ThemeId } from "@/lib/benchmark/questions/themes/types";
import { getOrderSeed, orderQuestions } from "@/lib/benchmark/ordering";
import { ORDER_IDS, SUITE_VERSION, type OrderId, type Workflow } from "@/lib/benchmark/schema";

/** Same 11 substantive questions in each of three deterministic presentation orders. */
export function buildThemeWorkflow(themeId: ThemeId, orderId: OrderId): Workflow {
  const bank = getThemeDiagnosticQuestionBank(themeId);
  const questions = orderQuestions(bank, orderId);
  return {
    id: `${SUITE_VERSION}-standard-o1-theme-${themeId}-${orderId}`,
    themeId,
    themeLabel: THEMES.find((theme) => theme.id === themeId)!.label,
    suiteVersion: SUITE_VERSION,
    hasWelcomePage: true,
    profile: "standard",
    occurrence: 1,
    occurrenceId: "o1",
    layout: "item",
    orderId,
    contentVersion: bank.contentVersion,
    attentionCheckContentVersion: 0,
    substantiveQuestionCount: questions.length,
    attentionCheckCount: 0,
    renderedQuestionCount: questions.length,
    questionCount: questions.length,
    pageCount: 1,
    questionsPerNavigationPage: questions.length,
    orderedQuestionIds: questions.map((question) => question.id),
    pages: [{ id: "page-01", index: 0, questions }],
  };
}

export function getThemeWorkflowManifest() {
  return THEMES.flatMap((theme) => {
    const variants = ORDER_IDS.map((orderId) => {
      const { pages, ...workflow } = buildThemeWorkflow(theme.id, orderId);
      if (pages.length !== 1) throw new Error(`Theme ${theme.id} must have one question page.`);
      return { ...workflow, url: `/survey/themes/${theme.id}/${orderId}`, orderSeed: getOrderSeed(orderId) };
    });
    if (new Set(variants.map((workflow) => workflow.orderedQuestionIds.join("|"))).size !== ORDER_IDS.length) {
      throw new Error(`Theme ${theme.id} must have three distinct question orders.`);
    }
    return variants;
  });
}
