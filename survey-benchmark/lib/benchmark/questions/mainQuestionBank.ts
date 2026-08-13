import { getOrderSeed, seededShuffle } from "@/lib/benchmark/ordering";
import { civicTheme } from "@/lib/benchmark/questions/themes/civic";
import { consumerTheme } from "@/lib/benchmark/questions/themes/consumer";
import { digitalTheme } from "@/lib/benchmark/questions/themes/digital";
import { educationTheme } from "@/lib/benchmark/questions/themes/education";
import { financeTheme } from "@/lib/benchmark/questions/themes/finance";
import { lifestyleTheme } from "@/lib/benchmark/questions/themes/lifestyle";
import {
  mainQuestionId,
  THEME_IDS,
  type ThemeDefinition,
  type ThemeId,
} from "@/lib/benchmark/questions/themes/types";
import { wellbeingTheme } from "@/lib/benchmark/questions/themes/wellbeing";
import { workTheme } from "@/lib/benchmark/questions/themes/work";
import {
  QUESTION_KINDS,
  type Occurrence,
  type OrderId,
  type QuestionBank,
  type QuestionKind,
  type SurveyQuestion,
} from "@/lib/benchmark/schema";

export type { ThemeId } from "@/lib/benchmark/questions/themes/types";

export const MAIN_QUESTION_CONTENT_VERSION = 1 as const;

/**
 * The canonical theme order is significant: seeded shuffles start from this
 * order, so do not rearrange it after benchmark data collection begins.
 */
const THEME_DEFINITIONS = [
  consumerTheme,
  digitalTheme,
  wellbeingTheme,
  educationTheme,
  workTheme,
  financeTheme,
  civicTheme,
  lifestyleTheme,
] as const satisfies readonly ThemeDefinition[];

/** The eight survey domains used as surface content in the substantive bank. */
export const THEMES = THEME_DEFINITIONS.map(({ id, label }) => ({ id, label }));

export type MainQuestionEntry = {
  themeId: ThemeId;
  question: SurveyQuestion;
};

function entriesFor(kind: QuestionKind): MainQuestionEntry[] {
  return THEME_DEFINITIONS.map((theme) => ({
    themeId: theme.id,
    question: theme.questions[kind],
  }));
}

/**
 * Canonical question-type view consumed by the sampler. The editable source
 * remains theme-oriented in `questions/themes`; this transpose keeps the
 * benchmark's existing type-bucket sampling behavior unchanged.
 */
export const MAIN_QUESTION_BANK = {
  "single-radio": entriesFor("single-radio"),
  "single-dropdown": entriesFor("single-dropdown"),
  "single-checkbox": entriesFor("single-checkbox"),
  "multiple-checkbox": entriesFor("multiple-checkbox"),
  likert: entriesFor("likert"),
  slider: entriesFor("slider"),
  ranking: entriesFor("ranking"),
  numeric: entriesFor("numeric"),
  "short-text": entriesFor("short-text"),
  "long-text": entriesFor("long-text"),
  "image-single-select": entriesFor("image-single-select"),
} satisfies Record<QuestionKind, readonly MainQuestionEntry[]>;

function validateCanonicalBank() {
  if (THEME_DEFINITIONS.length !== THEME_IDS.length) {
    throw new Error(
      `[main question bank] Expected exactly ${THEME_IDS.length} theme files.`
    );
  }

  const themeIds = new Set<ThemeId>();
  for (const theme of THEME_DEFINITIONS) {
    if (themeIds.has(theme.id)) {
      throw new Error(`[main question bank] Duplicate theme definition ${theme.id}.`);
    }
    if (!theme.label.trim()) {
      throw new Error(`[main question bank] Theme ${theme.id} must have a label.`);
    }
    themeIds.add(theme.id);
  }
  for (const themeId of THEME_IDS) {
    if (!themeIds.has(themeId)) {
      throw new Error(`[main question bank] Missing theme definition ${themeId}.`);
    }
  }

  const questionIds = new Set<string>();
  for (const kind of QUESTION_KINDS) {
    const bucket = MAIN_QUESTION_BANK[kind];
    if (bucket.length !== THEME_IDS.length) {
      throw new Error(
        `[main question bank] ${kind} must contain exactly ${THEME_IDS.length} theme entries.`
      );
    }
    const seenThemes = new Set<ThemeId>();
    for (const item of bucket) {
      if (!themeIds.has(item.themeId) || seenThemes.has(item.themeId)) {
        throw new Error(`[main question bank] ${kind} must contain every theme exactly once.`);
      }
      if (item.question.kind !== kind) {
        throw new Error(`[main question bank] ${item.question.id} is in the wrong format bucket.`);
      }
      const expectedId = mainQuestionId(item.themeId, kind);
      if (item.question.id !== expectedId) {
        throw new Error(
          `[main question bank] ${item.question.id} must use canonical id ${expectedId}.`
        );
      }
      if (questionIds.has(item.question.id)) {
        throw new Error(`[main question bank] Duplicate canonical id ${item.question.id}.`);
      }
      seenThemes.add(item.themeId);
      questionIds.add(item.question.id);
    }
  }
  if (questionIds.size !== QUESTION_KINDS.length * THEME_IDS.length) {
    throw new Error("[main question bank] Expected exactly 88 canonical substantive questions.");
  }
}

validateCanonicalBank();

/** Build one nested horizon prefix from a frozen sampling form. */
export function sampleMainQuestionBank(
  occurrence: Occurrence,
  orderId: OrderId
): QuestionBank {
  const sampledBuckets = new Map<QuestionKind, MainQuestionEntry[]>();
  for (const kind of QUESTION_KINDS) {
    sampledBuckets.set(
      kind,
      seededShuffle(
        MAIN_QUESTION_BANK[kind],
        `${getOrderSeed(orderId)}:main-bucket:${kind}`
      )
    );
  }

  const questions: SurveyQuestion[] = [];
  for (let block = 1; block <= occurrence; block += 1) {
    for (const kind of QUESTION_KINDS) {
      const sampled = sampledBuckets.get(kind)?.[block - 1];
      if (!sampled) {
        throw new Error(`[main question bank] Missing ${kind} sample for block ${block}.`);
      }
      questions.push({ ...sampled.question, block } as SurveyQuestion);
    }
  }

  return {
    id: `o${occurrence}`,
    occurrence,
    contentVersion: MAIN_QUESTION_CONTENT_VERSION,
    questions,
  };
}

/** Supplies the 11 same-theme questions used by the separate O1 theme diagnostic. */
export function getThemeDiagnosticQuestionBank(themeId: ThemeId): QuestionBank {
  const questions = QUESTION_KINDS.map((kind) => {
    const item = MAIN_QUESTION_BANK[kind].find(
      (candidate) => candidate.themeId === themeId
    );
    if (!item) {
      throw new Error(`[main question bank] Missing ${kind} question for theme ${themeId}.`);
    }
    return { ...item.question, block: 1 } as SurveyQuestion;
  });
  return {
    id: "o1",
    occurrence: 1,
    contentVersion: MAIN_QUESTION_CONTENT_VERSION,
    questions,
  };
}
