import type { SurveySample } from "@/lib/benchmark/sampling";
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
  type QuestionBank,
  type QuestionKind,
  type SurveyQuestion,
} from "@/lib/benchmark/schema";

export type { ThemeId } from "@/lib/benchmark/questions/themes/types";

export const MAIN_QUESTION_CONTENT_VERSION = 1 as const;

/**
 * Canonical display order. Sample enumeration uses the matching THEME_IDS order;
 * keep both lists stable within a suite release.
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
 * Canonical question-type lookup. The editable source remains theme-oriented in
 * `questions/themes`; samples retrieve every type for each selected theme.
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

/** Content selection is independent of presentation order. */
export function sampleMainQuestionBank(sample: SurveySample): QuestionBank {
  const questions = sample.themeIds.flatMap((themeId, index) =>
    QUESTION_KINDS.map((kind) => {
      const entry = MAIN_QUESTION_BANK[kind].find((item) => item.themeId === themeId);
      if (!entry) throw new Error(`[main question bank] Missing ${themeId}/${kind}.`);
      return { ...entry.question, block: index + 1 } as SurveyQuestion;
    })
  );
  return { id: `o${sample.occurrence}`, occurrence: sample.occurrence,
    contentVersion: MAIN_QUESTION_CONTENT_VERSION, questions };
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
