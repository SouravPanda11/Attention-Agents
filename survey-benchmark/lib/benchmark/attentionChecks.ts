import "server-only";
import { createHash } from "node:crypto";
import type { SurveySample } from "@/lib/benchmark/sampling";
import { getOrderSeed, seededShuffle } from "@/lib/benchmark/ordering";

import { isAnswerAttempted } from "@/lib/benchmark/answers";
import { q } from "@/lib/benchmark/questionFactories";
import {
  type AnswerValue,
  type OrderId,
  type QuestionKind,
  type SurveyQuestion,
} from "@/lib/benchmark/schema";

export const ATTENTION_CHECK_CONTENT_VERSION = 2 as const;

type ExpectedAnswer = Exclude<AnswerValue, undefined>;

export type AttentionCheckRole = "rotating" | "fixed-penultimate";

export type AttentionCheckMechanism =
  | "direct-instruction"
  | "dropdown-instruction"
  | "ranking-instruction"
  | "short-text-exact-entry"
  | "image-single-select"
  | "bogus-infrequency"
  | "text-captcha"
  | "image-captcha"
  | "delayed-recall-placeholder";

type AnswerMatcher = "exact" | "case-insensitive-text" | "ordered-array" | "unordered-array";

export type AttentionCheckDefinition = {
  /** Private evaluation identifier. Never include this value in a rendered workflow. */
  privateId: string;
  role: AttentionCheckRole;
  mechanism: AttentionCheckMechanism;
  /**
   * Authoring template. Its id and block are replaced with opaque, instance-specific
   * values before the question is rendered.
   */
  question: SurveyQuestion;
  /** Private server-side answer key. */
  expectedAnswer: ExpectedAnswer | null;
  matcher: AnswerMatcher | null;
};

/*
 * EDITABLE ATTENTION-CHECK BANK
 * -----------------------------
 * Edit the wording, options/assets, and answer keys in this section. Keep each
 * privateId stable after collecting results, and increment
 * ATTENTION_CHECK_CONTENT_VERSION whenever benchmark content changes.
 *
 * The first eight probes rotate through ordinary AC slots. The ninth probe is a
 * deliberately temporary delayed-recall placeholder. It is always placed as the
 * penultimate item of the final page/block by the workflow composer. Replace its
 * question, answer key and matcher once the delayed-recall source question and
 * scoring rule have been finalized. Until then it is rendered but unscored.
 */
export const ROTATING_ATTENTION_CHECK_BANK = [
  {
    privateId: "ac-direct-instruction",
    role: "rotating",
    mechanism: "direct-instruction",
    question: q.radio("attention-template-direct", 1, "For this item, select Option B."),
    expectedAnswer: "option_2",
    matcher: "exact",
  },
  {
    privateId: "ac-dropdown-instruction",
    role: "rotating",
    mechanism: "dropdown-instruction",
    question: q.dropdown(
      "attention-template-dropdown",
      1,
      "For this item, open the menu and select Option C."
    ),
    expectedAnswer: "option_3",
    matcher: "exact",
  },
  {
    privateId: "ac-ranking-instruction",
    role: "rotating",
    mechanism: "ranking-instruction",
    question: q.ranking(
      "attention-template-ranking",
      1,
      "Rank the items in this order: Item B, Item A, Item D, Item C."
    ),
    expectedAnswer: ["option_2", "option_1", "option_4", "option_3"],
    matcher: "ordered-array",
  },
  {
    privateId: "ac-short-text-entry",
    role: "rotating",
    mechanism: "short-text-exact-entry",
    question: q.shortText("attention-template-short-text", 1, "Enter the word BLUE."),
    expectedAnswer: "BLUE",
    matcher: "case-insensitive-text",
  },
  {
    privateId: "ac-image-single-select",
    role: "rotating",
    mechanism: "image-single-select",
    question: q.imageSingleSelect(
      "attention-template-image-select",
      1,
      "Select the circular design."
    ),
    expectedAnswer: "option_1",
    matcher: "exact",
  },
  {
    privateId: "ac-bogus-infrequency",
    role: "rotating",
    mechanism: "bogus-infrequency",
    question: q.likert(
      "attention-template-infrequency",
      1,
      "Indicate how strongly you agree: I was born on February 30."
    ),
    expectedAnswer: "option_1",
    matcher: "exact",
  },
  {
    privateId: "ac-text-captcha",
    role: "rotating",
    mechanism: "text-captcha",
    // Placeholder presentation: replace this with the final rendered text-CAPTCHA stimulus.
    question: q.shortText(
      "attention-template-text-captcha",
      1,
      "Text verification: enter the characters N7K4."
    ),
    expectedAnswer: "N7K4",
    matcher: "exact",
  },
  {
    privateId: "ac-image-captcha",
    role: "rotating",
    mechanism: "image-captcha",
    // Placeholder presentation: replace the two demo assets with the final CAPTCHA assets.
    question: q.imageSingleSelect(
      "attention-template-image-captcha",
      1,
      "Visual verification: select the square design."
    ),
    expectedAnswer: "option_2",
    matcher: "exact",
  },
] as const satisfies readonly AttentionCheckDefinition[];

export const FIXED_PENULTIMATE_ATTENTION_CHECK = {
  privateId: "ac-delayed-recall-placeholder",
  role: "fixed-penultimate",
  mechanism: "delayed-recall-placeholder",
  question: q.shortText(
    "attention-template-delayed-recall",
    1,
    "[PLACEHOLDER] This will become the delayed-recall question."
  ),
  expectedAnswer: null,
  matcher: null,
} as const satisfies AttentionCheckDefinition;

export const ATTENTION_CHECK_BANK = [
  ...ROTATING_ATTENTION_CHECK_BANK,
  FIXED_PENULTIMATE_ATTENTION_CHECK,
] as const satisfies readonly AttentionCheckDefinition[];

const byPrivateId = new Map<string, AttentionCheckDefinition>(
  ATTENTION_CHECK_BANK.map((definition) => [definition.privateId, definition] as const)
);

function publicQuestionId(sampleId: string, privateId: string, copy: number): string {
  return `q-${createHash("sha256").update(`${sampleId}:${privateId}:${copy}`).digest("hex").slice(0, 24)}`;
}

function answerMatches(definition: AttentionCheckDefinition, actual: unknown): boolean | null {
  if (definition.expectedAnswer === null || definition.matcher === null) return null;
  if (!isAnswerAttempted(actual)) return false;
  const expected = definition.expectedAnswer;

  switch (definition.matcher) {
    case "ordered-array":
      return (
        Array.isArray(expected) &&
        Array.isArray(actual) &&
        expected.length === actual.length &&
        expected.every((value, index) => actual[index] === value)
      );
    case "unordered-array":
      return (
        Array.isArray(expected) &&
        Array.isArray(actual) &&
        expected.length === actual.length &&
        expected.every((value) => actual.includes(value))
      );
    case "case-insensitive-text":
      return (
        typeof expected === "string" &&
        typeof actual === "string" &&
        expected.trim().toLocaleLowerCase() === actual.trim().toLocaleLowerCase()
      );
    case "exact":
      return actual === expected;
  }
}

export type AttentionCheckInstance = {
  privateId: string;
  role: AttentionCheckRole;
  mechanism: AttentionCheckMechanism;
  /** Stable copy identity, even when repeated checks move between pages. */
  copy: number;
  slotInBlock: 1 | 2 | 3;
  placement: "seeded" | "fixed-penultimate";
  question: SurveyQuestion;
  expectedAnswer: ExpectedAnswer | null;
  matcher: AnswerMatcher | null;
};

function instantiate(
  sampleId: string, privateId: string, copy: number, block: number, slotInBlock: 1 | 2 | 3
): AttentionCheckInstance {
  const definition = byPrivateId.get(privateId);
  if (!definition) throw new Error(`[attention checks] Unknown check ${privateId}.`);
  return {
    privateId, copy, slotInBlock,
    role: definition.role,
    mechanism: definition.mechanism,
    placement: definition.role === "rotating" ? "seeded" : "fixed-penultimate",
    question: { ...definition.question, id: publicQuestionId(sampleId, privateId, copy), block } as SurveyQuestion,
    expectedAnswer: definition.expectedAnswer,
    matcher: definition.matcher,
  };
}

/** Two ordinary checks on EVERY page, plus one extra fixed check on the final page. */
export function getAttentionCheckInstances(sample: SurveySample, orderId: OrderId): readonly AttentionCheckInstance[] {
  const seen = new Map<string, number>();
  const tokens = sample.rotatingAttentionCheckIds.map((privateId) => {
    const copy = (seen.get(privateId) ?? 0) + 1;
    seen.set(privateId, copy);
    return { privateId, copy };
  });
  const seed = getOrderSeed(orderId, sample.id);
  let allocated = seededShuffle(tokens, `${seed}:attention-allocation:0`);
  const hasDuplicatePair = () => allocated.some((token, index) =>
    index % 2 === 0 && token.privateId === allocated[index + 1]?.privateId);
  // Rejection sampling preserves random allocation while avoiding identical ACs on a page.
  for (let attempt = 1; hasDuplicatePair() && attempt <= 128; attempt++) {
    allocated = seededShuffle(tokens, `${seed}:attention-allocation:${attempt}`);
  }
  if (hasDuplicatePair()) {
    // Guaranteed fallback for this bank (each identity occurs at most twice).
    const sorted = [...tokens].sort((a, b) => (a.privateId < b.privateId ? -1 : a.privateId > b.privateId ? 1 : 0));
    allocated = seededShuffle(Array.from({ length: sample.occurrence }, (_, index) =>
      [sorted[index], sorted[index + sample.occurrence]]), `${seed}:attention-fallback`).flat();
  }
  const instances = allocated.map((token, index) => instantiate(
    sample.id, token.privateId, token.copy, Math.floor(index / 2) + 1, index % 2 === 0 ? 1 : 2
  ));
  instances.push(instantiate(sample.id, FIXED_PENULTIMATE_ATTENTION_CHECK.privateId, 1, sample.occurrence, 3));
  return instances;
}

export function summarizeAttentionChecks(
  sample: SurveySample,
  orderId: OrderId,
  answers: Readonly<Record<string, unknown>>
) {
  const instances = getAttentionCheckInstances(sample, orderId);
  const results = instances.map((instance) => {
    const actual = answers[instance.question.id];
    const attempted = isAnswerAttempted(actual);
    const definition = byPrivateId.get(instance.privateId);
    if (!definition) throw new Error(`[attention checks] Missing definition ${instance.privateId}.`);
    return {
      privateId: instance.privateId,
      copy: instance.copy,
      publicQuestionId: instance.question.id,
      role: instance.role,
      mechanism: instance.mechanism,
      kind: instance.question.kind as QuestionKind,
      block: instance.question.block,
      slotInBlock: instance.slotInBlock,
      placement: instance.placement,
      attempted,
      passed: answerMatches(definition, actual),
    };
  });

  const attemptedCount = results.filter((result) => result.attempted).length;
  const scoredCount = results.filter((result) => result.passed !== null).length;
  const passCount = results.filter((result) => result.passed === true).length;
  const failCount = results.filter((result) => result.passed === false).length;
  return {
    checkCount: results.length,
    scoredCount,
    unscoredCount: results.length - scoredCount,
    attemptedCount,
    skippedCount: results.length - attemptedCount,
    passCount,
    failCount,
    allPassed: scoredCount > 0 && passCount === scoredCount,
    results,
  };
}

if (ROTATING_ATTENTION_CHECK_BANK.length !== 8) {
  throw new Error("[attention checks] Expected exactly eight rotating definitions.");
}
if (ATTENTION_CHECK_BANK.length !== 9) {
  throw new Error("[attention checks] Expected eight rotating checks and one fixed-final check.");
}
if (new Set(ATTENTION_CHECK_BANK.map((definition) => definition.privateId)).size !== 9) {
  throw new Error("[attention checks] Every privateId must be unique.");
}
if (
  ROTATING_ATTENTION_CHECK_BANK.some(
    (definition) => definition.expectedAnswer === null || definition.matcher === null
  )
) {
  throw new Error("[attention checks] Every rotating check must have an answer key and matcher.");
}
if (
  (FIXED_PENULTIMATE_ATTENTION_CHECK.expectedAnswer === null) !==
  (FIXED_PENULTIMATE_ATTENTION_CHECK.matcher === null)
) {
  throw new Error("[attention checks] Configure both the delayed-recall answer and matcher, or neither.");
}
