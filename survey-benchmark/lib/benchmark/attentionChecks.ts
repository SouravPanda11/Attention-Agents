import "server-only";

import { isAnswerAttempted } from "@/lib/benchmark/answers";
import { q } from "@/lib/benchmark/questionFactories";
import {
  ORDER_IDS,
  type AnswerValue,
  type Occurrence,
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

type RotatingPrivateId = (typeof ROTATING_ATTENTION_CHECK_BANK)[number]["privateId"];

/*
 * Each form supplies the 15 rotating slots needed at o8:
 *   2 slots x 7 non-final blocks + 1 slot in the final block.
 *
 * The first eight entries of every form are a permutation of all eight rotating
 * checks. The remaining seven begin a second, differently paired pass. Thus all
 * checks receive coverage without evaluating every possible AC pairing.
 */
export const ROTATING_ATTENTION_CHECK_SCHEDULES = {
  order01: [
    "ac-direct-instruction",
    "ac-dropdown-instruction",
    "ac-ranking-instruction",
    "ac-short-text-entry",
    "ac-image-single-select",
    "ac-bogus-infrequency",
    "ac-text-captcha",
    "ac-image-captcha",
    "ac-ranking-instruction",
    "ac-image-single-select",
    "ac-direct-instruction",
    "ac-text-captcha",
    "ac-dropdown-instruction",
    "ac-image-captcha",
    "ac-bogus-infrequency",
  ],
  order02: [
    "ac-short-text-entry",
    "ac-bogus-infrequency",
    "ac-direct-instruction",
    "ac-image-captcha",
    "ac-ranking-instruction",
    "ac-text-captcha",
    "ac-dropdown-instruction",
    "ac-image-single-select",
    "ac-text-captcha",
    "ac-dropdown-instruction",
    "ac-short-text-entry",
    "ac-image-single-select",
    "ac-bogus-infrequency",
    "ac-ranking-instruction",
    "ac-direct-instruction",
  ],
  order03: [
    "ac-image-captcha",
    "ac-ranking-instruction",
    "ac-text-captcha",
    "ac-dropdown-instruction",
    "ac-bogus-infrequency",
    "ac-direct-instruction",
    "ac-image-single-select",
    "ac-short-text-entry",
    "ac-direct-instruction",
    "ac-bogus-infrequency",
    "ac-image-captcha",
    "ac-short-text-entry",
    "ac-ranking-instruction",
    "ac-image-single-select",
    "ac-text-captcha",
  ],
} as const satisfies Record<OrderId, readonly RotatingPrivateId[]>;

const byPrivateId = new Map<string, AttentionCheckDefinition>(
  ATTENTION_CHECK_BANK.map((definition) => [definition.privateId, definition] as const)
);

function opaqueHash(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36).padStart(7, "0");
}

function publicQuestionId(
  privateId: string,
  occurrence: Occurrence,
  orderId: OrderId,
  block: number,
  slotInBlock: number
): string {
  return `q-${opaqueHash(`${privateId}:o${occurrence}:${orderId}:b${block}:s${slotInBlock}`)}`;
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
  /** 1 or 2 within the logical block, independent of rendered position. */
  slotInBlock: 1 | 2;
  placement: "seeded" | "fixed-penultimate";
  question: SurveyQuestion;
  expectedAnswer: ExpectedAnswer | null;
  matcher: AnswerMatcher | null;
};

function instantiate(
  definition: AttentionCheckDefinition,
  occurrence: Occurrence,
  orderId: OrderId,
  block: number,
  slotInBlock: 1 | 2,
  placement: AttentionCheckInstance["placement"]
): AttentionCheckInstance {
  return {
    privateId: definition.privateId,
    role: definition.role,
    mechanism: definition.mechanism,
    slotInBlock,
    placement,
    question: {
      ...definition.question,
      id: publicQuestionId(definition.privateId, occurrence, orderId, block, slotInBlock),
      block,
    } as SurveyQuestion,
    expectedAnswer: definition.expectedAnswer,
    matcher: definition.matcher,
  };
}

/**
 * Returns exactly two AC instances per block.
 *
 * Non-final blocks receive two rotating checks. The final block receives one
 * rotating check and the fixed delayed-recall placeholder. Consumers must render
 * the fixed instance as the penultimate question of the final page/block.
 */
export function getAttentionCheckInstances(
  occurrence: Occurrence,
  orderId: OrderId
): readonly AttentionCheckInstance[] {
  const rotatingCount = occurrence * 2 - 1;
  const rotatingIds = ROTATING_ATTENTION_CHECK_SCHEDULES[orderId].slice(0, rotatingCount);
  let rotatingIndex = 0;
  const instances: AttentionCheckInstance[] = [];

  for (let block = 1; block <= occurrence; block += 1) {
    const isFinalBlock = block === occurrence;
    const rotatingSlots = isFinalBlock ? 1 : 2;

    for (let slot = 1; slot <= rotatingSlots; slot += 1) {
      const privateId = rotatingIds[rotatingIndex];
      rotatingIndex += 1;
      const definition = byPrivateId.get(privateId);
      if (!definition || definition.role !== "rotating") {
        throw new Error(`[attention checks] Missing rotating definition ${privateId}.`);
      }
      instances.push(
        instantiate(definition, occurrence, orderId, block, slot as 1 | 2, "seeded")
      );
    }

    if (isFinalBlock) {
      instances.push(
        instantiate(
          FIXED_PENULTIMATE_ATTENTION_CHECK,
          occurrence,
          orderId,
          block,
          2,
          "fixed-penultimate"
        )
      );
    }
  }

  return instances;
}

export function summarizeAttentionChecks(
  occurrence: Occurrence,
  orderId: OrderId,
  answers: Readonly<Record<string, unknown>>
) {
  const instances = getAttentionCheckInstances(occurrence, orderId);
  const results = instances.map((instance) => {
    const actual = answers[instance.question.id];
    const attempted = isAnswerAttempted(actual);
    const definition = byPrivateId.get(instance.privateId);
    if (!definition) throw new Error(`[attention checks] Missing definition ${instance.privateId}.`);
    return {
      privateId: instance.privateId,
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

for (const orderId of ORDER_IDS) {
  const ids = ROTATING_ATTENTION_CHECK_SCHEDULES[orderId];
  if (ids.length !== 15) {
    throw new Error(`[attention checks] ${orderId} must define 15 rotating slots for o8.`);
  }
  if (new Set(ids.slice(0, 8)).size !== 8) {
    throw new Error(`[attention checks] The first eight ${orderId} slots must cover all rotating checks.`);
  }
  for (const privateId of ids) {
    const definition = byPrivateId.get(privateId);
    if (!definition || definition.role !== "rotating") {
      throw new Error(`[attention checks] ${orderId} references invalid rotating check ${privateId}.`);
    }
  }
  for (let index = 0; index < 14; index += 2) {
    if (ids[index] === ids[index + 1]) {
      throw new Error(`[attention checks] ${orderId} repeats a check within rotating pair ${index / 2 + 1}.`);
    }
  }
}
