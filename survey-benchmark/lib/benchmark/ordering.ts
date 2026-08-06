import type { OrderId, QuestionBank, SurveyQuestion } from "@/lib/benchmark/schema";
import { validateQuestionBank } from "@/lib/benchmark/validation";

export const ORDER_CONFIGS = [
  { id: "order01", seed: "wab-order-01" },
  { id: "order02", seed: "wab-order-02" },
  { id: "order03", seed: "wab-order-03" },
  { id: "order04", seed: "wab-order-04" },
  { id: "order05", seed: "wab-order-05" },
] as const satisfies readonly { id: OrderId; seed: string }[];

function stringSeed(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function randomGenerator(seed: number) {
  let state = seed;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(values: readonly T[], seed: string): T[] {
  const shuffled = [...values];
  const random = randomGenerator(stringSeed(seed));
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
  }
  return shuffled;
}

export function getOrderSeed(orderId: OrderId): string {
  const config = ORDER_CONFIGS.find((candidate) => candidate.id === orderId);
  if (!config) throw new Error(`Unknown order id: ${orderId}`);
  return config.seed;
}

export function orderQuestions(bank: QuestionBank, orderId: OrderId): SurveyQuestion[] {
  validateQuestionBank(bank);
  const byId = new Map(bank.questions.map((question) => [question.id, question]));
  const canonicalIds = [...byId.keys()].sort();
  const priority = new Map(
    seededShuffle(canonicalIds, `${getOrderSeed(orderId)}:${bank.id}`).map((id, index) => [id, index])
  );
  const remaining = new Set(canonicalIds);
  const emitted = new Set<string>();
  const result: SurveyQuestion[] = [];

  while (remaining.size > 0) {
    const eligible = [...remaining]
      .filter((id) => (byId.get(id)?.dependsOn ?? []).every((dependency) => emitted.has(dependency)))
      .sort((left, right) => (priority.get(left) ?? 0) - (priority.get(right) ?? 0));

    if (eligible.length === 0) throw new Error(`Unable to resolve dependencies for ${bank.id}.`);
    const selected = eligible[0];
    remaining.delete(selected);
    emitted.add(selected);
    result.push(byId.get(selected)!);
  }

  return result;
}
