import { ORDER_IDS, OCCURRENCES, type Occurrence, type OrderId, type QuestionBank } from "@/lib/benchmark/schema";
import { orderQuestions } from "@/lib/benchmark/ordering";
import { sampleMainQuestionBank } from "@/lib/benchmark/questions/mainQuestionBank";
import { validateQuestionBank } from "@/lib/benchmark/validation";

/** Materialize an O=k prefix from one of the three frozen bucket shuffles. */
export function getQuestionBank(occurrence: Occurrence, orderId: OrderId): QuestionBank {
  return validateQuestionBank(sampleMainQuestionBank(occurrence, orderId));
}

// Fail fast if any frozen form violates counts, nesting, full-bank coverage, or
// presentation-order uniqueness.
for (const orderId of ORDER_IDS) {
  let priorIds = new Set<string>();
  for (const occurrence of OCCURRENCES) {
    const bank = getQuestionBank(occurrence, orderId);
    const ids = new Set(bank.questions.map((question) => question.id));
    for (const priorId of priorIds) {
      if (!ids.has(priorId)) {
        throw new Error(`[main question bank] ${orderId} o${occurrence} is not a nested prefix.`);
      }
    }
    priorIds = ids;
  }
  if (priorIds.size !== 88) {
    throw new Error(`[main question bank] ${orderId} o8 must contain all 88 canonical questions.`);
  }
}

for (const occurrence of OCCURRENCES) {
  const signatures = ORDER_IDS.map((orderId) =>
    orderQuestions(getQuestionBank(occurrence, orderId), orderId)
      .map((question) => question.id)
      .join("|")
  );
  if (new Set(signatures).size !== ORDER_IDS.length) {
    throw new Error(`[main question bank] o${occurrence} must have three distinct frozen forms.`);
  }
}
