import { ORDER_IDS, type Occurrence, type QuestionBank } from "@/lib/benchmark/schema";
import { orderQuestions } from "@/lib/benchmark/ordering";
import { validateQuestionBank } from "@/lib/benchmark/validation";
import { o1Bank } from "@/lib/benchmark/questions/o1";
import { o2Bank } from "@/lib/benchmark/questions/o2";
import { o3Bank } from "@/lib/benchmark/questions/o3";
import { o4Bank } from "@/lib/benchmark/questions/o4";
import { o5Bank } from "@/lib/benchmark/questions/o5";
import { o6Bank } from "@/lib/benchmark/questions/o6";
import { o7Bank } from "@/lib/benchmark/questions/o7";
import { o8Bank } from "@/lib/benchmark/questions/o8";

export const QUESTION_BANKS = [o1Bank, o2Bank, o3Bank, o4Bank, o5Bank, o6Bank, o7Bank, o8Bank].map(
  validateQuestionBank
) as readonly QuestionBank[];

for (const bank of QUESTION_BANKS) {
  const signatures = ORDER_IDS.map((orderId) => orderQuestions(bank, orderId).map((question) => question.id).join("|"));
  if (new Set(signatures).size !== ORDER_IDS.length) {
    throw new Error(`[workflow validation] ${bank.id} does not have five unique order variants.`);
  }
}

export function getQuestionBank(occurrence: Occurrence): QuestionBank {
  const bank = QUESTION_BANKS.find((candidate) => candidate.occurrence === occurrence);
  if (!bank) throw new Error(`Missing question bank for occurrence ${occurrence}.`);
  return bank;
}
