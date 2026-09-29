import type { SurveySample } from "@/lib/benchmark/sampling";
import { sampleMainQuestionBank } from "@/lib/benchmark/questions/mainQuestionBank";
import { validateQuestionBank } from "@/lib/benchmark/validation";

export function getQuestionBank(sample: SurveySample) {
  return validateQuestionBank(sampleMainQuestionBank(sample));
}
