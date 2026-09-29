import "server-only";

import { ROTATING_ATTENTION_CHECK_BANK } from "@/lib/benchmark/attentionChecks";
import { THEME_IDS, type ThemeId } from "@/lib/benchmark/questions/themes/types";
import { OCCURRENCES, getSampleLayouts, type Occurrence } from "@/lib/benchmark/schema";

export type SurveySample = {
  id: string;
  occurrence: Occurrence;
  themeIds: readonly ThemeId[];
  /** A multiset: repeated entries represent separate, independently answered instances. */
  rotatingAttentionCheckIds: readonly string[];
};

export function combinations<T>(values: readonly T[], count: number): T[][] {
  if (!Number.isInteger(count) || count < 0 || count > values.length) return [];
  if (count === 0) return [[]];
  return values.flatMap((value, index) =>
    combinations(values.slice(index + 1), count - 1).map((rest) => [value, ...rest])
  );
}

/** Enumeration order and IDs are frozen within a suite release. */
const samplesByOccurrence = new Map<Occurrence, readonly SurveySample[]>();

export function getSurveySamples(occurrence: Occurrence): readonly SurveySample[] {
  const cached = samplesByOccurrence.get(occurrence);
  if (cached) return cached;
  const ordinaryIds = ROTATING_ATTENTION_CHECK_BANK.map((check) => check.privateId);
  const selections = combinations(ordinaryIds, occurrence <= 4 ? 2 * occurrence : 2 * occurrence - 8);
  const samples = combinations(THEME_IDS, occurrence).flatMap((themeIds) =>
    selections.map((selection) => ({
      themeIds: Object.freeze(themeIds),
      rotatingAttentionCheckIds: Object.freeze(occurrence <= 4 ? selection : [...ordinaryIds, ...selection]),
    }))
  ).map((selection, index) => Object.freeze({
    id: `o${occurrence}-s${String(index + 1).padStart(4, "0")}`,
    occurrence,
    ...selection,
  }));
  samplesByOccurrence.set(occurrence, Object.freeze(samples));
  return samples;
}

export function getSurveySample(sampleId: string): SurveySample | undefined {
  const match = /^o([1-8])-s(\d{4})$/.exec(sampleId);
  if (!match) return undefined;
  const sample = getSurveySamples(Number(match[1]) as Occurrence)[Number(match[2]) - 1];
  return sample?.id === sampleId ? sample : undefined;
}

export function getSamplingSummary() {
  return OCCURRENCES.map((occurrence) => ({
    occurrence,
    sampleCount: getSurveySamples(occurrence).length,
    themeSelectionCount: combinations(THEME_IDS, occurrence).length,
    attentionSelectionCount: combinations(ROTATING_ATTENTION_CHECK_BANK, occurrence <= 4 ? 2 * occurrence : 2 * occurrence - 8).length,
    substantiveQuestionCount: 11 * occurrence,
    ordinaryAttentionCheckCount: 2 * occurrence,
    fixedAttentionCheckCount: 1,
    questionCount: 13 * occurrence + 1,
    layouts: getSampleLayouts(occurrence),
    navigationPageCount: occurrence,
    itemPageCount: occurrence === 1 ? null : 1,
    workflowInstanceCount: getSurveySamples(occurrence).length * getSampleLayouts(occurrence).length * 3,
    orderCount: 3,
  }));
}
