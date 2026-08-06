import {
  QUESTION_KINDS,
  type ChoiceOption,
  type QuestionBank,
  type SurveyQuestion,
} from "@/lib/benchmark/schema";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`[workflow validation] ${message}`);
}

function validateOptions(questionId: string, values: readonly ChoiceOption[]) {
  assert(values.length >= 2, `${questionId} must define at least two options.`);
  const optionValues = new Set<string>();
  for (const option of values) {
    assert(option.value.trim().length > 0, `${questionId} contains an empty option value.`);
    assert(option.label.trim().length > 0, `${questionId} contains an empty option label.`);
    assert(!optionValues.has(option.value), `${questionId} repeats option value ${option.value}.`);
    optionValues.add(option.value);
  }
}

function validateQuestion(question: SurveyQuestion, occurrence: number) {
  assert(question.id.trim().length > 0, "Every question must have an id.");
  assert(question.prompt.trim().length > 0, `${question.id} must have a prompt.`);
  assert(Number.isInteger(question.block), `${question.id} must have an integer block.`);
  assert(
    question.block >= 1 && question.block <= occurrence,
    `${question.id} has block ${question.block}; expected 1..${occurrence}.`
  );

  switch (question.kind) {
    case "single-radio":
    case "single-dropdown":
    case "likert":
      validateOptions(question.id, question.options);
      break;
    case "single-checkbox":
      validateOptions(question.id, question.options);
      assert(question.minSelections === 1 && question.maxSelections === 1, `${question.id} must require exactly one selection.`);
      break;
    case "multiple-checkbox":
      validateOptions(question.id, question.options);
      assert(question.minSelections >= 1, `${question.id} must require at least one selection.`);
      assert(question.maxSelections >= question.minSelections, `${question.id} has invalid selection bounds.`);
      assert(question.maxSelections <= question.options.length, `${question.id} allows more selections than options.`);
      break;
    case "slider":
      assert(question.min < question.max, `${question.id} must have min < max.`);
      assert(question.step > 0, `${question.id} must have a positive step.`);
      break;
    case "ranking":
      validateOptions(question.id, question.items);
      break;
    case "numeric":
      if (question.min !== undefined && question.max !== undefined) {
        assert(question.min <= question.max, `${question.id} has min greater than max.`);
      }
      if (question.step !== undefined) assert(question.step > 0, `${question.id} must have a positive step.`);
      break;
    case "short-text":
    case "long-text":
      assert(question.minLength >= 0, `${question.id} must have a non-negative minLength.`);
      assert(question.maxLength >= question.minLength, `${question.id} has invalid text-length bounds.`);
      if (question.kind === "long-text") assert(question.rows >= 2, `${question.id} must render at least two rows.`);
      break;
  }
}

function validateDependencies(bank: QuestionBank) {
  const ids = new Set(bank.questions.map((question) => question.id));
  for (const question of bank.questions) {
    for (const dependency of question.dependsOn ?? []) {
      assert(ids.has(dependency), `${question.id} depends on missing question ${dependency}.`);
      assert(dependency !== question.id, `${question.id} cannot depend on itself.`);
    }
  }

  const visiting = new Set<string>();
  const visited = new Set<string>();
  const byId = new Map(bank.questions.map((question) => [question.id, question]));

  function visit(questionId: string) {
    if (visited.has(questionId)) return;
    assert(!visiting.has(questionId), `Dependency cycle detected at ${questionId}.`);
    visiting.add(questionId);
    for (const dependency of byId.get(questionId)?.dependsOn ?? []) visit(dependency);
    visiting.delete(questionId);
    visited.add(questionId);
  }

  for (const question of bank.questions) visit(question.id);
}

export function validateQuestionBank(bank: QuestionBank): QuestionBank {
  assert(bank.id === `o${bank.occurrence}`, `${bank.id} does not match occurrence ${bank.occurrence}.`);
  assert(bank.contentVersion >= 1, `${bank.id} must have a positive contentVersion.`);
  assert(
    bank.questions.length === bank.occurrence * QUESTION_KINDS.length,
    `${bank.id} must contain ${bank.occurrence * QUESTION_KINDS.length} questions; found ${bank.questions.length}.`
  );

  const ids = new Set<string>();
  for (const question of bank.questions) {
    assert(!ids.has(question.id), `${bank.id} repeats question id ${question.id}.`);
    ids.add(question.id);
    validateQuestion(question, bank.occurrence);
  }

  for (let block = 1; block <= bank.occurrence; block += 1) {
    const questions = bank.questions.filter((question) => question.block === block);
    assert(questions.length === QUESTION_KINDS.length, `${bank.id} block ${block} must contain ten questions.`);
    for (const kind of QUESTION_KINDS) {
      const count = questions.filter((question) => question.kind === kind).length;
      assert(count === 1, `${bank.id} block ${block} must contain exactly one ${kind}; found ${count}.`);
    }
  }

  validateDependencies(bank);
  return bank;
}
