# Survey Benchmark

A standalone Next.js application for the fixed survey-workflow benchmark. The existing `survey-site` application is not used or modified by this project.

## Naming

- `v0` is the benchmark-suite release.
- `standard` is the first presentation profile, implemented with ordinary semantic HTML and standard web-development
  practices.
- Workflow IDs follow `v0-standard-o{occurrence}-{layout}-order{variant}`. Example:
  `v0-standard-o2-navigation-order03`.

This keeps the suite version independent of future observation-exposure profiles.

## Blueprint

There are 16 labeled workflow conditions and three frozen, seeded forms per condition:

- The substantive bank has 11 operational formats: ten text-based interaction formats plus one image-grounded
  single-selection format. The image format reuses radio selection; it is treated as a separate benchmark stratum
  because its stimulus modality is different.
- The canonical substantive bank contains 88 questions: 11 format buckets × 8 survey themes. Within each frozen form,
  the eight theme entries are independently permuted inside every format bucket. Block `j` takes entry `j` from each
  bucket, and `o{k}` takes Blocks 1 through `k`.
- Every occurrence block contains exactly 11 substantive questions, one from each operational format, plus exactly two
  embedded attention checks drawn from a separate attention-check bank.
- At `o{k}`, a workflow therefore contains `11k` substantive questions, `2k` attention checks and `13k` displayed items.
- Occurrence levels `o1` through `o8` contain 13 through 104 displayed items.
- `item` renders the complete ordered bank on one logical page.
- `navigation` renders one complete 13-item block per page.
- `order01` through `order03` are frozen seeded draws. A reload never silently changes content or order.
- Matched item/navigation instances use the identical flattened question sequence.
- Every canonical workflow URL begins with the same welcome page and `Start Survey` action.
- All questions are optional; Next and Submit never require a response.
- Navigation workflows provide both Previous and Next controls, with responses retained across pages.
- Submissions distinguish valid, attempted-but-invalid and skipped questions.

At `o1`, both layouts have one 13-item page. They are retained as a renderer/evaluation parity check.
The welcome page is tracked separately and is not included in the question-page count.
Its shared text can be edited in `lib/benchmark/welcome.ts`.

The attention-check bank has eight rotating mechanisms and one fixed delayed-recall placeholder. Non-final blocks use
two rotating checks; the final block uses one rotating check plus the fixed placeholder. Consequently, `o{k}` uses
`2k - 1` rotating placements and one fixed placement. The fixed check is always item 12 of the final 13-item block—the
penultimate question of the final page and of the one-page layout. The placeholder remains deliberately unscored until
its source question and answer resolver are specified in the bank file.

The eight rotating checks are fixed operational instances, not evidence that the benchmark covers every possible
attention mechanism. At long horizons the rotating instances repeat according to one of three frozen schedules; this
provides controlled exposure without evaluating every possible check pairing. Exact check identity and position are
matched across layouts. Private answer keys remain server-side and the browser receives only opaque public IDs.

## Editing questions

The 88 substantive questions are organized into eight author-facing theme files:

```text
lib/benchmark/questions/themes/
  consumer.ts
  digital.ts
  wellbeing.ts
  education.ts
  work.ts
  finance.ts
  civic.ts
  lifestyle.ts
```

Each file contains that theme's label and one question for every operational format, keyed by question type. Edit the
prompt, choices, range, or other question settings in the relevant theme file. Keep the `id(...)` call and the object
key unchanged so canonical IDs and question types remain stable.

`lib/benchmark/questions/mainQuestionBank.ts` is now aggregation machinery rather than an authoring file. It transposes
the eight theme-oriented definitions into the same 11 operational-format buckets used by deterministic sampling. The
three forms therefore do not duplicate the 88 canonical items: `o{k}` always selects the first `k` entries of every
shuffled bucket, substantive selections remain nested within a form, and `o8` contains every canonical item exactly
once. The aggregator also exports `getThemeDiagnosticQuestionBank(themeId)` for constructing the separate 11-item,
single-theme diagnostic forms.

The `q.*` helpers provide options and constraints. Optional final arguments customize choices or ranges:

```ts
q.radio("o02-b01-radio", 1, "Which option do you prefer?", ["Alpha", "Beta", "Gamma"]),
q.slider("o02-b01-slider", 1, "Choose a value.", { min: 1, max: 7, step: 1 }),
q.imageSingleSelect("o02-b01-image-single-select", 1, "Which image do you prefer?", [
  { label: "Option A", imageSrc: "/images/a.jpg", imageAlt: "Description of option A" },
  { label: "Option B", imageSrc: "/images/b.jpg", imageAlt: "Description of option B" },
]),
```

The eight rotating attention checks, their private answer keys, the three schedules, and the fixed delayed-recall
placeholder are all defined in
`lib/benchmark/attentionChecks.ts`. They are composed into workflows only after the substantive bank has been
validated and ordered, so adding checks does not alter substantive-question inclusion.

To create a dependency while keeping deterministic orders valid, add `dependsOn` with an object spread:

```ts
{
  ...q.shortText("o02-b02-short-text", 2, "Briefly explain your earlier selection."),
  dependsOn: ["o02-b01-radio"],
},
```

Keep IDs stable after collecting results. Increment `MAIN_QUESTION_CONTENT_VERSION` or
`ATTENTION_CHECK_CONTENT_VERSION` whenever the corresponding bank content changes.

Build-time validation rejects incorrect substantive totals, missing formats, duplicate IDs/options, invalid image
metadata or constraints, missing/cyclic dependencies and duplicate order variants. Workflow validation separately
requires every navigation page to contain 11 substantive questions and two attention checks and verifies that the
fixed placeholder is penultimate in the final block.

## Install and run

Requires Node.js 20+ and npm 10+.

```bash
npm install
npm run dev
```

Open `http://localhost:3001`. Port 3001 avoids conflicting with the original `survey-site` application on port 3000.

Useful checks:

```bash
npm run typecheck
npm run lint
npm run build
```

The machine-readable 48-instance scaffold manifest is available at `http://localhost:3001/api/manifest`.

## Stored results

The app creates `benchmark.sqlite` on the first logged event. It records page-level events and completed submissions,
including the workflow ID, content version, fixed order, ordered question IDs and answers. Attention-check attempts,
passes, failures and skipped checks are scored with the private key and stored separately. The database and SQLite WAL
files are ignored by Git.
Set `SURVEY_DB_PATH=:memory:` for temporary smoke tests that must not write study records to disk.

Session cookies are local-HTTP friendly by default. Set `SURVEY_COOKIE_SECURE=true` when deploying behind HTTPS.
