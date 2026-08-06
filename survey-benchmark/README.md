# Survey Benchmark

A standalone Next.js application for the fixed survey-workflow benchmark. The existing `survey-site` application is
not used or modified by this project.

## Naming

- `v0` is the benchmark-suite release.
- `standard` is the first presentation profile, implemented with ordinary semantic HTML and standard web-development
  practices.
- Workflow IDs follow `v0-standard-o{occurrence}-{layout}-order{variant}`. Example:
  `v0-standard-o2-navigation-order03`.

This keeps the suite version independent of future observation-exposure profiles.

## Blueprint

There are 16 labeled workflow conditions and five fixed order forms per condition:

- Occurrence levels `o1` through `o8` contain 10 through 80 questions.
- Every occurrence block contains one question of each of the ten interaction types.
- `item` renders the complete ordered bank on one logical page.
- `navigation` divides the same ordered bank into pages of exactly ten questions.
- `order01` through `order05` are deterministic. A reload never silently changes the order.
- Matched item/navigation instances use the identical flattened question sequence.

At `o1`, both layouts have one page. They are retained as a renderer/evaluation parity check.

## Editing questions

Edit only the occurrence bank you need:

```text
lib/benchmark/questions/o1.ts   # 10 questions
lib/benchmark/questions/o2.ts   # 20 questions
...
lib/benchmark/questions/o8.ts   # 80 questions
```

Every question is an explicit entry. The `q.*` helpers provide placeholder options and constraints so prompts can be
replaced immediately. Optional final arguments customize choices or ranges:

```ts
q.radio("o02-b01-radio", 1, "Which option do you prefer?", ["Alpha", "Beta", "Gamma"]),
q.slider("o02-b01-slider", 1, "Choose a value.", { min: 1, max: 7, step: 1 }),
```

To create a dependency while keeping deterministic orders valid, add `dependsOn` with an object spread:

```ts
{
  ...q.shortText("o02-b02-short-text", 2, "Briefly explain your earlier selection."),
  dependsOn: ["o02-b01-radio"],
},
```

Keep IDs stable after collecting results. Increment the bank's `contentVersion` whenever its content changes.

Build-time validation rejects incorrect totals, missing question types, duplicate IDs/options, invalid constraints,
missing/cyclic dependencies and duplicate order variants.

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

The machine-readable 80-instance manifest is available at `http://localhost:3001/api/manifest`.

## Stored results

The app creates `benchmark.sqlite` on the first logged event. It records page-level events and completed submissions,
including the workflow ID, content version, fixed order, ordered question IDs and answers. The database and SQLite WAL
files are ignored by Git.

Session cookies are local-HTTP friendly by default. Set `SURVEY_COOKIE_SECURE=true` when deploying behind HTTPS.
