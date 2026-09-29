"use client";

import { useState } from "react";
import { ORDER_IDS, type LayoutMode, type Occurrence, type OrderId } from "@/lib/benchmark/schema";

type SamplingRow = { occurrence: Occurrence; sampleCount: number; questionCount: number };

export function WorkflowLauncher({ sampling }: { sampling: readonly SamplingRow[] }) {
  const [occurrence, setOccurrence] = useState<Occurrence>(1);
  const [sampleNumber, setSampleNumber] = useState("1");
  const [order, setOrder] = useState<OrderId>("order01");
  const [layout, setLayout] = useState<LayoutMode>("navigation");
  const pageCount = layout === "item" ? 1 : occurrence;
  const row = sampling.find((entry) => entry.occurrence === occurrence)!;
  const sampleIndex = Number(sampleNumber);
  const valid = Number.isInteger(sampleIndex) && sampleIndex >= 1 && sampleIndex <= row.sampleCount;
  const sampleId = `o${occurrence}-s${String(sampleIndex).padStart(4, "0")}`;
  return (
    <section className="launcher-card">
      <div>
        <p className="eyebrow">Surveys with attention checks</p>
        <h2>Choose a survey sample</h2>
        <p>Each sample has three reproducible orders. At O = 2-8, compare the same sequence across pages or on one page.</p>
      </div>
      <div className="launcher-grid">
        <label>Occurrence level
          <select value={occurrence} onChange={(event) => {
            setOccurrence(Number(event.target.value) as Occurrence);
            setSampleNumber("1");
            if (event.target.value === "1") setLayout("navigation");
          }}>
            {sampling.map((entry) => <option key={entry.occurrence} value={entry.occurrence}>
              O = {entry.occurrence} - {entry.sampleCount.toLocaleString("en-US")} samples
            </option>)}
          </select>
        </label>
        <label>Sample number (1-{row.sampleCount.toLocaleString("en-US")})
          <input type="number" min={1} max={row.sampleCount} step={1} value={sampleNumber}
            onChange={(event) => setSampleNumber(event.target.value)} aria-invalid={!valid} />
        </label>
        <label>Layout
          <select value={layout} onChange={(event) => setLayout(event.target.value as LayoutMode)} disabled={occurrence === 1}>
            <option value="navigation">{occurrence === 1 ? "Single page" : `Navigation-heavy - ${occurrence} pages`}</option>
            {occurrence > 1 && <option value="item">Item-heavy - one page</option>}
          </select>
        </label>
        <label>Question order
          <select value={order} onChange={(event) => setOrder(event.target.value as OrderId)}>
            {ORDER_IDS.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
      </div>
      <div className="launcher-summary">
        <span>{11 * occurrence} normal questions</span>
        <span>{2 * occurrence} ordinary ACs + 1 fixed AC</span>
        <span>{row.questionCount} questions total</span>
        <span>{pageCount} question page{pageCount === 1 ? "" : "s"}</span>
      </div>
      {valid ? <div className="button-row">
        <a className="primary-button launcher-button" href={`/survey/samples/${sampleId}/${order}${layout === "item" ? "/item" : ""}`}>Open survey</a>
        <a href={`/api/manifest?sampleId=${sampleId}&layout=${layout}`}>Inspect this layout&apos;s three orders</a>
      </div> : <p role="alert">Choose a sample number between 1 and {row.sampleCount}.</p>}
    </section>
  );
}
