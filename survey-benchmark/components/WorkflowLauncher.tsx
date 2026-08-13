"use client";

import { useState } from "react";
import {
  ATTENTION_CHECKS_PER_BLOCK,
  RENDERED_QUESTIONS_PER_BLOCK,
  SUBSTANTIVE_QUESTIONS_PER_BLOCK,
  type LayoutMode,
  type Occurrence,
  type OrderId,
} from "@/lib/benchmark/schema";

export function WorkflowLauncher() {
  const [occurrence, setOccurrence] = useState<Occurrence>(1);
  const [layout, setLayout] = useState<LayoutMode>("item");
  const [order, setOrder] = useState<OrderId>("order01");
  const substantiveQuestionCount = occurrence * SUBSTANTIVE_QUESTIONS_PER_BLOCK;
  const attentionCheckCount = occurrence * ATTENTION_CHECKS_PER_BLOCK;
  const questionCount = occurrence * RENDERED_QUESTIONS_PER_BLOCK;
  const pageCount = layout === "item" ? 1 : occurrence;
  const url = `/survey/standard/o${occurrence}/${layout}/${order}`;

  return (
    <section className="launcher-card">
      <div>
        <p className="eyebrow">Launch a fixed workflow instance</p>
        <h2>Workflow configuration</h2>
      </div>
      <div className="launcher-grid">
        <label>
          Occurrence level
          <select value={occurrence} onChange={(event) => setOccurrence(Number(event.target.value) as Occurrence)}>
            {[1, 2, 3, 4, 5, 6, 7, 8].map((value) => (
              <option key={value} value={value}>
                o{value} · {value * RENDERED_QUESTIONS_PER_BLOCK} displayed items
              </option>
            ))}
          </select>
        </label>
        <label>
          Layout
          <select value={layout} onChange={(event) => setLayout(event.target.value as LayoutMode)}>
            <option value="item">Item-heavy · all blocks on one page</option>
            <option value="navigation">Navigation-heavy · one 13-item block per page</option>
          </select>
        </label>
        <label>
          Fixed order
          <select value={order} onChange={(event) => setOrder(event.target.value as OrderId)}>
            {[1, 2, 3].map((value) => {
              const orderId = `order${String(value).padStart(2, "0")}` as OrderId;
              return (
                <option key={orderId} value={orderId}>
                  Order {String(value).padStart(2, "0")}
                </option>
              );
            })}
          </select>
        </label>
      </div>
      <div className="launcher-summary">
        <span>{substantiveQuestionCount} substantive questions</span>
        <span>{attentionCheckCount} attention check{attentionCheckCount === 1 ? "" : "s"}</span>
        <span>{questionCount} displayed items</span>
        <span>{pageCount} page{pageCount === 1 ? "" : "s"}</span>
        <span>standard profile</span>
      </div>
      <a className="primary-button launcher-button" href={url}>
        Open workflow
      </a>
    </section>
  );
}
