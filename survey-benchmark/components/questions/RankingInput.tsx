"use client";

import { useState } from "react";
import type { RankingQuestion } from "@/lib/benchmark/schema";

export function RankingInput({
  question,
  value,
  onChange,
}: {
  question: RankingQuestion;
  value: string[] | undefined;
  onChange: (value: string[]) => void;
}) {
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const itemValues = question.items.map((item) => item.value);
  const validStoredOrder =
    value?.length === itemValues.length &&
    value.every((candidate) => itemValues.includes(candidate)) &&
    new Set(value).size === itemValues.length;
  const order = validStoredOrder ? value : itemValues;
  const byValue = new Map(question.items.map((item) => [item.value, item]));

  function move(from: number, to: number) {
    if (to < 0 || to >= order.length || from === to) return;
    const next = [...order];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    onChange(next);
  }

  return (
    <ol className="ranking-list" aria-label="Ranked items">
      {order.map((itemValue, index) => {
        const item = byValue.get(itemValue)!;
        return (
          <li
            key={item.value}
            className="ranking-item"
            draggable
            data-ranking-item={item.value}
            data-ranking-position={index + 1}
            onDragStart={(event) => {
              setDraggedIndex(index);
              event.dataTransfer.effectAllowed = "move";
              event.dataTransfer.setData("text/plain", item.value);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "move";
            }}
            onDrop={(event) => {
              event.preventDefault();
              if (draggedIndex !== null) move(draggedIndex, index);
              setDraggedIndex(null);
            }}
            onDragEnd={() => setDraggedIndex(null)}
          >
            <span className="ranking-position" aria-hidden="true">
              {index + 1}
            </span>
            <span className="ranking-label">{item.label}</span>
            <span className="ranking-actions">
              <button
                type="button"
                className="secondary-button compact-button"
                data-action="rank-up"
                data-question-id={question.id}
                data-item-value={item.value}
                disabled={index === 0}
                aria-label={`Move ${item.label} up`}
                onClick={() => move(index, index - 1)}
              >
                Up
              </button>
              <button
                type="button"
                className="secondary-button compact-button"
                data-action="rank-down"
                data-question-id={question.id}
                data-item-value={item.value}
                disabled={index === order.length - 1}
                aria-label={`Move ${item.label} down`}
                onClick={() => move(index, index + 1)}
              >
                Down
              </button>
            </span>
          </li>
        );
      })}
    </ol>
  );
}
