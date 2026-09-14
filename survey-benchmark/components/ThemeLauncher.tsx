"use client";

import { useState } from "react";

export function ThemeLauncher({ themes }: { themes: readonly { id: string; label: string }[] }) {
  const [theme, setTheme] = useState(themes[0].id);
  const [order, setOrder] = useState("order01");
  return (
    <section className="launcher-card">
      <div>
        <p className="eyebrow">Single-theme surveys</p>
        <h2>o1 theme configuration</h2>
        <p>11 questions from one theme, with no embedded attention checks.</p>
      </div>
      <div className="launcher-grid">
        <label>Theme
          <select value={theme} onChange={(event) => setTheme(event.target.value)}>
            {themes.map((entry) => <option key={entry.id} value={entry.id}>{entry.label}</option>)}
          </select>
        </label>
        <label>Question order
          <select value={order} onChange={(event) => setOrder(event.target.value)}>
            {["order01", "order02", "order03"].map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
      </div>
      <a className="primary-button launcher-button" href={`/survey/themes/${theme}/${order}`}>Open theme survey</a>
    </section>
  );
}
