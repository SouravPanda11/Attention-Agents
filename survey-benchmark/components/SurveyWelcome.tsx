"use client";

import { WELCOME_CONTENT } from "@/lib/benchmark/welcome";

export function SurveyWelcome({ workflowId, onStart }: { workflowId: string; onStart: () => void }) {
  return (
    <main className="welcome-shell" data-screen="welcome" data-workflow-id={workflowId}>
      <section className="welcome-card" aria-labelledby="welcome-title">
        <p className="eyebrow">{WELCOME_CONTENT.eyebrow}</p>
        <h1 id="welcome-title">{WELCOME_CONTENT.title}</h1>
        <p className="welcome-introduction">{WELCOME_CONTENT.introduction}</p>

        <div className="welcome-instructions">
          <h2>Before you begin</h2>
          <ul>
            {WELCOME_CONTENT.instructions.map((instruction) => (
              <li key={instruction}>{instruction}</li>
            ))}
          </ul>
        </div>

        <button type="button" className="primary-button welcome-start-button" data-action="start-survey" onClick={onStart}>
          {WELCOME_CONTENT.startLabel}
        </button>
      </section>
    </main>
  );
}
