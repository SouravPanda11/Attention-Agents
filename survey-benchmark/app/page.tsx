import { WorkflowLauncher } from "@/components/WorkflowLauncher";
import {
  ATTENTION_CHECKS_PER_BLOCK,
  OCCURRENCES,
  RENDERED_QUESTIONS_PER_BLOCK,
  SUBSTANTIVE_QUESTIONS_PER_BLOCK,
} from "@/lib/benchmark/schema";

export default function HomePage() {
  return (
    <main className="home-shell">
      <header className="hero">
        <p className="eyebrow">Survey Benchmark · suite v0 · standard presentation</p>
        <h1>Fixed survey workflows for web-agent evaluation</h1>
        <p>
          Eight occurrence levels are crossed with item-heavy and navigation-heavy layouts. Every condition has three
          deterministic, matched question orders.
        </p>
      </header>

      <WorkflowLauncher />

      <section className="matrix-card" aria-labelledby="matrix-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Blueprint</p>
            <h2 id="matrix-heading">Workflow matrix</h2>
          </div>
          <a href="/api/manifest">Open machine-readable manifest</a>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Occurrence</th>
                <th>Substantive</th>
                <th>Attention checks</th>
                <th>Displayed items</th>
                <th>Item-heavy pages</th>
                <th>Navigation-heavy pages</th>
                <th>Order forms</th>
              </tr>
            </thead>
            <tbody>
              {OCCURRENCES.map((occurrence) => (
                <tr key={occurrence}>
                  <td>o{occurrence}</td>
                  <td>{occurrence * SUBSTANTIVE_QUESTIONS_PER_BLOCK}</td>
                  <td>{occurrence * ATTENTION_CHECKS_PER_BLOCK}</td>
                  <td>{occurrence * RENDERED_QUESTIONS_PER_BLOCK}</td>
                  <td>1</td>
                  <td>{occurrence}</td>
                  <td>3 matched orders</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="matrix-note">
          Each logical block contains 11 substantive questions and two embedded attention checks. The fixed delayed-
          recall placeholder appears once in the final block, as its penultimate item. At o1, both layouts contain one
          13-item page; keeping both labels provides a renderer and evaluation-pipeline parity check.
        </p>
      </section>
    </main>
  );
}
