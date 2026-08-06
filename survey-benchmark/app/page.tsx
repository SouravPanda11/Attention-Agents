import { WorkflowLauncher } from "@/components/WorkflowLauncher";
import { OCCURRENCES } from "@/lib/benchmark/schema";

export default function HomePage() {
  return (
    <main className="home-shell">
      <header className="hero">
        <p className="eyebrow">Survey Benchmark · suite v0 · standard presentation</p>
        <h1>Fixed survey workflows for web-agent evaluation</h1>
        <p>
          Eight occurrence levels are crossed with item-heavy and navigation-heavy layouts. Every condition has five
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
                <th>Questions</th>
                <th>Item-heavy pages</th>
                <th>Navigation-heavy pages</th>
                <th>Order forms</th>
              </tr>
            </thead>
            <tbody>
              {OCCURRENCES.map((occurrence) => (
                <tr key={occurrence}>
                  <td>o{occurrence}</td>
                  <td>{occurrence * 10}</td>
                  <td>1</td>
                  <td>{occurrence}</td>
                  <td>5 matched orders</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="matrix-note">
          At o1, both layouts contain one ten-question page. Keeping both labels provides a parity check for the
          renderer and evaluation pipeline.
        </p>
      </section>
    </main>
  );
}
