import { WorkflowLauncher } from "@/components/WorkflowLauncher";
import { ThemeLauncher } from "@/components/ThemeLauncher";
import { THEMES } from "@/lib/benchmark/questions/mainQuestionBank";
import { getSamplingSummary } from "@/lib/benchmark/sampling";
import { SUITE_VERSION } from "@/lib/benchmark/schema";

export default function HomePage() {
  const sampling = getSamplingSummary();
  return (
    <main className="home-shell">
      <header className="hero">
        <p className="eyebrow">Survey Benchmark | suite {SUITE_VERSION} | standard presentation</p>
        <h1>Survey coverage and long-horizon agent behavior</h1>
        <p>7,575 attention-check samples and eight theme-only baselines, each with three reproducible question orders.
          At O = 2-8, matched navigation-heavy and item-heavy variants compare multiple pages with one long page.</p>
      </header>
      <ThemeLauncher themes={THEMES} />
      <WorkflowLauncher sampling={sampling} />
      <section className="matrix-card" aria-labelledby="matrix-heading">
        <div className="section-heading">
          <div><p className="eyebrow">Sampling design</p><h2 id="matrix-heading">Occurrence levels</h2></div>
          <a href="/api/manifest">Open paginated manifest</a>
        </div>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Occurrence</th><th>Theme selections</th><th>AC selections</th>
              <th>Samples</th><th>Questions</th><th>Navigation pages</th><th>Item pages</th><th>Orders per layout</th></tr></thead>
            <tbody>{sampling.map((row) => <tr key={row.occurrence}>
              <td>{row.occurrence}</td><td>{row.themeSelectionCount}</td><td>{row.attentionSelectionCount}</td>
              <td>{row.sampleCount.toLocaleString("en-US")}</td><td>{row.questionCount}</td>
              <td>{row.navigationPageCount}</td><td>{row.itemPageCount ?? "-"}</td><td>3</td>
            </tr>)}</tbody>
          </table>
        </div>
        <p className="matrix-note">O = 1-4 selects ordinary ACs without repetition. O = 5-8 includes all eight,
          then selects which appear twice. Navigation pages contain 11 normal questions and two ordinary ACs;
          the final page adds the fixed penultimate AC. Item-heavy surveys place the identical sequence on one page.
          O = 1 has one layout. Welcome screens are separate from question pages.</p>
      </section>
    </main>
  );
}
