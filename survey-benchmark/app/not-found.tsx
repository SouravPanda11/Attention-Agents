import Link from "next/link";

export default function NotFoundPage() {
  return (
    <main className="survey-shell">
      <section className="completion-card">
        <p className="eyebrow">Workflow not found</p>
        <h1>Invalid benchmark configuration</h1>
        <p>Choose a valid occurrence, layout and order from the workflow launcher.</p>
        <Link className="primary-button" href="/">
          Return to launcher
        </Link>
      </section>
    </main>
  );
}
