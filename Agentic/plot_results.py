"""Generate shareable charts and tables from saved runs; no model/server calls."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

from plots import attempts, efficiency, formats, overview, themes
from plots.common import (attempt_rows, by_model, format_rows, group_id, load_groups, parser,
                          summary_rows, theme_rows, write_csv)

CHARTS = {
    "overview": (overview.draw, "01_overview", "Submission and answer completion",
                 "Compare surveys reaching Submit with questions receiving a valid answer. Counts inside the outcome bars refer to questions."),
    "formats": (formats.draw, "02_question_types", "Performance by question type",
                "Pooled over selected themes and orders. Denominators include only items in accepted submissions; image selections in DOM mode use text labels."),
    "themes": (themes.draw, "03_theme_order", "Themes and orders",
               "Each cell shows valid answers / all questions in attempted runs. The color scale always spans 0–100%. n is the number of runs in that cell."),
    "attempts": (attempts.draw, "04_attempts", "How many attempts does a question use?",
                 "Actual model calls per question, split by the agent's final local outcome. Unanswered includes exhausted budgets and any voluntary skips."),
    "efficiency": (efficiency.draw, "05_efficiency", "Time and retry reasons",
                   "Every dot is a run. Error counters are events, can overlap, and should not be added as if they were distinct questions."),
}


def table(rows, columns):
    def cell(value):
        if value is None:
            return "—"
        return escape(f"{value:.1f}" if isinstance(value, float) else str(value))
    return ("<div class='table-wrap'><table><thead><tr>" +
            "".join(f"<th>{escape(label)}</th>" for _, label in columns) + "</tr></thead><tbody>" +
            "".join("<tr>" + "".join(f"<td>{cell(row.get(key))}</td>" for key, _ in columns) + "</tr>" for row in rows) +
            "</tbody></table></div>")


STYLE = """body{font:16px/1.6 system-ui,sans-serif;color:#243447;background:#f4f6f8;margin:0}
main{max-width:1200px;margin:36px auto;padding:32px;background:white;border-radius:12px}
h1,h2,h3{line-height:1.2}h1{font-size:32px}h2{margin-top:42px}a{color:#126d79}
.note{border-left:4px solid #197f86;background:#eff7f7;padding:14px 20px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}
th,td{border-bottom:1px solid #dce3e8;padding:10px;text-align:right}th:first-child,td:first-child{text-align:left}
th{background:#f1f4f7}img{max-width:100%;height:auto}small{color:#526275}
code{background:#f1f4f7;padding:2px 5px}details{margin:16px 0}
@media print{body{background:white}main{padding:0;margin:0}h2{break-after:avoid}img{break-inside:avoid}}"""


def page(title, body):
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"


def write_report(protocol, runs, out, selected, exports):
    summaries = summary_rows(runs)
    tables = {"summary": summaries, "question_types": format_rows(runs),
              "theme_order": theme_rows(runs), "attempts": attempt_rows(runs)}
    for name, rows in tables.items():
        write_csv(out / f"{name}.csv", rows)
    metadata = {"generated_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                "source_runs": [r["_source"] for r in runs], "charts": selected,
                "runs": len(runs), "models": list(by_model(runs)),
                "summary": summaries}
    (out / "plot_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    body = "<h1>Survey agent results</h1>"
    body += (f"<p>{len(runs)} runs · {len(summaries)} model(s) · "
             f"{sum(r['question_count'] for r in runs)} questions across attempted runs</p>")
    body += "<div class='note'><strong>What this measures:</strong> successful form completion, not semantic answer quality. "
    body += "Submitting a survey does not mean every question was answered. Invalid answers cleared after three attempts appear as skipped at submission.</div>"
    if protocol.get("attention_check_content_version") == 0:
        body += "<p>No attention checks were included; attention scores are not applicable.</p>"
    body += "<h2>Model summary</h2>"
    body += table(summaries, [("model", "Model"), ("runs", "Runs"), ("submitted_runs", "Submitted"),
                             ("valid_answers", "Valid answers"), ("questions", "All questions"),
                             ("end_to_end_valid_pct", "Valid (%)"), ("fully_valid_runs", "Fully valid surveys"),
                             ("model_calls", "Model calls"), ("mean_run_seconds", "Mean seconds/run")])
    body += "<p>Valid (%) counts server-valid answers over all questions in attempted runs, including failed submissions. "
    body += "Pending scheduled runs are not included. Running summaries are excluded. Each saved run directory is counted once.</p>"
    body += "<p>Download tables: " + " · ".join(f"<a href='{name}.csv'>{name.replace('_', ' ')} CSV</a>" for name in tables) + "</p>"
    body += "<details><summary>Included settings and data coverage</summary>"
    body += table([{"setting": key, "value": value} for key, value in protocol.items()], [("setting", "Setting"), ("value", "Value")])
    coverage = [{"model": model, "themes": ", ".join(sorted({str(r.get('theme_id')) for r in items})),
                 "orders": ", ".join(sorted({str(r.get('order_id')) for r in items})),
                 "batches": len({Path(r["_source"]).parent.parent.name for r in items})}
                for model, items in by_model(runs).items()]
    body += table(coverage, [("model", "Model"), ("themes", "Themes"), ("orders", "Orders"), ("batches", "Batches")])
    body += "<p>Compare models with matching coverage. Different prompts/settings are placed in separate report folders. "
    body += "Repeated runs across batches are pooled; filter with --batch to isolate an experiment. Source files are listed in plot_manifest.json.</p></details>"
    for name in selected:
        _, stem, title, description = CHARTS[name]
        if (name == "formats" and not tables["question_types"]) or not (out / f"{stem}.png").exists():
            body += f"<h2>{escape(title)}</h2><p>No eligible data for this chart.</p>"
            continue
        body += f"<h2>{escape(title)}</h2><p>{escape(description)}</p>"
        body += f"<img src='{stem}.png' alt='{escape(title)}'>"
        body += "<p>Download: " + " · ".join(f"<a href='{stem}.{ext}'>{ext.upper()}</a>" for ext in exports) + "</p>"
    body += "<h2>Question-type table</h2>"
    body += table(tables["question_types"], [("model", "Model"), ("kind", "Question type"), ("valid", "Valid"),
                                          ("skipped", "Skipped"), ("invalid", "Invalid"),
                                          ("submitted_items", "Submitted items"), ("valid_pct", "Valid (%)")])
    body += "<p>Question-type rates pool item counts across themes and orders rather than averaging percentages. "
    body += "These are descriptive results. With one run per theme/order, small differences do not establish a consistent order effect. "
    body += "Timing reflects the local hardware and server configuration.</p>"
    body += "<small>Generated offline from saved run summaries. No model calls, screenshots, or survey responses are included in this report.</small>"
    (out / "index.html").write_text(page("Survey agent results", body), encoding="utf-8")


def main():
    cli = parser(__doc__)
    cli.add_argument("--plots", nargs="+", choices=list(CHARTS), default=list(CHARTS))
    args = cli.parse_args()
    groups = load_groups(args.runs_dir, args.models, args.batch)
    root = args.output_dir or args.runs_dir / "plots"
    root.mkdir(parents=True, exist_ok=True)
    exports = list(dict.fromkeys(["png", *args.formats]))  # PNG previews make the HTML portable.
    links = []
    for protocol, runs in groups:
        out = root if len(groups) == 1 else root / f"protocol-{group_id(protocol)}"
        out.mkdir(parents=True, exist_ok=True)
        for name in args.plots:
            CHARTS[name][0](runs, out, exports)
        write_report(protocol, runs, out, args.plots, exports)
        links.append((out.name, protocol))
        print(f"{len(runs)} runs -> {out / 'index.html'}")
    if len(groups) > 1:
        body = "<h1>Survey agent reports</h1><p>Different experiment settings are reported separately.</p><ul>"
        for name, protocol in links:
            label = f"{protocol.get('prompt_version')} · {protocol.get('observation')} · {protocol.get('behavior')} · {name}"
            body += f"<li><a href='{name}/index.html'>{escape(label)}</a></li>"
        (root / "index.html").write_text(page("Survey agent reports", body + "</ul>"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"Plotting failed: {exc}", file=sys.stderr)
        sys.exit(1)
