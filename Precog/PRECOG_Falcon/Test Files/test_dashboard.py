
# Build an HTML dashboard and paper-ready charts from PRECOG test results.

import csv
import html
import json
import statistics
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    path = RESULTS / name

    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot(name, rows, x_key, lines, ylabel, title):
    if not rows:
        return ""

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for group in sorted({row["topic"] for row in rows}):
        data = sorted(
            (row for row in rows if row["topic"] == group),
            key=lambda row: float(row[x_key]),
        )

        for key, label in lines:
            ax.plot(
                [float(row[x_key]) for row in data],
                [float(row[key]) for row in data],
                marker="o",
                label=f"{group}: {label}",
            )

    ax.set(
        xlabel=x_key.replace("_", " ").title(),
        ylabel=ylabel,
        title=title,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=180)

    plt.close(fig)

    return (
        f'<figure><img src="figures/{name}.png" '
        f'alt="{html.escape(title)}">'
        f'<figcaption><a href="figures/{name}.pdf">'
        f"Paper figure (PDF)</a></figcaption></figure>"
    )


def table(rows, fields):
    if not rows:
        return "<p>No measurements saved yet.</p>"

    headings = "".join(
        f"<th>{html.escape(label)}</th>"
        for _, label in fields
    )

    body = "".join(
        "<tr>" + "".join(
            f'<td>{html.escape(str(row.get(key, "")))}</td>'
            for key, _ in fields
        ) + "</tr>"
        for row in rows
    )

    return (
        f'<div class="scroll"><table><tr>{headings}</tr>'
        f"{body}</table></div>"
    )


def main():
    baseline = read_csv("comparison.csv")
    cached = read_csv("cached_performance.csv")

    path = RESULTS / "correctness.json"
    correctness = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists() else {}
    )

    sections = []

    if baseline:
        online = statistics.median(
            float(row["speedup"]) for row in baseline
        )

        speedup = plot(
            "online_speedup", baseline, "context_tokens",
            [("speedup", "PRECOG online speedup")],
            "Speedup (x)", "Full context vs PRECOG",
        )

        latency = plot(
            "online_latency", baseline, "context_tokens",
            [
                ("baseline_s", "Full context"),
                ("precog_s", "PRECOG"),
            ],
            "Seconds", "Online inference latency",
        )

        memory = plot(
            "gpu_memory", baseline, "context_tokens",
            [
                ("baseline_peak_mib", "Full context"),
                ("precog_peak_mib", "PRECOG"),
            ],
            "Additional peak allocated MiB",
            "GPU allocation comparison",
        )

        sections.append(
            "<h2>1. Primary research comparison</h2>"
            f"<p><strong>Median online speedup: {online:.2f}x"
            f"</strong> across {len(baseline)} cases.</p>"
            + speedup + latency + memory
            + table(baseline, [
                ("topic", "Topic"),
                ("context_tokens", "Context"),
                ("question_tokens", "Question"),
                ("baseline_s", "Baseline s"),
                ("precog_s", "PRECOG s"),
                ("speedup", "Speedup"),
                ("baseline_peak_mib", "Baseline MiB"),
                ("precog_peak_mib", "PRECOG MiB"),
                ("encoding_s", "Encode once s"),
                ("stored_state_mib", "Stored state MiB"),
                ("matched", "Matched"),
                ("total", "Total"),
            ])
        )
    else:
        sections.append(
            "<h2>1. Primary research comparison</h2>"
            "<p>Run PRECOG_Comparison.py first.</p>"
        )

    if cached:
        speed = statistics.median(
            float(row["speedup"]) for row in cached
        )

        figure = plot(
            "cached_speedup", cached, "tokens",
            [("speedup", "Multi-token / sequential")],
            "Speedup (x)", "Cached continuation optimization",
        )

        sections.append(
            "<h2>2. Cached implementation performance</h2>"
            f"<p>Median speedup: {speed:.2f}x.</p>"
            + figure
            + table(cached, [
                ("topic", "Topic"),
                ("tokens", "Tokens"),
                ("sequential_s", "Sequential s"),
                ("multitoken_s", "Multi-token s"),
                ("speedup", "Speedup"),
            ])
        )

    if correctness:
        cases = correctness.get("cases", [])
        total = sum(row["total"] for row in cases)
        matched = sum(row["matched"] for row in cases)

        generation = correctness.get("generation", [])
        gen_total = sum(row["total"] for row in generation)
        gen_match = sum(row["matched"] for row in generation)

        smoke = (
            "PASS" if correctness.get("smoke_passed")
            else "NOT VERIFIED"
        )

        sections.append(
            "<h2>3. Correctness and features</h2>"
            f"<p>Teacher-forced: {matched}/{total}; "
            f"generated: {gen_match}/{gen_total}; "
            f"file/state smoke test: {smoke}.</p>"
            + table(cases, [
                ("topic", "Topic"),
                ("tokens", "Tokens"),
                ("matched", "Matched"),
                ("total", "Total"),
                ("logit_max", "Logit max difference"),
                ("conv_max", "Conv max difference"),
                ("ssm_max", "SSM max difference"),
            ])
        )

    content = "\n".join(
        f"<section>{section}</section>"
        for section in sections
    )

    # HTML and CSS are page content, not Python comments.
    webpage = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PRECOG Research Evaluation</title>
<style>
body {{ font: 15px system-ui, Arial; background: #f3f5f9;
        color: #243247; margin: 0; }}
header {{ background: #1d2e49; color: white; padding: 28px 5%; }}
main {{ max-width: 1050px; margin: 22px auto; padding: 0 16px; }}
section {{ background: white; border-radius: 12px; padding: 24px;
           margin: 18px 0; box-shadow: 0 1px 6px #0001; }}
figure {{ margin: 20px 0; }}
img {{ width: 100%; max-width: 850px; }}
figcaption {{ font-size: 13px; }}
.scroll {{ overflow: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 9px;
          text-align: left; white-space: nowrap; }}
th {{ background: #edf1f7; }}
a {{ color: #2362a6; }}
</style>
</head>
<body>
<header>
<h1>PRECOG Research Evaluation</h1>
<p>Falcon Mamba 7B · Full-context baseline vs recurrent-state injection</p>
</header>
<main>
{content}
<section>
<h2>Measurement notes</h2>
<p>Online PRECOG time excludes offline document encoding.
GPU charts represent additional peak PyTorch-allocated memory,
not total VRAM. The benchmark uses small synthetic samples,
and prediction agreement is not answer-quality evaluation.
Timings include CPU/GPU result transfers.</p>
<p>Publication figures are saved in results/figures/.</p>
</section>
</main>
</body>
</html>'''

    output = RESULTS / "dashboard.html"
    output.write_text(webpage, encoding="utf-8")

    print("Dashboard:", output)
    print("Figures:", FIGURES)
    webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
