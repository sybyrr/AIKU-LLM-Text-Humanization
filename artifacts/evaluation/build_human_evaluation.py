"""Export an illustrative allocation grid and source-grounded human-eval chart.

The allocation CSV is unavailable locally, so the grid is explicitly schematic.
Actual response counts and intervals are read from the shared Markdown report.
Requires Matplotlib.
"""
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import NormalDist
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aiku-mpl-config"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "human_eval_results_share_20260907.md"
lines = SOURCE.read_text(encoding="utf-8").splitlines()
start, = [i for i, line in enumerate(lines) if line.startswith("| 실제 버전 |")]
rows = []
for line in lines[start+2:]:
    if not line.startswith("|"):
        break
    fields = [v.strip() for v in line.strip("|").split("|")]
    variant = fields[0].strip("*")
    match = re.fullmatch(r"(\d+)/(\d+) \((\d+)%\)", fields[1].strip("*"))
    successes, n, percent = map(int, match.groups())
    lo, hi = map(float, fields[3].rstrip("%").split("–"))
    if not math.isclose(successes / n * 100, percent, abs_tol=1e-10):
        raise ValueError("Reported rate disagrees with response counts")
    z = NormalDist().inv_cdf(0.975)
    phat, denom = successes / n, 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / denom
    computed = [round(100*(center-half), 1), round(100*(center+half), 1)]
    if computed != [lo, hi]:
        raise ValueError(f"Reported Wilson interval mismatch: {variant}")
    rows.append({"variant": variant, "judged_human": successes, "n": n,
                 "human_judgment_percent": percent, "wilson_95_percent": [lo, hi]})
variants = [row["variant"] for row in rows]
if variants != ["Human", "G0", "G1", "G2"] or sum(row["n"] for row in rows) != 200:
    raise ValueError("Unexpected evaluation design")

# Show only example rows. These labels are not the actual private allocation.
seed = 20260907
rng = random.Random(seed)
all_permutations = list(itertools.permutations(variants))
while True:
    permutations = rng.sample(all_permutations, 5)
    # Keep the schematic visually varied; this is not a claim about actual balance.
    if all(max([row[c] for row in permutations].count(v) for v in variants) <= 2
           for c in range(4)):
        break
illustrative = dict(zip([1, 2, 3, 4, 50], permutations))
if any(set(v) != set(variants) for v in illustrative.values()):
    raise ValueError("Each illustrated article must have all four variants once")
data = {
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sections": ["1. 조사 설계와 응답 현황", "2. 주요 결과"],
    "design": {"common_articles": 50, "raters": 4, "versions_per_article": 4,
               "judgments_per_rater": 50, "total_judgments": 200,
               "task": "Binary judgment: written by AI or by a human",
               "one_rater_per_article_version": True},
    "allocation_diagram": {"illustrative_only": True,
                           "actual_master_allocation_csv_available": False,
                           "example_seed": seed,
                           "display_note": "Example rows chosen for visual variety, not evidence of actual rater balance",
                           "rater_column_order": ["R1", "R2", "R3", "R4"],
                           "shown_article_rows": illustrative},
    "results": rows,
    "notes": [
        "Allocation cells demonstrate the design; they do not reproduce the missing master_allocation.csv.",
        "The figure shows example articles 1-4 and 50, with an ellipsis for omitted rows.",
        "Participants saw blinded document codes and text; version labels and detector scores were hidden.",
        "G0 denotes original AI text, G1 the first DPOP generator, and G2 the additionally trained generator.",
        "This measures perceived authorship, not naturalness, factual accuracy, or content preservation directly.",
        "Reported Wilson binomial 95% intervals are reproduced and independently checked from counts.",
        "These descriptive per-condition intervals do not account for article pairing or rater clustering.",
        "Each article/version received one judgment; this design cannot estimate per-document inter-rater agreement.",
        "No significance test or definitive model ranking is inferred from interval overlap.",
    ],
}
(OUT / "human_evaluation.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

INK, MUTED, GRID = "#182435", "#627083", "#E7ECF2"
COLORS = {"Human": "#2A8C7B", "G0": "#BD8540", "G1": "#A8BAE8", "G2": "#345BCE"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "text.color": INK,
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "svg.hashsalt": "aiku-human-evaluation",
})


def save(fig, stem, title, description):
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white", metadata={"Description": description})
    fig.savefig(OUT / f"{stem}.svg", facecolor="white",
                metadata={"Title": title, "Description": description, "Date": None})
    plt.close(fig)


fig = plt.figure(figsize=(9.5, 6.6), facecolor="white")
fig.text(0.5, 0.945, "Human Evaluation — Design", ha="center", va="center", fontsize=24, weight="bold")
fig.text(0.5, 0.883, "50 articles × 4 versions · 4 raters · 200 judgments", ha="center", fontsize=12, color=MUTED)
ax = fig.add_axes([0.205, 0.26, 0.745, 0.50])
shown = [1, 2, 3, 4, None, 50]
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(5.5, -0.5)
ax.set_xticks(range(4), ["R1", "R2", "R3", "R4"])
ax.xaxis.tick_top()
ax.set_yticks(range(6), [f"Article {v:02d}" if v is not None else "⋮" for v in shown])
ax.tick_params(axis="x", length=0, labelsize=16, pad=14)
ax.tick_params(axis="y", length=0, labelsize=13, pad=14)
for label in ax.get_xticklabels():
    label.set_weight("bold")
for spine in ax.spines.values():
    spine.set_visible(False)
for r, article in enumerate(shown):
    for c in range(4):
        if article is None:
            ax.text(c, r, "⋮", ha="center", va="center", fontsize=23, color=MUTED)
            continue
        variant = illustrative[article][c]
        ax.add_patch(Rectangle((c-0.475, r-0.445), 0.95, 0.89, facecolor=COLORS[variant], linewidth=0))
        ax.text(c, r, variant, ha="center", va="center", fontsize=16, weight="bold",
                color=INK if variant in ("G0", "G1") else "white")
fig.text(0.5, 0.171, "One version of each article per rater · All four versions covered", ha="center", fontsize=12, color=INK)
fig.text(0.5, 0.10, "Illustrative allocation; actual assignment records are not shown", ha="center", fontsize=10.5, color=MUTED)
fig.text(0.5, 0.057, "Version labels were hidden from participants", ha="center", fontsize=10.5, color=MUTED)
save(fig, "human_evaluation_design", "Human Evaluation — Design",
     "Illustrative, not actual, allocation grid. Rows are common articles; columns are four raters. Each article has Human/G0/G1/G2 assigned once. Actual study: 50 articles, 200 judgments. Source: human_eval_results_share_20260907.md; master_allocation.csv unavailable.")


fig, ax = plt.subplots(figsize=(10, 6.6), facecolor="white")
fig.subplots_adjust(left=0.13, right=0.97, top=0.79, bottom=0.245)
fig.text(0.5, 0.945, "Human Evaluation — Results", ha="center", va="center", fontsize=24, weight="bold")
fig.text(0.5, 0.883, "50 judgments per version · 4 raters", ha="center", fontsize=12, color=MUTED)
values = [row["human_judgment_percent"] for row in rows]
bars = ax.bar(range(4), values, width=0.56, color=[COLORS[v] for v in variants], zorder=3)
yerr = [[row["human_judgment_percent"] - row["wilson_95_percent"][0] for row in rows],
        [row["wilson_95_percent"][1] - row["human_judgment_percent"] for row in rows]]
ax.errorbar(range(4), values, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1.5,
            capsize=5, capthick=1.5, zorder=4)
for i, row in enumerate(rows):
    ax.text(i, row["wilson_95_percent"][1] + 2.7, f"{row['human_judgment_percent']}%",
            ha="center", va="bottom", fontsize=19, weight="bold")
ax.set_xlim(-0.65, 3.65)
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.set_ylabel("Judged as human-written", fontsize=14, labelpad=12)
ax.set_xticks(range(4), ["Human\nOriginal text", "G0\nOriginal AI", "G1\nFirst DPOP", "G2\nAdditional DPOP"])
ax.tick_params(axis="both", length=0, labelsize=12.5, pad=10, colors=INK)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
fig.text(0.5, 0.074, "Error bars: Wilson 95% confidence intervals", ha="center", fontsize=11, color=MUTED)
save(fig, "human_evaluation_results", "Human Evaluation — Results",
     f"Reported human-written judgment rates and Wilson 95% intervals: {rows}. Perceived authorship, not a direct quality rating. Source: {SOURCE.name}.")
print(json.dumps({"results": rows, "illustrative_allocation": illustrative,
                  "output_stems": ["human_evaluation_design", "human_evaluation_results"]}, indent=2))
