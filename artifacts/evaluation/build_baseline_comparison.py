"""Plot the shared report's matched-n100 news prompting baseline comparison.

Requires Matplotlib. Reads reported aggregates; does not run model evaluation.
"""

from html.parser import HTMLParser
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aiku-mpl-config"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "humanizer_skill_baseline_n100_20260909.html"
STEM = "news_baseline_comparison"


class ReportTables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, value):
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


parser = ReportTables()
parser.feed(SOURCE.read_text(encoding="utf-8"))
table, = [t for t in parser.tables if all(c in t[0] for c in ("도메인", "방법", "N", "D ASR"))]
rows = [dict(zip(table[0], row)) for row in table[1:] if row[0] == "news"]
conditions = [("Humanizer-skill", "Prompting baseline"), ("SFT", "SFT"), ("DPO", "DPOP")]
data_rows = []
for report_label, label in conditions:
    row, = [r for r in rows if r["방법"] == report_label]
    data_rows.append({
        "condition": label,
        "source_condition": report_label,
        "n": int(row["N"]),
        "raw_asr_percent": {"RoBERTa": float(row["D ASR"].rstrip("%")),
                            "SCRN": float(row["SCRN ASR"].rstrip("%"))},
        "semantic_cosine": float(row["의미 cos"]),
        "collapse_percent": float(row["급성붕괴"].rstrip("%")),
    })
sample_sizes = {row["n"] for row in data_rows}
if len(sample_sizes) != 1:
    raise ValueError("Comparison conditions have different sample sizes")
n, = sample_sizes
data = {
    "domain": "news",
    "matched_test_n": n,
    "rows": data_rows,
    "source": str(SOURCE.relative_to(ROOT)),
    "source_section": "도메인별 상세 결과",
    "rates_are_reported_rounded_values": True,
    "protocol_source": "scripts/eval_humanizer_skill_baseline.py",
    "notes": [
        "All three conditions are compared on the same 100 document IDs.",
        "Prompting baseline is the project's Humanizer-skill v1.6.0 batch prompt adaptation.",
        "The source report labels DPOP as DPO.",
        "Each detector's ASR threshold is the 95th percentile of human scores on the sampled documents.",
        "ASR is raw detector pass rate, with collapse reported separately.",
        "Semantic cosine compares output with original AI input; embeddings are truncated at 256 tokens.",
        "Acute collapse: a whitespace-free character 6-gram repeated more than 20 times.",
        "The separate n=2044 full-test results are not used in this comparison.",
    ],
}
(OUT / f"{STEM}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

INK, MUTED, GRID = "#182435", "#627083", "#E7ECF2"
COLORS = ["#BD8540", "#A8BAE8", "#345BCE"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "text.color": INK,
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "svg.hashsalt": "aiku-news-baseline-comparison",
})
fig = plt.figure(figsize=(10, 7), facecolor="white")
fig.text(0.5, 0.953, "Baseline Comparison — News", ha="center", va="center", fontsize=24, weight="bold")
fig.text(0.5, 0.901, f"Same {n:,} test texts across all methods", ha="center", fontsize=12, color=MUTED)

ax = fig.add_axes([0.12, 0.367, 0.83, 0.413])
detectors = ["RoBERTa", "SCRN"]
width, gap = 0.21, 0.035
for j, (row, color) in enumerate(zip(data_rows, COLORS)):
    positions = [i + (j - 1) * (width + gap) for i in range(len(detectors))]
    values = [row["raw_asr_percent"][d] for d in detectors]
    bars = ax.bar(positions, values, width=width, label=row["condition"], color=color, zorder=3)
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        if value == 0:
            # An explicit baseline marker makes a genuine 0% distinguishable from missing data.
            ax.plot([x - width / 2, x + width / 2], [0, 0], color=color,
                    linewidth=3, solid_capstyle="butt", clip_on=False, zorder=4)
        ax.text(x, value + 2.5, f"{value:.1f}%", ha="center", va="bottom",
                fontsize=15, weight="bold", color=color if value == 0 else INK)

ax.set_xlim(-0.63, 1.63)
ax.set_ylim(0, 106)
ax.set_xticks(range(len(detectors)), detectors)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.set_ylabel("Raw ASR ↑", fontsize=14, labelpad=12)
ax.tick_params(axis="both", length=0, labelsize=13, pad=9, colors=INK)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="center", bbox_to_anchor=(0.5, 0.836), bbox_transform=fig.transFigure,
          ncol=3, frameon=False, fontsize=13, handlelength=1.4, columnspacing=1.8)

x_metric = 0.18
xs = [0.57, 0.735, 0.89]
for y in (0.247, 0.190, 0.125):
    fig.add_artist(Line2D([0.16, 0.96], [y, y], transform=fig.transFigure, color=GRID, linewidth=0.9))
fig.text(x_metric, 0.269, "Quality metric", ha="left", va="center", fontsize=13, weight="bold", color=MUTED)
for x, label, color in zip(xs, ["Prompting", "SFT", "DPOP"], [COLORS[0], MUTED, COLORS[2]]):
    fig.text(x, 0.269, label, ha="center", va="center", fontsize=13, weight="bold", color=color)
for y, label, key, fmt, suffix in [
    (0.218, "Semantic cosine ↑", "semantic_cosine", ".3f", ""),
    (0.155, "Collapse rate ↓", "collapse_percent", ".1f", "%"),
]:
    fig.text(x_metric, y, label, ha="left", va="center", fontsize=14)
    for x, row in zip(xs, data_rows):
        fig.text(x, y, f"{row[key]:{fmt}}{suffix}", ha="center", va="center", fontsize=15)

fig.text(0.5, 0.063, "ASR threshold: 95th percentile of human scores (per detector)",
         ha="center", fontsize=10.5, color=MUTED)
description = (
    f"Matched news test n={n}, prompting baseline / SFT / DPOP. "
    f"Source: {SOURCE.name}, domain-specific results. "
    "Prompting baseline is the project adaptation of Humanizer-skill v1.6.0. "
    "RoBERTa ASR: 0.0 / 73.0 / 78.0%; SCRN ASR: 0.0 / 63.0 / 66.0%."
)
fig.savefig(OUT / f"{STEM}.png", dpi=300, facecolor="white", metadata={"Description": description})
fig.savefig(OUT / f"{STEM}.svg", facecolor="white",
            metadata={"Title": "Baseline Comparison — News", "Description": description, "Date": None})
plt.close(fig)
print(json.dumps({"rows": data_rows, "exports": [str(OUT / f"{STEM}.{ext}") for ext in ("png", "svg", "json")]}, indent=2))
