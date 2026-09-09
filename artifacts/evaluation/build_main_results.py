"""Export the reported full-test news results as a slide-ready Matplotlib figure.

Run with a Python environment containing Matplotlib. Values are read from the
shared HTML report, not recomputed from predictions. No model evaluation runs.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aiku-mpl-config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "experiment_summary_20260907.html"
STEM = "news_main_results"


class ReportTables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

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


def news_row(tables, header):
    table, = [table for table in tables if header in table[0]]
    row, = [row for row in table[1:] if row[0] == "news"]
    return dict(zip(table[0], row))


def pair(text):
    values = re.findall(r"(?:\d*\.\d+|\d+)", text)
    if len(values) != 2:
        raise ValueError(f"Expected an SFT/DPO pair, got {text!r}")
    return list(map(float, values))


parser = ReportTables()
parser.feed(SOURCE.read_text(encoding="utf-8"))
main = news_row(parser.tables, "D ASR SFT→DPO")
quality = news_row(parser.tables, "collapse SFT→DPO")
n = int(quality["평가 n"].replace(",", ""))
asr = {
    "RoBERTa": pair(main["D ASR SFT→DPO"]),
    "SCRN": pair(main["SCRN ASR SFT→DPO"]),
}
cosine = pair(main["cosine SFT→DPO"])
collapse = pair(quality["collapse SFT→DPO"])

data = {
    "domain": "news",
    "test_n": n,
    "conditions": ["SFT", "DPOP"],
    "raw_asr_percent": asr,
    "semantic_cosine": cosine,
    "collapse_percent": collapse,
    "rates_are_reported_rounded_values": True,
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sections": ["주 지표 — Raw ASR과 의미 보존",
                        "보수적 참고 — 비붕괴 조건부 ASR (n and collapse only)"],
    "protocol_source": "pipeline/evaluate.py",
    "notes": [
        "Report condition DPO is labeled DPOP to match the project's actual training method.",
        "Raw ASR is the fraction of all outputs passing the detector; collapse is reported separately.",
        "Each detector threshold is the 95th percentile of human scores in the evaluation set, not 0.5.",
        "Semantic cosine compares generator output with the original AI input using ko-sroberta-multitask, truncated at 256 tokens.",
        "Acute collapse: a whitespace-free character 6-gram repeated more than 20 times.",
        "This full-test comparison does not include the separate matched-n100 prompting baseline.",
        "Error bars and exact counts are not reconstructed from rounded aggregate rates.",
    ],
}
(OUT / f"{STEM}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

INK = "#182435"
MUTED = "#627083"
SFT = "#A8BAE8"
DPOP = "#345BCE"
GRID = "#E7ECF2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "text.color": INK,
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
    "svg.hashsalt": "aiku-news-main-results",
})

fig = plt.figure(figsize=(10, 7), facecolor="white")
fig.text(0.5, 0.953, "Main Results — News", ha="center", va="center", fontsize=24, weight="bold")
fig.text(0.5, 0.901, f"Test set: {n:,} texts", ha="center", fontsize=12, color=MUTED)

ax = fig.add_axes([0.12, 0.367, 0.83, 0.413])
width = 0.28
for j, (label, color) in enumerate(zip(data["conditions"], [SFT, DPOP])):
    positions = [i + (j - 0.5) * (width + 0.035) for i in range(len(asr))]
    values = [v[j] for v in asr.values()]
    bars = ax.bar(positions, values, width=width, label=label, color=color, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.2,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=17,
                weight="bold", color=INK)

for i, (sft, dpop) in enumerate(asr.values()):
    y = max(sft, dpop) + 13
    a, b = i - (width + 0.035) / 2, i + (width + 0.035) / 2
    ax.plot([a, a, b, b], [y - 1.8, y, y, y - 1.8], color=MUTED, linewidth=1.1)
    ax.text(i, y + 1.6, f"+{dpop - sft:.1f} pp", ha="center", va="bottom",
            fontsize=12, weight="bold", color=DPOP)

ax.set_xlim(-0.63, 1.63)
ax.set_ylim(0, 106)
ax.set_xticks(range(len(asr)), list(asr))
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
          ncol=2, frameon=False, fontsize=13, handlelength=1.4, columnspacing=2.0)

# A compact numeric table avoids giving cosine and collapse a misleading shared scale.
left, right = 0.16, 0.94
x_metric, x_sft, x_dpop = 0.18, 0.69, 0.865
for y in (0.247, 0.190, 0.125):
    fig.add_artist(Line2D([left, right], [y, y], transform=fig.transFigure,
                          color=GRID, linewidth=0.9))
for x, text, color, align in [(x_metric, "Quality metric", MUTED, "left"),
                              (x_sft, "SFT", MUTED, "center"),
                              (x_dpop, "DPOP", DPOP, "center")]:
    fig.text(x, 0.269, text, ha=align, va="center", fontsize=13, weight="bold", color=color)
for y, label, values, fmt in [(0.218, "Semantic cosine ↑", cosine, ".3f"),
                             (0.155, "Collapse rate ↓", collapse, ".1f")]:
    fig.text(x_metric, y, label, ha="left", va="center", fontsize=14)
    for x, value in zip([x_sft, x_dpop], values):
        suffix = "%" if label.startswith("Collapse") else ""
        fig.text(x, y, f"{value:{fmt}}{suffix}", ha="center", va="center", fontsize=15)

fig.text(0.5, 0.063, "ASR threshold: 95th percentile of human scores (per detector)",
         ha="center", fontsize=10.5, color=MUTED)

description = (
    f"News full test n={n}. SFT vs DPOP: RoBERTa raw ASR {asr['RoBERTa']}, "
    f"SCRN raw ASR {asr['SCRN']}, semantic cosine {cosine}, collapse percent {collapse}. "
    f"Rounded values from {SOURCE.name}. Threshold: per-detector human score p95."
)
fig.savefig(OUT / f"{STEM}.png", dpi=300, facecolor="white", metadata={"Description": description})
fig.savefig(OUT / f"{STEM}.svg", facecolor="white",
            metadata={"Title": "Main Results — News", "Description": description, "Date": None})
plt.close(fig)
print(json.dumps({"n": n, "ASR": asr, "cosine": cosine, "collapse": collapse,
                  "exports": [str(OUT / f"{STEM}.{ext}") for ext in ("png", "svg", "json")]}, indent=2))
