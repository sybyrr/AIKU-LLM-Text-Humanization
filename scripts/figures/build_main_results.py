"""Plot full-test news metrics from results/metrics/news_main_results.json."""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figures"
METRICS = ROOT / "results/metrics"
OUT.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)
STEM = "news_main_results"
SOURCE = METRICS / f"{STEM}.json"
data = json.loads(SOURCE.read_text(encoding="utf-8"))
n = data["test_n"]
asr = data["raw_asr_percent"]
cosine = data["semantic_cosine"]
collapse = data["collapse_percent"]
if data["conditions"] != ["SFT", "DPOP"] or set(asr) != {"RoBERTa", "SCRN"}:
    raise ValueError("Unexpected news comparison conditions")
if any(len(values) != 2 for values in [*asr.values(), cosine, collapse]):
    raise ValueError("Every metric must contain an SFT/DPOP pair")

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
                  "exports": [str(OUT / f"{STEM}.{ext}") for ext in ("png", "svg")] + [str(METRICS / f"{STEM}.json")]}, indent=2))
