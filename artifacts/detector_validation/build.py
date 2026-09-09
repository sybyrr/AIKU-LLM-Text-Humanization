"""Build a source-grounded detector validation figure and an offline HTML viewer.

Requires Matplotlib. Run from any directory: python path/to/build.py
Values are rounded rates transcribed from notes/50-파이프라인-현황.md:102–109;
they are not recomputed from per-example predictions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aiku-mpl-config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


OUT = Path(__file__).resolve().parent
STEM = "detector_validation"
INK = "#182435"
MUTED = "#627083"
BLUE = "#345BCE"
GRID = "#E7ECF2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "text.color": INK,
    "axes.unicode_minus": False,
    "svg.fonttype": "path",  # Keep exported labels independent of installed fonts.
    "svg.hashsalt": "aiku-detector-validation",
    "pdf.fonttype": 42,
})

data = {
    "detector": "klue/roberta-base (fine-tuned)",
    "domain": "news",
    "training_article_ids": 750,
    "training_texts": {"human": 750, "qwen": 750, "exaone": 750},
    "held_out_article_ids": 2250,
    "threshold_ai": 0.5,
    "stage": "Before detector gating and generator rewriting; failed generations excluded",
    "rates_are_rounded": True,
    "rows": [
        {"label": "인간 원문", "metric": "TNR", "correct_prediction": "Human → Human", "correct_percent": 85},
        {"label": "Qwen3-8B", "metric": "TPR", "correct_prediction": "AI → AI", "correct_percent": 99},
        {"label": "EXAONE-3.5-7.8B", "metric": "TPR", "correct_prediction": "AI → AI", "correct_percent": 100},
    ],
    "human_fpr_percent": 15,
    "source": "notes/50-파이프라인-현황.md:102–109",
    "protocol_source": "scripts/news_gate_frozen.py:110–197",
    "note": "Reported rounded rates only. Exact confusion counts, confidence intervals and an ROC curve are not reconstructed.",
}
(OUT / f"{STEM}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="white")
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.79)
fig.suptitle("Detector Validation", y=0.955, fontsize=23, weight="bold")
fig.text(0.5, 0.872, "Threshold: P(AI) ≥ 0.5", ha="center", fontsize=12, color=MUTED)

labels = ["Human", "Qwen3-8B", "EXAONE-3.5-7.8B"]
values = [row["correct_percent"] for row in data["rows"]]
bars = ax.bar(labels, values, width=0.58, color=BLUE, zorder=3)
ax.set_ylim(0, 110)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.set_ylabel("Correct classification rate", fontsize=13, labelpad=10)
ax.tick_params(axis="both", length=0, labelsize=12, pad=9, colors=INK)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0,
            f"{value}%", ha="center", va="bottom", fontsize=18,
            weight="bold", color=INK)

metadata = {"Description": "Reported rounded news detector validation rates: Human TNR 85%, Qwen TPR 99%, EXAONE TPR 100%; P(AI) >= 0.5. Source: notes/50-파이프라인-현황.md:102–109."}
fig.savefig(OUT / f"{STEM}.png", dpi=300, facecolor="white", metadata=metadata)
fig.savefig(OUT / f"{STEM}.svg", facecolor="white", metadata={**metadata, "Title": "Detector Validation", "Date": None})
plt.close(fig)

svg = (OUT / f"{STEM}.svg").read_text(encoding="utf-8")
svg = svg[svg.index("<svg"):]
svg = svg.replace("<svg ", '<svg role="img" aria-label="Detector Validation: Human 85%, Qwen3-8B 99%, EXAONE-3.5-7.8B 100%; threshold 0.5" ', 1)
page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Detector Validation</title>
  <style>
    body { margin: 0; background: white; }
    main { max-width: 1080px; margin: 0 auto; }
    svg { display: block; width: 100%; height: auto; }
    @media print { @page { margin: 0; } main { max-width: none; } }
  </style>
</head>
<body><main>__SVG__</main></body>
</html>
""".replace("__SVG__", svg)
(OUT / f"{STEM}.html").write_text(page, encoding="utf-8")
print(f"Created {OUT / STEM}.{{html,png,svg,json}}")
