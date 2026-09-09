"""Build slide-ready cross-domain summary and heatmaps from reported aggregates.

Requires Matplotlib and NumPy. No generation or detector inference is performed.
"""
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "cross_domain_current6_full_20260909.html"


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = self.row = self.cell = None
        self.cards = []
        self.card = None
        self.card_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            if self.card is not None:
                self.card_depth += 1
            elif dict(attrs).get("class") == "card":
                self.card, self.card_depth = [], 1
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, text):
        if self.card is not None and text.strip():
            self.card.append(text.strip())
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag == "div" and self.card is not None:
            self.card_depth -= 1
            if self.card_depth == 0:
                self.cards.append(self.card)
                self.card = None
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


p = Tables()
p.feed(SOURCE.read_text(encoding="utf-8"))
detail, = [t for t in p.tables if "D DPO" in t[0] and "source" in t[0]]
rows = [dict(zip(detail[0], row)) for row in detail[1:]]
domains = list(dict.fromkeys(row["source"] for row in rows))
display_names = {"news": "News", "essay": "Essay", "persona": "Persona",
                 "written": "Written", "petition512B": "Petition", "wiki512B": "Wiki"}
labels = [display_names[d] for d in domains]
index = {(r["source"], r["target"]): r for r in rows}
if len(rows) != 36 or len(index) != 36 or len(domains) != 6:
    raise ValueError("Expected a complete 6 by 6 matrix")


def matrix(column):
    return np.array([[float(index[s, t][column].rstrip("%")) for t in domains] for s in domains])


metrics = {
    detector: {condition: matrix(f"{prefix} {report_label}")
               for condition, report_label in [("SFT", "SFT"), ("DPOP", "DPO")]}
    for detector, prefix in [("RoBERTa", "D"), ("SCRN", "SCRN")]
}
diagonal = np.eye(6, dtype=bool)
cell_means = {
    detector: {condition: {"same_domain": float(values[diagonal].mean()),
                          "cross_domain": float(values[~diagonal].mean())}
               for condition, values in matrices.items()}
    for detector, matrices in metrics.items()
}
macros = {d: {c: {} for c in ("SFT", "DPOP")} for d in metrics}
for title, dpop_value, sft_value in p.cards:
    detector = "SCRN" if title.startswith("SCRN") else "RoBERTa"
    group = "cross_domain" if "off-diagonal" in title else "same_domain"
    macros[detector]["DPOP"][group] = float(dpop_value.rstrip("%"))
    macros[detector]["SFT"][group] = float(re.search(r"SFT (\d+\.\d+)%", sft_value)[1])
for detector in macros:
    for condition in macros[detector]:
        if set(macros[detector][condition]) != {"same_domain", "cross_domain"}:
            raise ValueError("Incomplete report macro summary")
        for group, value in macros[detector][condition].items():
            if abs(value - cell_means[detector][condition][group]) > 0.11:
                raise ValueError("Report macro and displayed cells disagree beyond rounding")
data = {
    "source": str(SOURCE.relative_to(ROOT)),
    "source_section": "36개 조합 상세 수치",
    "domain_order": domains,
    "target_test_n": {t: int(index[t, t]["N"].replace(",", "")) for t in domains},
    "raw_asr_percent": {d: {c: v.tolist() for c, v in m.items()} for d, m in metrics.items()},
    "macro_asr_percent": macros,
    "means_of_rounded_cells_for_validation": cell_means,
    "collapse_percent": {c: matrix(f"붕괴 {r}").tolist() for c, r in [("SFT", "SFT"), ("DPOP", "DPO")]},
    "semantic_cosine": {c: matrix(f"cos {r}").tolist() for c, r in [("SFT", "SFT"), ("DPOP", "DPO")]},
    "notes": [
        "Both cell values and macro means are read from the report. Macro means are not re-rounded from already-rounded cell values.",
        "Same-domain is the unweighted mean of 6 diagonal cells; cross-domain is the unweighted mean of 30 off-diagonal cells.",
        "Every source generator is applied to the full test split of each target; source-target combinations reuse target documents.",
        "Scoring uses the target domain's RoBERTa and SCRN, each with its full-human-score 95th-percentile threshold.",
        "DPO in the report is displayed as DPOP; no second-round G2 is included.",
        "Raw ASR includes collapsed outputs. High ASR alone does not establish successful content-preserving transfer.",
        "Semantic cosine is based on 256-token-truncated embeddings and cannot validate full-document quality.",
        "The short display labels Petition and Wiki denote the petition512B and wiki512B tracks.",
        "No uncertainty intervals are reconstructed from rounded aggregate results.",
    ],
}
(OUT / "cross_domain_results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

INK, MUTED, GRID = "#182435", "#627083", "#E7ECF2"
COLORS = {"SFT": "#A8BAE8", "DPOP": "#345BCE"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "svg.hashsalt": "aiku-cross-domain-results",
})


def save(fig, stem, title, description):
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white", metadata={"Description": description})
    fig.savefig(OUT / f"{stem}.svg", facecolor="white",
                metadata={"Title": title, "Description": description, "Date": None})
    plt.close(fig)


# Summary slide: compare matched and unmatched source/target domains on each detector.
fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True, facecolor="white")
fig.subplots_adjust(left=0.095, right=0.965, top=0.735, bottom=0.24, wspace=0.20)
fig.text(0.5, 0.946, "Cross-domain Generalization", ha="center", va="center", fontsize=24, weight="bold")
fig.text(0.5, 0.893, "6 training domains × 6 target domains · Full test sets", ha="center", fontsize=12, color=MUTED)
fig.legend(handles=[Patch(color=COLORS[c], label=c) for c in COLORS],
           loc="center", bbox_to_anchor=(0.5, 0.83), ncol=2, frameon=False, fontsize=13)
for ax, (detector, values) in zip(axes, macros.items()):
    for j, condition in enumerate(COLORS):
        xs = np.array([0, 1]) + (j - 0.5) * 0.36
        heights = [values[condition][group] for group in ("same_domain", "cross_domain")]
        bars = ax.bar(xs, heights, width=0.28, color=COLORS[condition], zorder=3)
        for bar, value in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f"{value:.1f}%",
                    ha="center", va="bottom", fontsize=13, weight="bold")
    ax.set_title(detector, fontsize=16, weight="bold", pad=15)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1], ["Same domain", "Cross domain"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.tick_params(axis="both", length=0, pad=9, labelsize=12, colors=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
axes[0].set_ylabel("Macro raw ASR ↑", labelpad=10, fontsize=14)
fig.text(0.5, 0.142, "Same domain: 6 combinations  |  Cross domain: 30 combinations", ha="center", fontsize=11, color=MUTED)
fig.text(0.5, 0.093, "Equal weight per combination · Target-domain detectors · Human 95th-percentile thresholds", ha="center", fontsize=10, color=MUTED)
save(fig, "cross_domain_summary", "Cross-domain Generalization",
     f"SFT/DPOP same-domain and cross-domain macro raw ASR: {macros}. Source: {SOURCE.name}.")

# Detail slide: DPOP-only heatmaps with a single fixed scale and outlined diagonal.
fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), facecolor="white")
fig.subplots_adjust(left=0.12, right=0.875, bottom=0.23, top=0.78, wspace=0.43)
fig.text(0.5, 0.945, "Cross-domain ASR — DPOP", ha="center", va="center", fontsize=24, weight="bold")
fig.text(0.5, 0.892, "Rows: generator training domain  ·  Columns: target test / detector domain", ha="center", fontsize=12, color=MUTED)
cmap = LinearSegmentedColormap.from_list("aiku_blue", ["#F3F6FC", "#C4D2F2", "#7191DF", "#345BCE", "#182F7A"])
for ax, detector in zip(axes, metrics):
    values = metrics[detector]["DPOP"]
    im = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="equal")
    ax.set_title(detector, fontsize=16, weight="bold", pad=13)
    ax.set_xticks(range(6), labels, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(6), labels)
    ax.tick_params(axis="both", length=0, pad=8, labelsize=11, colors=INK)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", fontsize=11,
                    weight="bold" if i == j else "normal", color="white" if values[i,j] >= 60 else INK)
        ax.add_patch(Rectangle((i-0.47, i-0.47), 0.94, 0.94, fill=False, edgecolor="#162033", linewidth=1.6))
cax = fig.add_axes([0.91, 0.26, 0.015, 0.48])
cbar = fig.colorbar(im, cax=cax, ticks=[0, 20, 40, 60, 80, 100])
cbar.ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
cbar.ax.tick_params(length=0, pad=7, labelsize=10)
cbar.outline.set_visible(False)
cbar.set_label("Raw ASR ↑", labelpad=10, fontsize=12)
fig.text(0.5, 0.11, "Outlined cells: same domain  ·  Cell values: raw ASR (%)", ha="center", fontsize=11, color=MUTED)
fig.text(0.5, 0.06, "Threshold: target-domain human score, 95th percentile  ·  Collapsed outputs included in raw ASR", ha="center", fontsize=10, color=MUTED)
save(fig, "cross_domain_heatmap", "Cross-domain ASR — DPOP",
     f"DPOP RoBERTa/SCRN 6x6 raw ASR; rows train domains, columns target domains. Full held-out tests. Source: {SOURCE.name}.")
print(json.dumps({"macro_asr_percent": macros, "output_stems": ["cross_domain_summary", "cross_domain_heatmap"]}, indent=2))
