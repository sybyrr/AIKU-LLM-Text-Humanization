"""Plot detector adaptation and the incremental effect of a second DPOP phase.

Reads reported aggregates from loop/RESULTS.md. Requires Matplotlib. This does
not rerun models or claim to validate unavailable per-document predictions.
"""
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aiku-mpl-config"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "loop/RESULTS.md"
source = SOURCE.read_text(encoding="utf-8")


def table(header):
    lines = source.splitlines()
    start = lines.index(header)
    headers = [v.strip() for v in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append(dict(zip(headers, [v.strip() for v in line.strip("|").split("|")])))
    return rows


def number(value):
    return float(value.replace("−", "-").rstrip("%p"))


internal = table("| 평가 축 | G1 ASR | G2 ASR | G2−G1 | 해석 |")
external = table("| 변형 | 평균 AI작성률 | AI 판정률 | ASR |")
n = int(re.search(r"동일 test ID ([\d,]+)개", source)[1].replace(",", ""))
rows = []
for source_label, detector, role in [
    ("frozen team detector", "D0", "Frozen"), ("D1", "D1", "Adapted"),
]:
    row, = [r for r in internal if r["평가 축"] == source_label]
    rows.append({"detector": detector, "role": role,
                 "G1_asr_percent": number(row["G1 ASR"]),
                 "G2_asr_percent": number(row["G2 ASR"]),
                 "delta_pp": number(row["G2−G1"])})
ck = {r["변형"]: r for r in external}
g1_ai = int(re.search(r"\((\d+)/", ck["G1"]["AI 판정률"])[1])
g2_ai = int(re.search(r"\((\d+)/", ck["G2"]["AI 판정률"])[1])
for variant, count in [("G1", g1_ai), ("G2", g2_ai)]:
    if round(100 * (1 - count / n), 2) != number(ck[variant]["ASR"]):
        raise ValueError("CopyKiller counts do not agree with reported ASR")
rows.append({"detector": "CopyKiller", "role": "External",
             "G1_asr_percent": number(ck["G1"]["ASR"]),
             "G2_asr_percent": number(ck["G2"]["ASR"]),
             "G1_ai_count": g1_ai, "G2_ai_count": g2_ai,
             "delta_pp": 100 * (g1_ai - g2_ai) / n})
detected = {r["detector"]: round(100 - r["G1_asr_percent"], 2) for r in rows[:2]}
data = {
    "domain": "news", "matched_test_n": n,
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sections": ["D1 게이트", "G1과 G2 비교", "CopyKiller 전수 paired 평가"],
    "conditions": {"G1": "Existing generator after SFT and first DPOP",
                   "G2": "G1 further trained by DPOP using D1-mined hard negatives"},
    "G1_ai_detection_rate_percent": detected,
    "generator_comparison": rows,
    "notes": [
        "Objective: test whether a generator improves with additional DPOP informed by a detector adapted to its outputs.",
        "G1 and G2 are evaluated on the same 1985 held-out news document IDs.",
        "D1 was trained using human, original AI, and G1 outputs from train documents, not the test outputs plotted here.",
        "D1's demonstrated detection improvement is specific to G1 outputs in this news experiment.",
        "D1 threshold 0.9996544122695923 was calibrated on dev-cal; independent dev-gate human FPR was 6.31%.",
        "The existing D0 retains its own evaluation protocol; it is not relabeled as a newly calibrated detector.",
        "CopyKiller is an external detector not used for generator training; its AI threshold is 50%.",
        "CopyKiller delta is computed from the 16/1985 difference in detections; rounded ASR endpoints differ by 0.80 pp.",
        "The 1985-document loop evaluation differs from the main 2044-document evaluation in data and protocol.",
        "Reported aggregates, not a fresh rerun. No significance or confidence intervals are inferred for D0 or D1.",
        "No second adapted detector D2, cross-domain G2 evaluation, or full semantic-quality results are supplied here.",
    ],
}
(OUT / "adversarial_results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

INK, MUTED, GRID = "#182435", "#627083", "#E7ECF2"
LIGHT, BLUE = "#A8BAE8", "#345BCE"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "text.color": INK,
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "svg.hashsalt": "aiku-adversarial-results",
})


def style(ax, ylabel):
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel(ylabel, fontsize=14, labelpad=12)
    ax.tick_params(axis="both", length=0, labelsize=13, pad=10, colors=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def save(fig, stem, title, description):
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white", metadata={"Description": description})
    fig.savefig(OUT / f"{stem}.svg", facecolor="white",
                metadata={"Title": title, "Description": description, "Date": None})
    plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 5.8), facecolor="white")
fig.subplots_adjust(left=0.135, right=0.96, bottom=0.23, top=0.79)
fig.text(0.5, 0.945, "Detector Adaptation", ha="center", va="center", fontsize=24, weight="bold")
fig.text(0.5, 0.875, f"Same G1 outputs · News test n = {n:,}", ha="center", fontsize=12, color=MUTED)
bars = ax.bar([0, 1], list(detected.values()), width=0.52, color=[LIGHT, BLUE], zorder=3)
for bar, value in zip(bars, detected.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2.3, f"{value:.2f}%",
            ha="center", va="bottom", fontsize=20, weight="bold")
ax.set_xlim(-0.65, 1.65)
ax.set_xticks([0, 1], ["D0\nFrozen detector", "D1\nAdapted detector"])
style(ax, "AI detection rate on G1 outputs ↑")
fig.text(0.5, 0.064, "D1 learned from G1 outputs on training documents", ha="center", fontsize=11, color=MUTED)
save(fig, "detector_adaptation", "Detector Adaptation",
     f"AI detection rates on the same {n} held-out G1 outputs: {detected}. Values equal 100 minus reported raw ASR. Source: {SOURCE.relative_to(ROOT)}.")


fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.235, top=0.765)
fig.text(0.5, 0.945, "Additional DPOP — Generator Performance", ha="center", va="center", fontsize=23, weight="bold")
fig.text(0.5, 0.889, f"Same news test documents · n = {n:,}", ha="center", fontsize=12, color=MUTED)
fig.legend(handles=[Patch(color=LIGHT, label="G1 · First DPOP"), Patch(color=BLUE, label="G2 · Additional DPOP")],
           loc="center", bbox_to_anchor=(0.5, 0.829), ncol=2, frameon=False, fontsize=13, columnspacing=2)
for j, (condition, color) in enumerate([("G1", LIGHT), ("G2", BLUE)]):
    positions = [i + (j - 0.5) * 0.36 for i in range(3)]
    values = [row[f"{condition}_asr_percent"] for row in rows]
    bars = ax.bar(positions, values, width=0.29, color=color, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.1, f"{value:.2f}%",
                ha="center", va="bottom", fontsize=14, weight="bold")
ax.set_xlim(-0.6, 2.6)
ax.set_xticks(range(3), ["D0\nFrozen detector", "D1\nAdapted detector", "CopyKiller\nExternal detector"])
style(ax, "Raw ASR ↑")
fig.text(0.5, 0.071, "G2 uses D1-selected hard negatives · CopyKiller is used only for evaluation",
         ha="center", fontsize=11, color=MUTED)
save(fig, "adversarial_generator_comparison", "Additional DPOP — Generator Performance",
     f"Paired G1 vs G2 raw ASR on {n} news documents. {rows}. Source: {SOURCE.relative_to(ROOT)}. Detector-specific thresholds; CopyKiller threshold 50%.")
print(json.dumps({"test_n": n, "detection_on_G1": detected, "comparison": rows,
                  "output_stems": ["detector_adaptation", "adversarial_generator_comparison"]}, indent=2))
