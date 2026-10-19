#!/usr/bin/env python3
"""Decision Index 0.3 chart (served params x Full) for the Crivo model cards.

Frontier models are drawn solid and joined by the staircase (best Full at or below each size); the rest are faded.
Colors follow the board's architecture palette (multimodalart/jev-decision-index); each architecture
also gets its own marker shape, because blue and purple are hard to tell apart with colour blindness.

  curl -sL https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json -o index.json
  uv run --no-project --python 3.12 --with matplotlib python plot_pareto.py index.json crivo-150m/assets/pareto.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

SURFACE, INK, INK2, GRID = "#FBF8F1", "#232323", "#55524B", "#E3DED2"
HERO, HERO_FILL = "Crivo-150M", "#FFD21E"
ARCH = {  # technique -> (label, colour, marker)
    "autoregressive": ("autoregressive", "#3B7CF6", "o"),
    "encoder": ("encoder", "#3E8E5E", "D"),
    "GLi family": ("GLi family", "#FF9D00", "^"),
    "diffusion": ("diffusion", "#A855F7", "s"),
}
XMAX = 4.2e10  # just past the largest submission (~36B)
LEAD_UNTIL = 7.5e8  # nothing below this size scores higher than Crivo-150M (checked in main)

# labels: name -> (text, offset in points)
LABELS = {
    "Bekko System One v0 17M": ("Bekko 17M", (6, -13)),
    "Bekko System One v0 68M": ("Bekko 68M", (-9, 3)),
    "tasksource Decider nano (150M)": ("tasksource nano", (-10, 3)),
    "Dinah-0": ("Dinah-0", (10, -3)),
    "Supersonic Labs Julia 1": ("Julia-1", (-10, -1)),
    "Bekko System One v0 400M": ("Bekko 400M", (-10, 3)),
    "vLLM-SR d3-edge (0.6B)": ("vLLM-SR d3-edge 0.6B", (-10, 3)),
    "jiwo 0.8B v2": ("jiwo 0.8B", (-10, 3)),
    "vLLM-SR d3-lite (0.8B)": ("vLLM-SR d3-lite 0.8B", (-10, 3)),
    "LiquidAI d1-3B": ("LiquidAI d1-3B", (8, -10)),
    "vLLM-SR d3 (27B)": ("vLLM-SR d3 27B", (-10, 3)),
}
# labels placed at (x, y) in data units with a leader line: name -> (text, xy, ha)
PLACED = {
    "LiquidAI d1-omni-600M": ("LiquidAI d1-omni 0.6B", (9.5e8, 1.4), "left"),
    "Together Tev1 Qwen3.5-0.8B": ("Together Tev1 0.8B", (1.6e9, 9.0), "left"),
    "Cloudflare clef": ("Cloudflare\nClef 28B", (1.35e10, 33.0), "left"),
    "Fastino GLiDE no-thinking (28B)": ("Fastino GLiDE 28B", (1.6e10, 64.5), "right"),
}
SIZE_LABEL = {1e7: "10M", 3e7: "30M", 1e8: "100M", 3e8: "300M", 1e9: "1B", 3e9: "3B", 1e10: "10B",
              3e10: "30B", 1e11: "100B"}


def load(src):
    d = json.load(open(src))
    rows = []
    for m in d["models"]:
        v, meta = m.get("v03") or {}, m.get("meta") or {}
        if v.get("v") is not None and meta.get("served_params"):
            rows.append({"p": meta["served_params"], "v": v["v"], "name": v["name"],
                         "arch": meta.get("technique") if meta.get("technique") in ARCH else "autoregressive"})
    return d, rows


def frontier(rows):
    out, best = [], -1.0
    for r in sorted(rows, key=lambda r: (r["p"], -r["v"])):
        if r["v"] > best:
            best = r["v"]
            out.append(r)
    return out


def style(ax, xlim, ylim, ticks):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([SIZE_LABEL[t] for t in ticks])
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.set_xlabel("Served parameters (log scale)", color=INK2, fontsize=9.5)


def main(src, out):
    d, rows = load(src)
    front = frontier(rows)
    hero = next(r for r in rows if r["name"] == HERO)
    above = [r for r in rows if r["p"] < LEAD_UNTIL and r["v"] > hero["v"]]
    assert not above, above

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(13, 7.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax, (1.2e7, XMAX), (-2.5, 72), [3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10])
    ax.set_ylabel("Full score (0–100)", color=INK2, fontsize=10)

    # Pareto frontier as the board draws it: a line through the frontier models, the rest faded
    fx, fy = [], []  # staircase: best Full at or below each size
    for i, r in enumerate(front):
        if i:
            fx.append(r["p"]); fy.append(front[i - 1]["v"])
        fx.append(r["p"]); fy.append(r["v"])
    fx.append(XMAX); fy.append(front[-1]["v"])
    ax.plot(fx, fy, color=INK, lw=2, zorder=4, solid_joinstyle="miter")
    on = {r["name"] for r in front}
    for key, (_, color, marker) in ARCH.items():
        for faded in (True, False):
            pts = [r for r in rows if r["arch"] == key and r["name"] != HERO and (r["name"] not in on) == faded]
            ax.scatter([r["p"] for r in pts], [r["v"] for r in pts], s=55 if faded else 85, color=color,
                       marker=marker, edgecolors=INK, linewidths=0.6 if faded else 1.2,
                       alpha=0.3 if faded else 1, zorder=3 if faded else 5)
    ax.scatter([hero["p"]], [hero["v"]], s=420, color=HERO_FILL, marker="D", edgecolors=INK, linewidths=2,
               zorder=6)
    ax.annotate(f"Crivo-150M\nFull {hero['v']:.2f}", (hero["p"], hero["v"]), xytext=(-4, 26),
                textcoords="offset points", ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=INK, zorder=7)

    for r in rows:
        if r["name"] in LABELS:
            text, (dx, dy) = LABELS[r["name"]]
            ax.annotate(text, (r["p"], r["v"]), xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx > 0 else "right", va="center", fontsize=8.5, color=INK2, zorder=5)
        if r["name"] in PLACED:
            text, xy, ha = PLACED[r["name"]]
            ax.annotate(text, (r["p"], r["v"]), xytext=xy, textcoords="data", ha=ha, va="center",
                        fontsize=8.5, color=INK2, zorder=5,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkA=2, shrinkB=5))
    missing = (set(LABELS) | set(PLACED)) - {r["name"] for r in rows}
    assert not missing, missing

    handles = [Line2D([], [], ls="", marker=m, markersize=8, markerfacecolor=c, markeredgecolor=INK,
                      markeredgewidth=0.7, label=lab) for lab, c, m in ARCH.values()]
    handles.append(Line2D([], [], ls="", marker="D", markersize=11, markerfacecolor=HERO_FILL,
                          markeredgecolor=INK, markeredgewidth=1.6, label=HERO))
    handles.append(Line2D([], [], color=INK, lw=2, label="Pareto frontier (no smaller model scores higher)"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2,
              borderaxespad=0.8)

    fig.suptitle("Decision Index 0.3 · Full score vs. model size", x=0.01, ha="left", fontsize=16,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.915, f"{len(rows)} submissions with a known size · board snapshot {d['generated_utc'][:10]} · "
             "Full = 20% public + 50% private same-skill + 30% private new-domain tasks",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    main(*sys.argv[1:])
