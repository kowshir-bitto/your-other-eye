
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

CAT = ["#1f6fd0", "#d4442a", "#12907d", "#8b5cf6", "#b8860b"]
BLUE, RED, TEAL, PURPLE, GOLD = CAT

INK = "#1a1a1a"
INK_2 = "#4a4a4a"
MUTED = "#8a8a8a"
GRID = "#e4e4e2"
SURFACE = "#fcfcfb"
CHANCE = "#9a9a9a"

FIGDIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "figures"


def use_paper_style(base: float = 9.0) -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name: str, also_pdf: bool = True) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / f"{name}.png"
    fig.savefig(png)
    if also_pdf:
        fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"    -> {png.name}" + ("  + .pdf" if also_pdf else ""))
    return png


def annotate(ax, x, y, text, dx=0, dy=8, color=INK, weight="normal", size=None):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha="center", va="bottom", color=color, fontweight=weight,
                fontsize=size)
