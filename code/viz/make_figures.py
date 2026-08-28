
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS
from viz.style import (BLUE, CAT, CHANCE, GOLD, GRID, INK, INK_2, MUTED,
                       PURPLE, RED, TEAL, annotate, save, use_paper_style)

R = PATHS.results


def _read(name: str) -> pd.DataFrame | None:
    p = R / name
    if not p.is_file():
        print(f"    [skip] {name} not found")
        return None
    return pd.read_csv(p)


def f1_concept():
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def eye(cx, cy, r, face, edge, label, sub):
        ax.add_patch(plt.Circle((cx, cy), r, facecolor=face, edgecolor=edge,
                                linewidth=1.8, zorder=3))
        ax.add_patch(plt.Circle((cx + r * 0.42, cy + r * 0.06), r * 0.20,
                                facecolor="#f6e2b8", edgecolor=edge,
                                linewidth=0.8, zorder=4))
        ax.text(cx, cy - r - 0.32, label, ha="center", va="top",
                fontsize=9, fontweight="bold", color=INK)
        ax.text(cx, cy - r - 0.62, sub, ha="center", va="top",
                fontsize=8, color=INK_2)

    eye(1.6, 3.1, 0.78, "#e8ccc0", RED, "OD (right)", "diagnosis $y_{OD}$")
    eye(3.6, 3.1, 0.78, "#e8ccc0", RED, "OS (left)", "diagnosis $y_{OS}$")

    ax.annotate("", xy=(3.6, 4.35), xytext=(1.6, 4.35),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.4))
    ax.text(2.6, 4.5, "one patient", ha="center", fontsize=9,
            fontweight="bold", color=INK)

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.55, 1.05), 4.1, 3.75, boxstyle="round,pad=0.06,rounding_size=0.14",
        facecolor="none", edgecolor=GRID, linewidth=1.4, zorder=1))

    ax.text(2.6, 1.35, "shared: age · sex · systemic health\n"
                       "camera · operator · clinic · day",
            ha="center", va="center", fontsize=8.2, color=INK_2)

    ax.annotate("", xy=(5.75, 3.1), xytext=(4.85, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.6))

    ax.add_patch(mpatches.FancyBboxPatch(
        (5.95, 2.95), 3.6, 1.5, boxstyle="round,pad=0.06,rounding_size=0.14",
        facecolor="#eaf0fb", edgecolor=BLUE, linewidth=1.6, zorder=2))
    ax.text(7.75, 3.95, r"$z_{\rm pat}$  patient factor", ha="center",
            fontsize=9.5, fontweight="bold", color=INK)
    ax.text(7.75, 3.35, "predictable from the fellow eye\n(pulled together)",
            ha="center", va="center", fontsize=8.2, color=INK_2)

    ax.add_patch(mpatches.FancyBboxPatch(
        (5.95, 1.05), 3.6, 1.5, boxstyle="round,pad=0.06,rounding_size=0.14",
        facecolor="#e6f3f0", edgecolor=TEAL, linewidth=1.6, zorder=2))
    ax.text(7.75, 2.05, r"$z_{\rm path}$  pathology factor", ha="center",
            fontsize=9.5, fontweight="bold", color=INK)
    ax.text(7.75, 1.45, "identity not recoverable\ngiven the diagnosis",
            ha="center", va="center", fontsize=8.2, color=INK_2)

    ax.text(5.0, 0.35,
            "Confounders held fixed, pathology free to vary: the fellow eye is a matched control.",
            ha="center", fontsize=8.6, style="italic", color=INK_2)
    return save(fig, "f1_concept")


def f2_leakage():
    df = _read("e1_leakage_audit.csv")
    if df is None:
        return
    df = df.sort_values("R@1", ascending=False)
    short = {"dinov2_vitb14": "DINOv2\nViT-B/14", "dinov2_vits14": "DINOv2\nViT-S/14",
             "resnet50_in1k": "ResNet-50\n(IN-1k)"}
    labels = [f"{short.get(b, b)}\n{v}" for b, v in zip(df.backbone, df.variant)]
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    ax1.bar(x, df["R@1"], width=0.58, color=BLUE, zorder=3)
    ax1.axhline(df["chance_R@1"].iloc[0], color=CHANCE, ls="--", lw=1.4, zorder=4)
    ax1.text(-0.45, df["chance_R@1"].iloc[0] + 0.014,
             f"chance = {df['chance_R@1'].iloc[0]:.3f}", ha="left",
             fontsize=8, color=INK_2)
    for xi, v in zip(x, df["R@1"]):
        annotate(ax1, xi, v, f"{v:.3f}", dy=4, size=8.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.2)
    ax1.set_ylabel("fellow-eye retrieval  R@1")
    ax1.set_ylim(0, max(df["R@1"]) * 1.28)
    ax1.set_title("Patient identity survives preprocessing", loc="left",
                  fontweight="bold")

    ax2.bar(x, df["lift_R@1"], width=0.58, color=RED, zorder=3)
    ax2.axhline(1.0, color=CHANCE, ls="--", lw=1.4, zorder=4)
    for xi, v in zip(x, df["lift_R@1"]):
        annotate(ax2, xi, v, f"{v:.0f}x", dy=4, size=8.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.2)
    ax2.set_ylabel("lift over chance")
    ax2.set_ylim(0, max(df["lift_R@1"]) * 1.25)
    ax2.set_title("Stronger encoders leak more", loc="left", fontweight="bold")

    fig.suptitle("Given one eye, the model finds that patient's other eye",
                 x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "f2_leakage_audit")


def f3_split_protocol():
    small = _read("e2_split_protocol.csv")
    large = _read("e2_split_protocol_odir.csv")
    if small is None and large is None:
        return

    disc = {}
    dp = PATHS.tables / "t0_bilateral_discordance.csv"
    if dp.is_file():
        dd = pd.read_csv(dp)
        for name, lab in (("papila", "y_glaucoma"), ("odir", "y_abnormal"),
                          ("deepdrid", "y_referable")):
            row = dd[(dd.dataset == name) & (dd.label == lab)]
            if not row.empty:
                disc[name] = float(row.discordant_pct.iloc[0])

    cv_small, cv_large = _read("e7_cv_split.csv"), _read("e7_cv_split_odir.csv")
    small = cv_small if cv_small is not None else small
    large = cv_large if cv_large is not None else large
    ddr = _read("e7_cv_split_deepdrid.csv")

    panels = [(d, n, t) for d, n, t in
              [(small, "papila", "PAPILA  ·  glaucoma"),
               (large, "odir", "ODIR-5K  ·  normal vs abnormal"),
               (ddr, "deepdrid", "DeepDRiD  ·  referable DR")] if d is not None]
    panels.sort(key=lambda p: disc.get(p[1], 0.0))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 4.3),
                             squeeze=False, sharey=True)
    colors = {"patient-level": TEAL, "image-level": RED}

    for ax, (df, key, title) in zip(axes[0], panels):
        prim = df[(df.backbone == "dinov2_vitb14") & (df.variant == "canon")]
        df = prim if not prim.empty else df
        g = df.groupby("protocol").auc.agg(["mean", "std"])
        gap_txt = None
        if "repeat" in df.columns:
            piv = df.pivot_table(index="repeat", columns="protocol", values="auc")
            if {"image-level", "patient-level"}.issubset(piv.columns):
                dd = piv["image-level"] - piv["patient-level"]
                gap_txt = (f"{dd.mean():+.3f}",
                           f"{int((dd > 0).sum())}/{len(dd)} repeats")
        protos = ["patient-level", "image-level"]
        x = np.arange(len(protos))
        means = [g.loc[p, "mean"] for p in protos]
        sds = [g.loc[p, "std"] for p in protos]
        ax.bar(x, means, 0.5, yerr=sds, capsize=4,
               color=[colors[p] for p in protos], zorder=3,
               error_kw=dict(ecolor=INK_2, lw=1.1))
        for xi, m, s in zip(x, means, sds):
            annotate(ax, xi, m + (s or 0), f"{m:.3f}", dy=5, size=9)

        gap = means[1] - means[0]
        label_gap, sub_gap = (gap_txt if gap_txt else (f"{gap:+.3f}", ""))
        ax.annotate(label_gap, (0.5, max(means) + 0.055), ha="center",
                    fontsize=13, fontweight="bold", color=RED)
        if sub_gap:
            ax.annotate(sub_gap, (0.5, max(means) + 0.030), ha="center",
                        fontsize=8, color=INK_2)
        sub = (f"eyes disagree {disc[key]:.1f}% of the time"
               if key in disc else "")
        ax.set_title(f"{title}\n{sub}", loc="left", fontweight="bold", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["patient-\ngrouped", "image-\nlevel"], fontsize=8.5)
        ax.set_ylabel("AUC" if ax is axes[0][0] else "")
        ax.set_ylim(0.5, 1.0)

    fig.suptitle("The shortcut is worth most where a patient's two eyes agree",
                 x=0.02, y=0.985, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    src = ("repeated 5-fold cross-validation, 10 repeats" if cv_small is not None
           else "10 random partitions")
    fig.text(0.02, 0.01,
             f"mean $\\pm$ s.d. over {src}; identical features and classifier, "
             "only the grouping changes. Gap is paired within repeat.",
             fontsize=8, color=INK_2)
    return save(fig, "f3_split_protocol")


def f3b_split_by_variant():
    df = _read("e2_split_protocol.csv")
    if df is None:
        return
    variants = sorted(df.variant.unique())
    fig, ax = plt.subplots(figsize=(6.6, 3.9))

    width = 0.34
    x = np.arange(len(variants))
    colors = {"patient-level": TEAL, "image-level": RED}
    for i, proto in enumerate(["patient-level", "image-level"]):
        sub = df[df.protocol == proto]
        means = [sub[sub.variant == v].auc.mean() for v in variants]
        stds = [sub[sub.variant == v].auc.std() for v in variants]
        pos = x + (i - 0.5) * (width + 0.02)
        ax.bar(pos, means, width, yerr=stds, capsize=3, color=colors[proto],
               label=proto, zorder=3,
               error_kw=dict(ecolor=INK_2, lw=1.1, alpha=0.9))
        for p, m in zip(pos, means):
            annotate(ax, p, m, f"{m:.3f}", dy=3, size=8)

    for xi, v in zip(x, variants):
        pl = df[(df.protocol == "patient-level") & (df.variant == v)].auc.mean()
        il = df[(df.protocol == "image-level") & (df.variant == v)].auc.mean()
        ax.annotate(f"+{il-pl:.3f}", (xi, max(pl, il) + 0.055), ha="center",
                    fontsize=9, fontweight="bold", color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_xlabel("preprocessing variant")
    ax.set_ylabel("glaucoma AUC")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0.0, 1.0))
    ax.text(0.0, -0.24, "mean $\\pm$ s.d. over 10 random splits; "
                        "identical features and classifier throughout",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    fig.suptitle("Splitting by image instead of by patient inflates AUC (PAPILA)",
                 x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "f3b_split_by_variant")


def f4_mitigation():
    df = _read("e3_mitigation.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    base = df[df.method.str.startswith("none")]
    fams = [f for f in df.family.dropna().unique()] if "family" in df else []
    palette = {"CPMP-cond": TEAL, "CPMP-uncond": GOLD}

    for fam in fams:
        sub = df[df.family == fam].sort_values("rank")
        ax.plot(sub["R@1"], sub["auc"], "-o", color=palette.get(fam, PURPLE),
                label=fam, zorder=3, markeredgecolor="white", markeredgewidth=0.8)
        for _, r in sub.iterrows():
            if r["rank"] in (4, 32, 128):
                annotate(ax, r["R@1"], r["auc"], f"r={int(r['rank'])}",
                         dy=7, size=7.5, color=INK_2)

    if not base.empty:
        b = base.iloc[0]
        ax.plot(b["R@1"], b["auc"], "*", ms=17, color=BLUE, zorder=5,
                markeredgecolor="white", markeredgewidth=0.9,
                label="no mitigation")
        annotate(ax, b["R@1"], b["auc"], "frozen features", dy=10,
                 size=8.2, weight="bold")

    chance = 1.0 / 121
    ax.axvline(chance, color=CHANCE, ls="--", lw=1.4, zorder=2)
    ax.text(chance + 0.004, ax.get_ylim()[0] + 0.02,
            "chance R@1", rotation=90, fontsize=8, color=INK_2, va="bottom")

    ax.set_xlabel("fellow-eye retrieval R@1   (lower = less identity leaked)")
    ax.set_ylabel("glaucoma AUC   (higher = better)")
    ax.set_title("Removing identity costs accuracy faster than it removes identity",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left")
    ax.text(0.0, -0.17,
            "Ideal is top-left. Every method moves down before it moves left.",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    fig.tight_layout()
    return save(fig, "f4_mitigation_tradeoff")


def f5_dimensionality():
    df = _read("e4_dimensionality.csv")
    if df is None:
        return
    rem = df[df.sweep == "remove_top_k_nuisance"].sort_values("k")
    keep = df[df.sweep == "keep_top_k_pca"].sort_values("k")
    if rem.empty:
        return
    max_rank = int(rem["max_estimable_rank"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.7))

    ax1.plot(rem.k, rem["R@1"], "-o", color=TEAL, label="R@1 after removal",
             markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax1.axhline(rem["chance"].iloc[0], color=CHANCE, ls="--", lw=1.4)
    ax1.text(rem.k.max(), rem["chance"].iloc[0] + 0.012, "chance",
             ha="right", fontsize=8, color=INK_2)
    ax1.set_xlabel("nuisance directions removed  $k$")
    ax1.set_ylabel("fellow-eye retrieval R@1")
    ax1.set_ylim(0, max(rem["R@1"]) * 1.2)
    ax1.set_title("Removal barely dents identity", loc="left", fontweight="bold")
    ax1.legend(loc="lower left")

    ax2.plot(keep.k, keep["R@1"], "-o", color=BLUE, label="R@1",
             markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax2.plot(keep.k, keep["auc"], "-s", color=RED, label="glaucoma AUC",
             markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("PCA components kept  $k$")
    ax2.set_ylabel("score")
    ax2.set_title("Identity appears in the very first components",
                  loc="left", fontweight="bold")
    ax2.legend(loc="lower right")

    fig.suptitle(
        f"Patient identity is high-dimensional; with {max_rank} training patients "
        f"at most {max_rank} directions are even estimable",
        x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "f5_dimensionality")


def f6_low_label():
    df = _read("e5_low_label.csv")
    if df is None:
        return
    g = df.groupby("label_frac").agg(
        auc=("auc", "mean"), sd=("auc", "std"),
        npat=("n_labelled_patients", "first")).reset_index()

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(g.label_frac * 100, g.auc, "-o", color=BLUE, zorder=4,
            markeredgecolor="white", markeredgewidth=0.9,
            label="linear probe on frozen features")
    ax.fill_between(g.label_frac * 100, g.auc - g.sd.fillna(0),
                    g.auc + g.sd.fillna(0), color=BLUE, alpha=0.14, zorder=2)
    for _, r in g.iterrows():
        annotate(ax, r.label_frac * 100, r.auc,
                 f"{r.auc:.3f}\nn={int(r.npat)}", dy=7, size=7.6, color=INK_2)
    ax.axhline(0.5, color=CHANCE, ls="--", lw=1.3)
    ax.text(100, 0.512, "chance", ha="right", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 100])
    ax.set_xticklabels(["5%", "10%", "25%", "50%", "100%"])
    ax.set_xlabel("labelled patients (of 147 training patients)")
    ax.set_ylabel("glaucoma AUC")
    ax.set_ylim(0.45, 0.95)
    ax.legend(loc="lower right")
    ax.set_title("Label efficiency on PAPILA", loc="left", fontweight="bold")
    ax.text(0.0, -0.19, "mean $\\pm$ s.d. over 10 seeds; labels dropped by patient, never by eye",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    fig.tight_layout()
    return save(fig, "f6_label_efficiency")


def f7_similarity():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dataio.embeddings import EmbeddingStore

    try:
        s = EmbeddingStore.load("papila", "dinov2_vitb14", "canon").complete_pairs_only()
    except FileNotFoundError:
        print("    [skip] embeddings not found")
        return
    te = s.subset(s.df["split"].to_numpy() == "test")
    order = np.argsort(te.df["patient_id"].to_numpy(), kind="stable")
    z = te.feat[order]
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    sim = z @ z.T
    n = min(28, len(sim))
    sim = sim[:n, :n].copy()
    np.fill_diagonal(sim, np.nan)
    sim = sim - np.nanmean(sim, axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(GRID)
    im = ax1.imshow(sim, cmap=cmap, vmin=0, vmax=np.nanpercentile(sim, 99))
    for i in range(0, n, 2):
        ax1.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 2, 2, fill=False,
                                    edgecolor=RED, linewidth=1.3, zorder=5))
    ax1.set_title("Similarity above each eye's own mean", loc="left",
                  fontweight="bold")
    ax1.set_xlabel("eye index (ordered by patient)")
    ax1.set_ylabel("eye index")
    ax1.grid(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("cosine sim. $-$ row mean", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax1.text(0.5, -0.32,
             "Red squares mark the two eyes of one patient.",
             transform=ax1.transAxes, ha="center", fontsize=8, color=INK_2)

    full = te.feat / (np.linalg.norm(te.feat, axis=1, keepdims=True) + 1e-9)
    S = full @ full.T
    pair = te.pair_index()
    same = (pair[:, None] == pair[None, :]) & ~np.eye(len(pair), dtype=bool)
    diff = (pair[:, None] != pair[None, :])
    ax2.hist(S[diff], bins=60, color=MUTED, alpha=0.75, density=True,
             label="different patients", zorder=2)
    ax2.hist(S[same], bins=25, color=RED, alpha=0.85, density=True,
             label="same patient (fellow eye)", zorder=3)
    ax2.set_xlabel("cosine similarity")
    ax2.set_ylabel("density")
    ax2.legend(loc="upper left")
    ax2.set_title("Fellow eyes are systematically closer", loc="left",
                  fontweight="bold")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, "f7_similarity_structure")


def f8_pixel_baseline():
    small = _read("e6_pixel_baseline.csv")
    large = _read("e6_pixel_baseline_odir.csv")
    if small is None and large is None:
        return

    order = ["colour histogram", "raw pixels 32x32 gray", "raw pixels 64x64",
             "raw pixels 32x32", "raw pixels 16x16"]
    pretty = {"colour histogram": "colour histogram\n(no spatial info)",
              "raw pixels 32x32 gray": "pixels 32$\\times$32\ngrayscale",
              "raw pixels 64x64": "pixels\n64$\\times$64",
              "raw pixels 32x32": "pixels\n32$\\times$32",
              "raw pixels 16x16": "pixels\n16$\\times$16"}

    ddr = _read("e6_pixel_baseline_deepdrid.csv")

    def ncand(d):
        return int(d["n_test"].iloc[0]) - 1

    panels = [(d, f"{lbl}  (${ncand(d):,}$ candidates)".replace(",", "{,}"), ncand(d))
              for d, lbl in [(small, "PAPILA"), (ddr, "DeepDRiD"),
                             (large, "ODIR-5K")] if d is not None]
    panels.sort(key=lambda p: p[2])

    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 4.2),
                             squeeze=False)
    for ax, (df, title, ncand) in zip(axes[0], panels):
        learned = df[df.representation.str.contains("learned")]
        base = df[~df.representation.str.contains("learned")]
        base = base.set_index("representation").reindex(order).dropna(subset=["R@1"])

        y = np.arange(len(base))
        ax.barh(y, base["R@1"], height=0.6, color=MUTED, zorder=3,
                label="pixel baselines")
        for yi, v in zip(y, base["R@1"]):
            ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels([pretty.get(i, i) for i in base.index], fontsize=8)

        ax.set_ylim(len(base) - 0.4, -1.15)
        if not learned.empty:
            lv = float(learned["R@1"].iloc[0])
            ax.axvline(lv, color=BLUE, lw=2.2, zorder=5,
                       label="DINOv2 ViT-B/14 (learned)")
            ax.text(lv, -0.72, f"  {lv:.3f}", color=BLUE, fontsize=9.5,
                    fontweight="bold", va="center", ha="left")
            best = float(base["R@1"].max())
            ax.set_title(f"{title}\nlearned / best pixel = {lv/best:.2f}$\\times$",
                         loc="left", fontweight="bold", fontsize=9.5)
        else:
            ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)

        ax.set_xlabel("fellow-eye retrieval R@1")
        ax.set_xlim(0, max(df["R@1"]) * 1.22)
        ax.grid(axis="y", visible=False)

    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncols=2,
               bbox_to_anchor=(0.02, 0.935), fontsize=8.5)
    fig.suptitle("Does the encoder beat raw pixels? Only with a large candidate pool",
                 x=0.02, y=0.985, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.87))
    smallest = panels[0][2]
    fig.text(0.02, 0.008,
             f"The margin grows monotonically with the candidate pool. With {smallest} "
             "candidates every representation solves the task and the baselines are "
             "indistinguishable — the regime the published dispute was argued in.",
             fontsize=8, color=INK_2)
    return save(fig, "f8_pixel_baseline")


def f9_mitigation_both():
    small = _read("e3_mitigation.csv")
    large = _read("e3_mitigation_odir.csv")
    panels = [(d, t) for d, t in [(small, "PAPILA  (147 training pairs)"),
                                  (large, "ODIR-5K  (1,821 training pairs)")]
              if d is not None]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.2),
                             squeeze=False)
    palette = {"CPMP-cond": TEAL, "CPMP-uncond": GOLD}

    for ax, (df, title) in zip(axes[0], panels):
        base = df[df.method.str.startswith("none")]
        for fam in [f for f in df.get("family", pd.Series(dtype=str)).dropna().unique()]:
            sub = df[df.family == fam].sort_values("rank")
            ax.plot(sub["R@1"], sub["auc"], "-o", color=palette.get(fam, PURPLE),
                    label=fam, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            for _, r in sub.iterrows():
                if r["rank"] in (4, 32, 128):
                    annotate(ax, r["R@1"], r["auc"], f"r={int(r['rank'])}", dy=7,
                             size=7.2, color=INK_2)
        if not base.empty:
            b = base.iloc[0]
            ax.plot(b["R@1"], b["auc"], "*", ms=17, color=BLUE, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.9, label="no mitigation")
            ax.annotate("", xy=(df["R@1"].max(), df["auc"].min()),
                        xytext=(b["R@1"], b["auc"]),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2,
                                        ls=":", shrinkA=8, shrinkB=8))
        ax.set_xlabel("fellow-eye retrieval R@1  (lower is better)")
        ax.set_ylabel("task AUC  (higher is better)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Ideal is top-left. Both cohorts move the wrong way.",
                 x=0.02, y=0.985, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    return save(fig, "f9_mitigation_both")


def f10_asymmetry():
    df = _read("e8_asymmetry_odir.csv")
    if df is None:
        return
    order = ["per-eye (matched data)", "per-eye (all training images)",
             "difference  z(A) - z(B)", "concatenation [z(A); z(B)]"]
    pretty = {"per-eye (matched data)": "per-eye\nmatched",
              "per-eye (all training images)": "per-eye\nfull data",
              "difference  z(A) - z(B)": "bilateral\ndifference",
              "concatenation [z(A); z(B)]": "bilateral\nconcat"}
    g = (df.groupby("method").agg(auc=("auc", "mean"), sd=("auc", "std"))
           .reindex(order).dropna())
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cols = [MUTED if "per-eye" in m else TEAL for m in g.index]
    x = np.arange(len(g))
    ax.bar(x, g.auc, 0.64, yerr=g.sd, capsize=4, color=cols, zorder=3,
           error_kw=dict(ecolor=INK_2, lw=1.1))
    for xi, v, s in zip(x, g.auc, g.sd):
        annotate(ax, xi, v + s, f"{v:.3f}", dy=4, size=8.5)

    ax.axhline(0.5, color=CHANCE, ls="--", lw=1.3)
    ax.text(-0.45, 0.508, "chance", ha="left", fontsize=8, color=INK_2)

    best_base = g.loc[[m for m in g.index if "per-eye" in m], "auc"].max()
    best_bi = g.loc[[m for m in g.index if "per-eye" not in m], "auc"].max()
    top = float((g.auc + g.sd).max())
    yb = top + 0.055
    leg = top + 0.022
    ax.plot([1, 1, 3, 3], [leg, yb, yb, leg],
            color=RED, lw=1.4, zorder=6, clip_on=False)
    ax.text(2.0, yb + 0.008, f"+{best_bi-best_base:.3f} AUC", ha="center",
            va="bottom", fontsize=10.5, fontweight="bold", color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(m, m) for m in g.index], fontsize=8.5)
    ax.set_ylabel("AUC  ·  which eye is the abnormal one")
    ax.set_ylim(0.45, yb + 0.055)
    ax.set_title("Comparing the eyes beats scoring them alone",
                 loc="left", fontweight="bold")
    ax.text(0.0, -0.255,
            "675 ODIR pairs whose eyes carry different labels; "
            f"{int(df.n_test.iloc[0]//2)} held-out pairs, 20 bootstrap resamples.\n"
            "“matched” trains the per-eye model on exactly the paired eyes; "
            "bilateral models use $z(A)-z(B)$ or $[z(A);z(B)]$.\n"
            "Both bilateral models beat both baselines in 20/20 resamples.",
            transform=ax.transAxes, fontsize=7.8, color=INK_2, va="top")
    fig.tight_layout()
    return save(fig, "f10_asymmetry")


FIGURES = {
    "f1": f1_concept, "f2": f2_leakage, "f3": f3_split_protocol,
    "f3b": f3b_split_by_variant, "f4": f4_mitigation, "f5": f5_dimensionality,
    "f6": f6_low_label, "f7": f7_similarity, "f8": f8_pixel_baseline,
    "f9": f9_mitigation_both, "f10": f10_asymmetry,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(FIGURES))
    args = ap.parse_args()

    use_paper_style()
    print("Figures")
    for key in args.only:
        fn = FIGURES.get(key)
        if fn is None:
            print(f"  [?] unknown figure {key}")
            continue
        print(f"  [{key}] {fn.__name__}")
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
