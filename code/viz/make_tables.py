
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

R = PATHS.results
T = PATHS.tables


def _read(name: str) -> pd.DataFrame | None:
    p = R / name
    if not p.is_file():
        print(f"    [skip] {name} not found")
        return None
    return pd.read_csv(p)


def emit(df: pd.DataFrame, stem: str, caption: str, label: str,
         align: str | None = None, note: str = "") -> None:
    T.mkdir(parents=True, exist_ok=True)

    md = [f"**{caption}**", ""]
    md.append("| " + " | ".join(df.columns) + " |")
    md.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(v) for v in r.to_numpy()) + " |")
    if note:
        md += ["", f"_{note}_"]
    (T / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    al = align or ("l" + "r" * (len(df.columns) - 1))
    tex = [
        r"\begin{table}[t]", r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{tab:{label}}}",
        f"  \\begin{{tabular}}{{{al}}}", r"    \toprule",
        "    " + " & ".join(c.replace("_", r"\_") for c in df.columns) + r" \\",
        r"    \midrule",
    ]
    for _, r in df.iterrows():
        tex.append("    " + " & ".join(str(v).replace("_", r"\_")
                                       for v in r.to_numpy()) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        tex.append(r"  \\[2pt] {\footnotesize " + note + "}")
    tex.append(r"\end{table}")
    (T / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"    -> {stem}.md  + .tex")


def ci(point, lo, hi, nd=3) -> str:
    if not np.isfinite(point):
        return "--"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{point:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    return f"{point:.{nd}f}"


def t1_datasets() -> None:
    rows = []
    for name in ("papila", "odir"):
        man = PATHS.manifests / f"{name}.csv"
        if not man.is_file():
            continue
        df = pd.read_csv(man)
        ycols = [c for c in df.columns
                 if c.startswith("y_") and df[c].dropna().isin([0, 1]).all()]
        disc = _read("../tables/t0_bilateral_discordance.csv")
        d = ""
        p = T / "t0_bilateral_discordance.csv"
        if p.is_file():
            dd = pd.read_csv(p)
            dd = dd[dd.dataset == name]
            if not dd.empty:
                d = f"{dd.discordant_pct.min():.1f}--{dd.discordant_pct.max():.1f}%"
        rows.append({
            "Dataset": name.upper(),
            "Images": len(df),
            "Patients": df["patient_id"].nunique(),
            "Eyes/patient": round(len(df) / max(df["patient_id"].nunique(), 1), 2),
            "Labels": len(ycols),
            "Train/Val/Test (pat.)": "/".join(
                str(df[df.split == s]["patient_id"].nunique())
                for s in ("train", "val", "test")),
            "Bilateral discordance": d or "--",
        })
    if not rows:
        print("    [skip] no manifests")
        return
    emit(pd.DataFrame(rows), "t1_datasets",
         "Bilateral cohorts used in this work.", "datasets",
         note="Discordance is the fraction of patients whose two eyes carry "
              "different labels; it bounds how much genuine within-patient "
              "pathology variation a bilateral method can exploit.")


def t2_leakage() -> None:
    df = _read("e1_leakage_audit.csv")
    if df is None:
        return
    out = pd.DataFrame({
        "Backbone": df.backbone.str.replace("_", "-"),
        "Preproc.": df.variant,
        "dim": df.dim,
        "R@1": df["R@1"].map("{:.3f}".format),
        "chance": df["chance_R@1"].map("{:.4f}".format),
        "lift": df["lift_R@1"].map("{:.0f}$\\times$".format),
        "MRR": df["MRR"].map("{:.3f}".format),
        "PID acc.": df["pid_acc"].map("{:.3f}".format),
        "Age $R^2$": df["probe_a_age"].map("{:.3f}".format),
        "Sex AUC": df["probe_a_gender"].map("{:.3f}".format),
        "Glaucoma AUC": [ci(a, l, h) for a, l, h in
                         zip(df.task_auc, df.task_lo, df.task_hi)],
    })
    emit(out, "t2_leakage_audit",
         "Patient identity recoverable from frozen features on held-out patients.",
         "leakage",
         note="R@1 is fellow-eye retrieval among all test eyes. Circle-cropping "
              "and laterality canonicalisation remove the black frame and the "
              "mirror cue, yet identity is unaffected -- the signal is the "
              "retinal vasculature itself.")


def t3_headline() -> None:
    disc = {}
    dp = T / "t0_bilateral_discordance.csv"
    if dp.is_file():
        dd = pd.read_csv(dp)
        for name, lab in (("papila", "y_glaucoma"), ("odir", "y_abnormal"),
                          ("deepdrid", "y_referable")):
            row = dd[(dd.dataset == name) & (dd.label == lab)]
            if not row.empty:
                disc[name] = float(row.discordant_pct.iloc[0])

    rows = []
    for fname, key, label, task in (
            ("e7_cv_split.csv", "papila", "PAPILA", "glaucoma"),
            ("e7_cv_split_odir.csv", "odir", "ODIR-5K", "normal vs abnormal"),
            ("e7_cv_split_deepdrid.csv", "deepdrid", "DeepDRiD", "referable DR")):
        df = _read(fname)
        if df is None:
            continue
        g = df.groupby("protocol").auc.agg(["mean", "std"])
        if not {"patient-level", "image-level"}.issubset(g.index):
            continue
        pl, il = g.loc["patient-level"], g.loc["image-level"]
        piv = df.pivot_table(index="repeat", columns="protocol", values="auc")
        d = piv["image-level"] - piv["patient-level"]
        rows.append({
            "Cohort": label,
            "Patients": {"papila": 244, "odir": 3034, "deepdrid": 400}.get(key, "--"),
            "Task": task,
            "Eyes disagree": f"{disc[key]:.1f}\\%" if key in disc else "--",
            "Patient-grouped AUC": f"{pl['mean']:.3f} $\\pm$ {pl['std']:.3f}",
            "Image-level AUC": f"{il['mean']:.3f} $\\pm$ {il['std']:.3f}",
            "Inflation": f"\\textbf{{{d.mean():+.3f}}} $\\pm$ {d.std():.3f}",
            "Repeats $>0$": f"{int((d > 0).sum())}/{len(d)}",
        })
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["Eyes disagree"].rstrip("\\%"))
                  if r["Eyes disagree"] != "--" else 0)
    emit(pd.DataFrame(rows), "t3_headline",
         "The fellow-eye shortcut is worth most where a patient's two eyes agree.",
         "headline",
         note="Repeated 5-fold cross-validation, 10 repeats; identical features "
              "(DINOv2 ViT-B/14, canonicalised) and classifier throughout, only "
              "the grouping changes. Inflation is paired within repeat. Rows are "
              "ordered by bilateral discordance and the effect falls monotonically "
              "with it, reaching zero on the most discordant cohort (5/10 repeats "
              "positive is a coin flip). Cohorts whose eyes almost always share a "
              "diagnosis are the ones an image-level split flatters most, and a "
              "group can estimate its own exposure from its own discordance rate.")


def t6_pixel_baseline() -> None:
    frames = []
    for fname, label in (("e6_pixel_baseline.csv", "PAPILA"),
                         ("e6_pixel_baseline_deepdrid.csv", "DeepDRiD"),
                         ("e6_pixel_baseline_odir.csv", "ODIR-5K")):
        df = _read(fname)
        if df is None:
            continue
        ncand = int(df["n_test"].iloc[0]) - 1
        frames.append(df.assign(cohort=label, ncand=ncand))
    if not frames:
        return
    allf = pd.concat(frames, ignore_index=True)

    order = ["dinov2_vitb14 (learned)", "raw pixels 16x16", "raw pixels 32x32",
             "raw pixels 64x64", "raw pixels 32x32 gray", "colour histogram"]
    pretty = {"dinov2_vitb14 (learned)": "DINOv2 ViT-B/14 (learned)",
              "raw pixels 16x16": "raw pixels $16\\times16$",
              "raw pixels 32x32": "raw pixels $32\\times32$",
              "raw pixels 64x64": "raw pixels $64\\times64$",
              "raw pixels 32x32 gray": "raw pixels $32\\times32$ grayscale",
              "colour histogram": "colour histogram (no spatial info)"}

    out = {"Representation": [pretty.get(o, o) for o in order]}
    for label in [f.cohort.iloc[0] for f in frames]:
        sub = allf[allf.cohort == label].set_index("representation")
        n = int(sub["ncand"].iloc[0])
        col = []
        for o in order:
            col.append(f"{sub.loc[o, 'R@1']:.3f}" if o in sub.index else "--")
        out[f"{label} R@1 ({n} cand.)"] = col
    tab = pd.DataFrame(out)

    ratio = {"Representation": "\\textit{learned / best pixel}"}
    for label in [f.cohort.iloc[0] for f in frames]:
        sub = allf[allf.cohort == label].set_index("representation")
        learned = sub.loc["dinov2_vitb14 (learned)", "R@1"]
        best = sub.drop(index="dinov2_vitb14 (learned)")["R@1"].max()
        n = int(sub["ncand"].iloc[0])
        ratio[f"{label} R@1 ({n} cand.)"] = f"\\textbf{{{learned/best:.2f}$\\times$}}"
    tab = pd.concat([tab, pd.DataFrame([ratio])], ignore_index=True)

    emit(tab, "t6_pixel_baseline",
         "The near-duplicate control: does a learned encoder beat raw pixels?",
         "pixelbaseline",
         note="Wagner \\& Keane (npj Digital Medicine 2026) argued that reported "
              "fundus re-identification is explained by near-identical images, "
              "since raw pixels match RETFound. Their control is reproduced here "
              "in the bilateral setting. With 122 candidates every representation "
              "solves the task and the baselines are indistinguishable -- the "
              "regime the dispute was argued in. With 1,516 the learned encoder "
              "pulls clear, so the objection is testable but requires a candidate "
              "pool large enough for pixels to fail.")


def t7_interventions() -> None:
    rows = []

    def grab(fname, variant):
        df = _read(fname)
        if df is None:
            return None
        s = df[(df.backbone == "dinov2_vitb14") & (df.variant == variant)]
        return None if s.empty else s.iloc[0]

    base_p = grab("e1_leakage_audit.csv", "canon")
    base_o = grab("e1_leakage_audit_odir.csv", "canon")
    bg_p = grab("e1_leakage_audit.csv", "canonbg")
    bg_o = grab("e1_leakage_audit_odir.csv", "canonbg")

    def fmt(r, key):
        return f"{r[key]:.3f}" if r is not None else "--"

    def delta(r, b, key, invert=False):
        if r is None or b is None:
            return ""
        d = r[key] - b[key]
        good = (d < 0) if invert else (d > 0)
        arrow = "$\\downarrow$" if d < 0 else "$\\uparrow$"
        s = f"{d:+.3f}\\,{arrow}"
        return f"\\textbf{{{s}}}" if good else s

    rows.append({
        "Intervention": "none (baseline)",
        "PAPILA R@1": fmt(base_p, "R@1"), "PAPILA AUC": fmt(base_p, "task_auc"),
        "ODIR R@1": fmt(base_o, "R@1"), "ODIR AUC": fmt(base_o, "task_auc"),
        "Verdict": "--",
    })
    rows.append({
        "Intervention": "Ben Graham colour norm.",
        "PAPILA R@1": f"{fmt(bg_p,'R@1')} ({delta(bg_p, base_p, 'R@1', invert=True)})",
        "PAPILA AUC": f"{fmt(bg_p,'task_auc')} ({delta(bg_p, base_p, 'task_auc')})",
        "ODIR R@1": f"{fmt(bg_o,'R@1')} ({delta(bg_o, base_o, 'R@1', invert=True)})",
        "ODIR AUC": f"{fmt(bg_o,'task_auc')} ({delta(bg_o, base_o, 'task_auc')})",
        "Verdict": "\\textbf{both axes improve}",
    })

    adv = _read("adv_papila.csv")
    if adv is not None:
        zp = adv[adv.representation == "z_path"]
        rows.append({
            "Intervention": "adversarial adapter ($z_{\\rm path}$)",
            "PAPILA R@1": f"{zp['ret_R@1'].mean():.3f}",
            "PAPILA AUC": f"{zp['task_auc'].mean():.3f}",
            "ODIR R@1": "--", "ODIR AUC": "--",
            "Verdict": "hides, does not remove",
        })

    for fname, cohort in (("e3_mitigation.csv", "p"), ("e3_mitigation_odir.csv", "o")):
        df = _read(fname)
        if df is None:
            continue
        s = df[df.method == "CPMP-cond r=32"]
        if s.empty:
            continue
        globals().setdefault("_cpmp", {})[cohort] = s.iloc[0]
    cp = globals().get("_cpmp", {})
    if cp:
        rows.append({
            "Intervention": "subspace projection ($r{=}32$)",
            "PAPILA R@1": f"{cp['p']['R@1']:.3f}" if "p" in cp else "--",
            "PAPILA AUC": f"{cp['p']['auc']:.3f}" if "p" in cp else "--",
            "ODIR R@1": f"{cp['o']['R@1']:.3f}" if "o" in cp else "--",
            "ODIR AUC": f"{cp['o']['auc']:.3f}" if "o" in cp else "--",
            "Verdict": "backfires on ODIR",
        })

    emit(pd.DataFrame(rows), "t7_interventions",
         "Every intervention tried, on both axes and both cohorts.", "interventions",
         note="R@1 is fellow-eye retrieval (lower is better); AUC is the "
              "diagnostic task (higher is better). Only colour normalisation "
              "moves both the right way, and it is preprocessing, not "
              "representation learning. It is a mitigation, not a fix: "
              "retrieval stays at 45$\\times$ chance on PAPILA and "
              "406$\\times$ on ODIR.")


def t8_backbone() -> None:
    COHORTS = [("PAPILA", "", 2.9), ("ODIR-5K", "_odir", 22.2),
               ("DeepDRiD", "_deepdrid", 29.2)]
    BACKS = [("dinov2_vitb14", "DINOv2 ViT-B/14", ""),
             ("retfound_cfp", "RETFound ViT-L/16", "_retfound_cfp")]

    rows = []
    for label, dsfx, dsc in COHORTS:
        for key, bname, bsfx in BACKS:
            e1 = _read(f"e1_leakage_audit{dsfx}.csv")
            r = a = None
            if e1 is not None:
                s = e1[(e1.backbone == key) & (e1.variant == "canon")]
                if not s.empty:
                    r, a = float(s.iloc[0]["R@1"]), float(s.iloc[0]["lift_R@1"])
                    auc = float(s.iloc[0]["task_auc"])
            cv = _read(f"e7_cv_split{dsfx}{bsfx}.csv")
            gap = sd = pos = n = None
            if cv is not None:
                piv = cv.pivot_table(index="repeat", columns="protocol", values="auc")
                if {"image-level", "patient-level"}.issubset(piv.columns):
                    d = piv["image-level"] - piv["patient-level"]
                    gap, sd, pos, n = d.mean(), d.std(), int((d > 0).sum()), len(d)
            if r is None and gap is None:
                continue
            rows.append({
                "Cohort": label if key == BACKS[0][0] else "",
                "Eyes disagree": f"{dsc:.1f}\\%" if key == BACKS[0][0] else "",
                "Encoder": bname,
                "R@1": f"{r:.3f}" if r is not None else "--",
                "lift": f"{a:.0f}$\\times$" if a is not None else "--",
                "Task AUC": f"{auc:.3f}" if r is not None else "--",
                "Inflation": (f"{gap:+.3f} $\\pm$ {sd:.3f}" if gap is not None else "--"),
                "Repeats $>0$": f"{pos}/{n}" if gap is not None else "--",
            })
    if not rows:
        return
    emit(pd.DataFrame(rows), "t8_backbone",
         "Identity leakage and split-protocol inflation are not the same quantity.",
         "backbone",
         note="RETFound, pretrained on 1.6M retinal images, retrieves the fellow "
              "eye far less often than a general-purpose DINOv2 at comparable "
              "diagnostic AUC -- domain pretraining does reduce measurable "
              "leakage. Yet it produces the \\emph{larger} inflation on every "
              "cohort. Reducing how much identity a representation carries does "
              "not reduce how much an evaluation protocol lets a classifier "
              "exploit it. The discordance ordering, however, replicates under "
              "both encoders.")


def t3_split() -> None:
    df = _read("e2_split_protocol.csv")
    if df is None:
        return
    g = (df.groupby(["backbone", "variant", "protocol"])
           .agg(auc=("auc", "mean"), sd=("auc", "std"),
                fellow=("frac_test_with_fellow_in_train", "mean"))
           .reset_index())
    rows = []
    for (b, v), sub in g.groupby(["backbone", "variant"]):
        pl = sub[sub.protocol == "patient-level"]
        il = sub[sub.protocol == "image-level"]
        if pl.empty or il.empty:
            continue
        rows.append({
            "Backbone": b.replace("_", "-"),
            "Preproc.": v,
            "Patient-level AUC": f"{pl.auc.iloc[0]:.3f} $\\pm$ {pl.sd.iloc[0]:.3f}",
            "Image-level AUC": f"{il.auc.iloc[0]:.3f} $\\pm$ {il.sd.iloc[0]:.3f}",
            "Inflation": f"+{il.auc.iloc[0]-pl.auc.iloc[0]:.3f}",
            "Test eyes w/ fellow in train": f"{il.fellow.iloc[0]*100:.0f}\\%",
        })
    if not rows:
        return
    emit(pd.DataFrame(rows), "t3_split_protocol",
         "Cost of an image-level split, measured on identical features.", "split",
         note="Mean $\\pm$ s.d. over 10 random partitions. Only the grouping "
              "changes; features, classifier and split sizes are held fixed.")


def t4_mitigation() -> None:
    df = _read("e3_mitigation.csv")
    if df is None:
        return
    keep = df[(df["rank"].isin([0, 4, 16, 32, 64, 128]))].copy()
    out = pd.DataFrame({
        "Method": keep.method,
        "R@1": keep["R@1"].map("{:.3f}".format),
        "lift": keep["lift_R@1"].map("{:.0f}$\\times$".format),
        "Age $R^2$": keep["probe_a_age"].map("{:.3f}".format),
        "Sex AUC": keep["probe_a_gender"].map("{:.3f}".format),
        "Glaucoma AUC": [ci(a, l, h) for a, l, h in zip(keep.auc, keep.lo, keep.hi)],
    })
    emit(out, "t4_mitigation",
         "Identity leakage against diagnostic accuracy for each defence.",
         "mitigation",
         note="No configuration reaches chance-level retrieval (0.008) at any "
              "usable accuracy. Conditioning on the diagnosis preserves AUC far "
              "better than the unconditional variant, confirming that most "
              "between-patient variance in this cohort is label-relevant.")


def t5_label_efficiency() -> None:
    df = _read("e5_low_label.csv")
    if df is None:
        return
    g = (df.groupby("label_frac")
           .agg(auc=("auc", "mean"), sd=("auc", "std"),
                npat=("n_labelled_patients", "first"),
                nimg=("n_labelled_images", "first"))
           .reset_index())
    out = pd.DataFrame({
        "Labelled fraction": (g.label_frac * 100).map("{:.0f}\\%".format),
        "Patients": g.npat,
        "Images": g.nimg,
        "Glaucoma AUC": [f"{a:.3f} $\\pm$ {s:.3f}" if np.isfinite(s) else f"{a:.3f}"
                         for a, s in zip(g.auc, g.sd)],
    })
    emit(out, "t5_label_efficiency",
         "Label efficiency of a linear probe on frozen features.", "labeleff",
         note="Mean $\\pm$ s.d. over 10 seeds. Labels are dropped by patient: "
              "dropping one eye of a pair would leak its label through the fellow.")


def main() -> int:
    print("Tables")
    for fn in (t1_datasets, t2_leakage, t3_headline, t3_split, t4_mitigation,
               t5_label_efficiency, t6_pixel_baseline, t7_interventions,
               t8_backbone):
        print(f"  [{fn.__name__}]")
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
