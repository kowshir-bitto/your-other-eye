
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

checks: list[tuple[str, float, float]] = []


def tolerance(quoted: float) -> float:
    s = f"{quoted!r}"
    if "e" in s or "E" in s:
        return 5e-4
    decimals = len(s.split(".")[1]) if "." in s else 0
    if decimals == 0:
        return 1.0
    return 0.5 * 10 ** (-decimals) + 1e-9


def read(name):
    p = PATHS.results / name
    return pd.read_csv(p) if p.is_file() else None


def check(label, quoted, actual):
    checks.append((label, quoted, actual))


def leak(dsfx, backbone, variant, col):
    df = read(f"e1_leakage_audit{dsfx}.csv")
    if df is None:
        return None
    s = df[(df.backbone == backbone) & (df.variant == variant)]
    return None if s.empty else float(s.iloc[0][col])


def infl(dsfx, bsfx=""):
    df = read(f"e7_cv_split{dsfx}{bsfx}.csv")
    if df is None:
        return None
    piv = df.pivot_table(index="repeat", columns="protocol", values="auc")
    if not {"image-level", "patient-level"}.issubset(piv.columns):
        return None
    d = piv["image-level"] - piv["patient-level"]
    return {"gap": d.mean(), "sd": d.std(), "pos": int((d > 0).sum()),
            "pat": piv["patient-level"].mean(), "img": piv["image-level"].mean()}


def px(dsfx):
    df = read(f"e6_pixel_baseline{dsfx}.csv")
    if df is None:
        return None
    s = df.set_index("representation")
    learned = float(s.loc["dinov2_vitb14 (learned)", "R@1"])
    best = float(s.drop(index="dinov2_vitb14 (learned)")["R@1"].max())
    return {"learned": learned, "best": best, "ratio": learned / best,
            "ncand": int(s["n_test"].iloc[0]) - 1}


check("ODIR DINOv2 R@1", 0.292, leak("_odir", "dinov2_vitb14", "canon", "R@1"))
check("ODIR DINOv2 lift", 442, leak("_odir", "dinov2_vitb14", "canon", "lift_R@1"))
check("PAPILA raw R@1", 0.361, leak("", "dinov2_vitb14", "raw", "R@1"))
check("PAPILA crop R@1", 0.393, leak("", "dinov2_vitb14", "crop", "R@1"))
check("PAPILA canon R@1", 0.410, leak("", "dinov2_vitb14", "canon", "R@1"))
check("PAPILA age R2", 0.560, leak("", "dinov2_vitb14", "canon", "probe_a_age"))
check("PAPILA sex AUC", 0.674, leak("", "dinov2_vitb14", "canon", "probe_a_gender"))

for dsfx, name, q_learn, q_best, q_ratio, q_n in (
        ("", "PAPILA", 0.418, 0.352, 1.19, 121),
        ("_deepdrid", "DeepDRiD", 0.230, 0.150, 1.53, 199),
        ("_odir", "ODIR", 0.292, 0.152, 1.92, 1515)):
    p = px(dsfx)
    if p:
        check(f"{name} px learned", q_learn, p["learned"])
        check(f"{name} px best", q_best, p["best"])
        check(f"{name} px ratio", q_ratio, p["ratio"])
        check(f"{name} px ncand", q_n, p["ncand"])

e9 = read("e9_same_eye_control_deepdrid_fields.csv")
if e9 is not None:
    s = e9[e9.representation.str.contains("learned")].iloc[0]
    check("E9 fellow same field", 0.138, float(s["fellow_eye_same_field_R@1"]))
    check("E9 same eye diff field", 0.102, float(s["same_eye_diff_field_R@1"]))
    check("E9 fellow diff field", 0.072, float(s["fellow_eye_diff_field_R@1"]))
    check("E9 ratio", 1.41, float(s["ratio_sameeye_over_fellow_matched_field"]))

for dsfx, name, q_pat, q_img, q_gap, q_sd, q_pos in (
        ("", "PAPILA", 0.749, 0.798, 0.049, 0.016, 10),
        ("_odir", "ODIR", 0.760, 0.765, 0.005, 0.004, 8),
        ("_deepdrid", "DeepDRiD", 0.824, 0.823, -0.000, 0.010, 5)):
    r = infl(dsfx)
    if r:
        check(f"{name} patient AUC", q_pat, r["pat"])
        check(f"{name} image AUC", q_img, r["img"])
        check(f"{name} inflation", q_gap, r["gap"])
        check(f"{name} inflation sd", q_sd, r["sd"])
        check(f"{name} repeats>0", q_pos, r["pos"])

e3o = read("e3_mitigation_odir.csv")
if e3o is not None:
    base = e3o[e3o.method.str.startswith("none")].iloc[0]
    check("ODIR CPMP base R@1", 0.292, float(base["R@1"]))
    check("ODIR CPMP base AUC", 0.757, float(base["auc"]))
    worst = e3o.loc[e3o["R@1"].idxmax()]
    check("ODIR CPMP worst R@1", 0.343, float(worst["R@1"]))
adv = read("adv_papila.csv")
if adv is not None:
    zp = adv[adv.representation == "z_path"]
    check("PAPILA adversarial R@1", 0.336, float(zp["ret_R@1"].mean()))
for dsfx, name, q_r, q_a in (("", "PAPILA", 0.369, 0.838),
                             ("_odir", "ODIR", 0.268, 0.763)):
    check(f"{name} bengraham R@1", q_r, leak(dsfx, "dinov2_vitb14", "canonbg", "R@1"))
    check(f"{name} bengraham AUC", q_a, leak(dsfx, "dinov2_vitb14", "canonbg", "task_auc"))

for dsfx, name, q_r, q_auc, q_gap in (("", "PAPILA", 0.139, 0.811, 0.058),
                                      ("_odir", "ODIR", 0.222, 0.728, 0.016),
                                      ("_deepdrid", "DeepDRiD", 0.070, 0.762, -0.004)):
    check(f"{name} RETFound R@1", q_r, leak(dsfx, "retfound_cfp", "canon", "R@1"))
    check(f"{name} RETFound AUC", q_auc, leak(dsfx, "retfound_cfp", "canon", "task_auc"))
    r = infl(dsfx, "_retfound_cfp")
    if r:
        check(f"{name} RETFound inflation", q_gap, r["gap"])
check("PAPILA RETFound patient AUC", 0.788,
      (infl("", "_retfound_cfp") or {}).get("pat"))

e8 = read("e8_asymmetry_odir.csv")
if e8 is not None:
    g = e8.groupby("method").agg(acc=("acc", "mean"), auc=("auc", "mean"))
    for m, q_acc, q_auc in (
            ("concatenation [z(A); z(B)]", 0.734, 0.814),
            ("difference  z(A) - z(B)", 0.730, 0.813),
            ("per-eye (all training images)", 0.688, 0.761),
            ("per-eye (matched data)", 0.668, 0.726)):
        if m in g.index:
            check(f"E8 {m[:22]} acc", q_acc, float(g.loc[m, "acc"]))
            check(f"E8 {m[:22]} auc", q_auc, float(g.loc[m, "auc"]))

dp = PATHS.tables / "t0_bilateral_discordance.csv"
if dp.is_file():
    dd = pd.read_csv(dp)
    for ds, lab, q in (("papila", "y_glaucoma", 2.87),
                       ("odir", "y_abnormal", 22.25),
                       ("deepdrid", "y_referable", 29.25)):
        row = dd[(dd.dataset == ds) & (dd.label == lab)]
        if not row.empty:
            check(f"{ds} discordance %", q, float(row.discordant_pct.iloc[0]))


def main() -> int:
    print("Verifying draft numbers against outputs/results/")
    print("=" * 68)
    bad = missing = 0
    for label, quoted, actual in checks:
        if actual is None:
            print(f"  [MISSING] {label:<34} quoted {quoted}")
            missing += 1
            continue
        tol = tolerance(quoted)
        ok = abs(quoted - actual) <= tol
        if not ok:
            bad += 1
        mark = "ok  " if ok else "WRONG"
        print(f"  [{mark}] {label:<34} quoted {quoted:>8.3f}   actual {actual:>8.3f}")
    print("=" * 68)
    print(f"{len(checks)} checks   {bad} mismatched   {missing} missing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
