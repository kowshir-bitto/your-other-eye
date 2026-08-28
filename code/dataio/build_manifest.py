
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS, SEED

SCHEMA_CORE = ["dataset", "patient_id", "eye", "image_id", "image_path", "pair_id"]

PAPILA_COLS = [
    "ID", "Age", "Gender", "Diagnosis", "dioptre_1", "dioptre_2", "astigmatism",
    "Phakic", "IOP_Pneumatic", "IOP_Perkins", "Pachymetry", "Axial_Length", "VF_MD",
]
PAPILA_DIAGNOSIS = {0: "healthy", 1: "glaucoma", 2: "suspect"}


def build_papila() -> pd.DataFrame:
    root = PATHS.papila
    img_dir = root / "FundusImages"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"PAPILA images not found at {img_dir}; run download_papila.py")

    def read_sheet(fname: str, eye: str) -> pd.DataFrame:
        df = pd.read_excel(root / "ClinicalData" / fname, header=None, skiprows=3)
        df.columns = PAPILA_COLS
        df["eye"] = eye
        df["pid_num"] = df["ID"].astype(str).str.lstrip("#").str.strip().str.zfill(3)
        return df

    clin = pd.concat(
        [read_sheet("patient_data_od.xlsx", "OD"),
         read_sheet("patient_data_os.xlsx", "OS")],
        ignore_index=True,
    )

    rows = []
    for f in sorted(img_dir.glob("*.jpg")):
        m = re.fullmatch(r"RET(\d+)(OD|OS)", f.stem)
        if not m:
            print(f"  [warn] unexpected filename skipped: {f.name}")
            continue
        rows.append({"pid_num": m.group(1), "eye": m.group(2),
                     "image_path": str(f.relative_to(PATHS.raw)).replace("\\", "/")})
    imgs = pd.DataFrame(rows)

    df = imgs.merge(clin, on=["pid_num", "eye"], how="left", validate="one_to_one")
    missing = df["Diagnosis"].isna().sum()
    if missing:
        print(f"  [warn] {missing} images without clinical rows")

    out = pd.DataFrame({
        "dataset": "papila",
        "patient_id": "papila_" + df["pid_num"],
        "eye": df["eye"],
        "image_path": df["image_path"],
        "y_diagnosis3": df["Diagnosis"].astype("Int64"),
        "y_glaucoma": (df["Diagnosis"] == 1).astype(int),
        "y_nonhealthy": (df["Diagnosis"] > 0).astype(int),
        "y_vf_md": pd.to_numeric(df["VF_MD"], errors="coerce"),
        "a_age": pd.to_numeric(df["Age"], errors="coerce"),
        "a_gender": pd.to_numeric(df["Gender"], errors="coerce").astype("Int64"),
        "a_axial_length": pd.to_numeric(df["Axial_Length"], errors="coerce"),
        "a_pachymetry": pd.to_numeric(df["Pachymetry"], errors="coerce"),
        "a_iop": pd.to_numeric(df["IOP_Pneumatic"], errors="coerce")
                  .fillna(pd.to_numeric(df["IOP_Perkins"], errors="coerce")),
        "a_dioptre_1": pd.to_numeric(df["dioptre_1"], errors="coerce"),
        "a_dioptre_2": pd.to_numeric(df["dioptre_2"], errors="coerce"),
        "a_phakic": pd.to_numeric(df["Phakic"], errors="coerce").astype("Int64"),
    })
    out["image_id"] = out["patient_id"] + "_" + out["eye"]
    out["pair_id"] = out["patient_id"]
    return out


ODIR_CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
ODIR_NAMES = {
    "N": "normal", "D": "diabetes", "G": "glaucoma", "C": "cataract",
    "A": "amd", "H": "hypertension", "M": "myopia", "O": "other",
}


ODIR_HF_LABELS = {
    "normal": "y_normal",
    "diabetes": "y_diabetes",
    "glaucoma": "y_glaucoma",
    "cataract": "y_cataract",
    "age_related_macular_degeneration": "y_amd",
    "hypertension": "y_hypertension",
    "pathological_myopia": "y_myopia",
    "other_diseases_abnormalities": "y_other",
}


def _find_odir_csv(root: Path) -> Path | None:
    for name in ("odir_metadata.csv", "full_df.csv", "data.csv", "odir.csv"):
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    hits = list(root.rglob("*.csv"))
    return hits[0] if hits else None


def build_odir_hf(csv: Path, root: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv)
    img_dir = csv.parent / "images"

    out = pd.DataFrame({
        "dataset": "odir",
        "patient_id": "odir_" + raw["patient_id"].astype(int).astype(str).str.zfill(4),
        "eye": raw["eye"].astype(str).str.upper(),
        "image_path": raw["filename"].map(
            lambda n: str((img_dir / n).relative_to(PATHS.raw)).replace("\\", "/")),
        "a_age": pd.to_numeric(raw["age"], errors="coerce"),
        "a_gender": raw["sex"].astype(str).str.upper().str[0].map({"M": 1, "F": 0}).astype("Int64"),
    })

    lab = raw["label"].astype(str)
    for value, col in ODIR_HF_LABELS.items():
        out[col] = (lab == value).astype(int)
    out["y_abnormal"] = (lab != "normal").astype(int)

    out["image_id"] = out["patient_id"] + "_" + out["eye"]
    out["pair_id"] = out["patient_id"]

    missing = [p for p in out["image_path"]
               if not (PATHS.raw / p).is_file()]
    if missing:
        print(f"  [warn ] {len(missing)} image files absent, dropping")
        out = out[[(PATHS.raw / p).is_file() for p in out["image_path"]]]

    counts = out.groupby("pair_id")["eye"].nunique()
    complete = counts[counts == 2].index
    n_drop = (~out["pair_id"].isin(complete)).sum()
    if n_drop:
        print(f"  [odir ] dropped {n_drop} images from incomplete pairs")
    return out[out["pair_id"].isin(complete)].copy()


def build_odir() -> pd.DataFrame:
    root = PATHS.odir
    if not root.is_dir() or not any(root.rglob("*.jpg")):
        raise FileNotFoundError(
            f"ODIR images not found under {root}; run download_odir.py first")

    csv = _find_odir_csv(root)
    if csv is None:
        raise FileNotFoundError(f"no annotation CSV found under {root}")
    print(f"  [odir ] annotations: {csv.relative_to(root)}")
    raw = pd.read_csv(csv)
    print(f"  [odir ] columns: {list(raw.columns)[:16]}")

    if {"patient_id", "eye", "filename", "label"}.issubset(raw.columns):
        return build_odir_hf(csv, root)

    img_index = {p.name: p for p in root.rglob("*.jpg")}
    print(f"  [odir ] indexed {len(img_index)} image files")

    recs: list[dict] = []
    cols = {c.lower().strip(): c for c in raw.columns}

    def col(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    c_id = col("id", "patient id", "patient_id")
    c_age = col("patient age", "patient_age", "age")
    c_sex = col("patient sex", "patient_sex", "sex", "gender")
    c_left = col("left-fundus", "left_fundus")
    c_right = col("right-fundus", "right_fundus")
    c_lkw = col("left-diagnostic keywords", "left_diagnostic_keywords")
    c_rkw = col("right-diagnostic keywords", "right_diagnostic_keywords")
    c_file = col("filename", "file", "image")
    c_eyecol = col("eye")

    if c_left and c_right:
        for _, r in raw.iterrows():
            for eye, cfile, ckw in (("OS", c_left, c_lkw), ("OD", c_right, c_rkw)):
                fn = r.get(cfile)
                if not isinstance(fn, str):
                    continue
                recs.append({
                    "pid": str(r[c_id]) if c_id else Path(fn).stem.split("_")[0],
                    "eye": eye, "fname": fn,
                    "kw": (r.get(ckw) if ckw else "") or "",
                    "age": r.get(c_age) if c_age else np.nan,
                    "sex": r.get(c_sex) if c_sex else np.nan,
                    **{f"y_{ODIR_NAMES[k]}": r.get(k, np.nan) for k in ODIR_CLASSES},
                })
    elif c_file:
        for _, r in raw.iterrows():
            fn = str(r[c_file])
            stem = Path(fn).stem
            eye = (str(r[c_eyecol]).upper() if c_eyecol
                   else ("OS" if stem.lower().endswith("left") else "OD"))
            eye = {"LEFT": "OS", "L": "OS", "RIGHT": "OD", "R": "OD"}.get(eye, eye)
            recs.append({
                "pid": str(r[c_id]) if c_id else stem.split("_")[0],
                "eye": eye, "fname": fn,
                "kw": (r.get(c_lkw if eye == "OS" else c_rkw) if c_lkw else "") or "",
                "age": r.get(c_age) if c_age else np.nan,
                "sex": r.get(c_sex) if c_sex else np.nan,
                **{f"y_{ODIR_NAMES[k]}": r.get(k, np.nan) for k in ODIR_CLASSES},
            })
    else:
        raise ValueError(f"cannot interpret ODIR annotation schema: {list(raw.columns)}")

    df = pd.DataFrame(recs).drop_duplicates(subset=["pid", "eye"])

    df["_path"] = df["fname"].map(lambda n: img_index.get(str(n)))
    n_missing = df["_path"].isna().sum()
    if n_missing:
        print(f"  [warn ] {n_missing}/{len(df)} annotated images not on disk; dropped")
        df = df[df["_path"].notna()].copy()

    out = pd.DataFrame({
        "dataset": "odir",
        "patient_id": "odir_" + df["pid"].astype(str).str.zfill(4),
        "eye": df["eye"],
        "image_path": df["_path"].map(
            lambda p: str(Path(p).relative_to(PATHS.raw)).replace("\\", "/")),
        "a_age": pd.to_numeric(df["age"], errors="coerce"),
        "a_gender": df["sex"].astype(str).str.upper().str[0].map({"M": 1, "F": 0}).astype("Int64"),
        "kw": df["kw"].astype(str).str.lower(),
    })
    for k in ODIR_CLASSES:
        c = f"y_{ODIR_NAMES[k]}"
        if c in df:
            out[c + "_pat"] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    kw = out["kw"]
    out["y_normal"] = kw.str.contains("normal fundus").astype(int)
    out["y_diabetes"] = kw.str.contains("diabetic retinopathy|proliferative retinopathy").astype(int)
    out["y_glaucoma"] = kw.str.contains("glaucoma").astype(int)
    out["y_cataract"] = kw.str.contains("cataract").astype(int)
    out["y_amd"] = kw.str.contains("age-related macular degeneration|macular degeneration").astype(int)
    out["y_hypertension"] = kw.str.contains("hypertensive retinopathy").astype(int)
    out["y_myopia"] = kw.str.contains("myopi").astype(int)
    out["y_abnormal"] = (1 - out["y_normal"]).astype(int)
    out = out.drop(columns=["kw"])

    out["image_id"] = out["patient_id"] + "_" + out["eye"]
    out["pair_id"] = out["patient_id"]

    counts = out.groupby("pair_id")["eye"].nunique()
    complete = counts[counts == 2].index
    dropped = out[~out["pair_id"].isin(complete)]
    if len(dropped):
        print(f"  [odir ] dropped {len(dropped)} images from incomplete pairs")
    return out[out["pair_id"].isin(complete)].copy()


def build_deepdrid(fields: str = "first") -> pd.DataFrame:
    root = PATHS.raw / "deepdrid"
    csv = root / "deepdrid_metadata.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} missing; run download_deepdrid.py")
    raw = pd.read_csv(csv)

    if fields == "first":
        raw = (raw.sort_values(["patient_id", "eye", "field"])
                  .groupby(["patient_id", "eye"], as_index=False).first())

    grade = pd.to_numeric(raw["dr_grade"], errors="coerce")
    out = pd.DataFrame({
        "dataset": "deepdrid",
        "patient_id": "ddr_" + raw["patient_id"].astype(int).astype(str).str.zfill(4),
        "eye": raw["eye"],
        "image_path": raw["path"],
        "field": raw["field"].astype(int),
        "y_dr_grade": grade.astype("Int64"),
        "y_referable": (grade >= 1).astype("Int64"),
        "y_severe": (grade >= 2).astype("Int64"),
        "a_quality": pd.to_numeric(raw["quality"], errors="coerce"),
        "a_clarity": pd.to_numeric(raw["clarity"], errors="coerce"),
    })
    out["image_id"] = (out["patient_id"] + "_" + out["eye"]
                       + "_f" + out["field"].astype(str))
    out["pair_id"] = out["patient_id"]

    missing = [p for p in out["image_path"] if not (PATHS.raw / p).is_file()]
    if missing:
        print(f"  [warn ] {len(missing)} files absent, dropping")
        out = out[[(PATHS.raw / p).is_file() for p in out["image_path"]]]

    counts = out.groupby("pair_id")["eye"].nunique()
    complete = counts[counts == 2].index
    n_drop = (~out["pair_id"].isin(complete)).sum()
    if n_drop:
        print(f"  [ddr  ] dropped {n_drop} images from incomplete pairs")
    return out[out["pair_id"].isin(complete)].copy()


def assign_splits(
    df: pd.DataFrame,
    strat_col: str,
    frac=(0.6, 0.15, 0.25),
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if strat_col in df.columns:
        pat_label = df.groupby("patient_id")[strat_col].max().fillna(0).astype(int)
    else:
        pat_label = pd.Series(0, index=df["patient_id"].unique())

    split_of: dict[str, str] = {}
    for lab, grp in pat_label.groupby(pat_label):
        pats = grp.index.to_numpy()
        rng.shuffle(pats)
        n = len(pats)
        n_tr = int(round(frac[0] * n))
        n_va = int(round(frac[1] * n))
        for p in pats[:n_tr]:
            split_of[p] = "train"
        for p in pats[n_tr:n_tr + n_va]:
            split_of[p] = "val"
        for p in pats[n_tr + n_va:]:
            split_of[p] = "test"

    df = df.copy()
    df["split"] = df["patient_id"].map(split_of)

    idx = df.index.to_numpy()
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(idx)
    n = len(idx)
    n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
    img_split = pd.Series("test", index=df.index)
    img_split.loc[idx[:n_tr]] = "train"
    img_split.loc[idx[n_tr:n_tr + n_va]] = "val"
    df["split_image"] = img_split
    return df


def report(df: pd.DataFrame, name: str) -> None:
    print(f"\n  {name}: {len(df)} images / {df['patient_id'].nunique()} patients")
    print(f"    eyes per patient: "
          f"{df.groupby('patient_id')['eye'].nunique().value_counts().to_dict()}")
    if "split" in df:
        pv = (df.groupby("split")
                .agg(images=("image_id", "size"), patients=("patient_id", "nunique")))
        print("    splits (patient-grouped):")
        for s, r in pv.iterrows():
            print(f"      {s:<6} {r['images']:>6} images  {r['patients']:>5} patients")
    ycols = [c for c in df.columns if c.startswith("y_") and df[c].dropna().isin([0, 1]).all()]
    if ycols:
        print("    label prevalence:")
        for c in ycols[:12]:
            pos = int(df[c].sum())
            print(f"      {c:<22} {pos:>5} / {len(df)}  ({100*pos/len(df):5.1f}%)")


def pair_discordance(df: pd.DataFrame, ycols: list[str]) -> pd.DataFrame:
    rows = []
    for c in ycols:
        if c not in df.columns:
            continue
        w = df.pivot_table(index="pair_id", columns="eye", values=c, aggfunc="first")
        if not {"OD", "OS"}.issubset(w.columns):
            continue
        w = w.dropna()
        if w.empty:
            continue
        disc = (w["OD"] != w["OS"]).sum()
        both_pos = ((w["OD"] == 1) & (w["OS"] == 1)).sum()
        rows.append({
            "label": c, "pairs": len(w),
            "discordant": int(disc), "discordant_pct": round(100 * disc / len(w), 2),
            "both_positive": int(both_pos),
            "pos_rate": round(100 * (w["OD"].sum() + w["OS"].sum()) / (2 * len(w)), 2),
        })
    return pd.DataFrame(rows).sort_values("discordant_pct", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["papila", "odir", "deepdrid", "all"],
                    default="all")
    args = ap.parse_args()

    PATHS.manifests.mkdir(parents=True, exist_ok=True)
    print("Build bilateral manifest")
    print("=" * 60)

    built: dict[str, pd.DataFrame] = {}

    if args.dataset in ("papila", "all"):
        try:
            p = build_papila()
            p = assign_splits(p, "y_nonhealthy")
            report(p, "PAPILA")
            p.to_csv(PATHS.manifests / "papila.csv", index=False)
            built["papila"] = p
        except FileNotFoundError as e:
            print(f"  [skip ] papila: {e}")

    if args.dataset in ("odir", "all"):
        try:
            o = build_odir()
            o = assign_splits(o, "y_abnormal")
            report(o, "ODIR-5K")
            o.to_csv(PATHS.manifests / "odir.csv", index=False)
            built["odir"] = o
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip ] odir: {e}")

    if args.dataset in ("deepdrid", "all"):
        try:
            d = assign_splits(build_deepdrid("first"), "y_referable")
            report(d, "DeepDRiD")
            d.to_csv(PATHS.manifests / "deepdrid.csv", index=False)
            built["deepdrid"] = d
            df_all = assign_splits(build_deepdrid("all"), "y_referable")
            df_all.to_csv(PATHS.manifests / "deepdrid_fields.csv", index=False)
            print(f"  [ddr  ] deepdrid_fields.csv: {len(df_all)} images "
                  f"(both fields, for the same-eye control)")
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip ] deepdrid: {e}")

    if not built:
        print("\n!! nothing built")
        return 1

    print("\n" + "=" * 60)
    print("BILATERAL DISCORDANCE  (how often the two eyes disagree)")
    print("=" * 60)
    all_disc = []
    for name, d in built.items():
        ycols = [c for c in d.columns
                 if c.startswith("y_") and d[c].dropna().isin([0, 1]).all()]
        t = pair_discordance(d, ycols)
        if not t.empty:
            t.insert(0, "dataset", name)
            print(f"\n{name}:")
            print(t.to_string(index=False))
            all_disc.append(t)
    if all_disc:
        disc = pd.concat(all_disc, ignore_index=True)
        disc.to_csv(PATHS.tables / "t0_bilateral_discordance.csv", index=False)
        print(f"\n  -> {PATHS.tables / 't0_bilateral_discordance.csv'}")

    combined = pd.concat(built.values(), ignore_index=True)
    combined.to_csv(PATHS.manifests / "all.csv", index=False)
    print(f"\nWrote {PATHS.manifests / 'all.csv'}  ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
