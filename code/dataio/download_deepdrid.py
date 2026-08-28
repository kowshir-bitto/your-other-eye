
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

REPO = "https://github.com/deepdrdoc/DeepDRiD.git"
RAW = "https://raw.githubusercontent.com/deepdrdoc/DeepDRiD/master"
SPLITS = {
    "train": "regular_fundus_images/regular-fundus-training/regular-fundus-training.csv",
    "val": "regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv",
}


def fetch_labels() -> pd.DataFrame:
    frames = []
    for split, rel in SPLITS.items():
        r = requests.get(f"{RAW}/{rel}", timeout=90)
        if r.status_code != 200:
            print(f"  [warn ] {split}: HTTP {r.status_code}")
            continue
        df = pd.read_csv(io.StringIO(r.text))
        df["src_split"] = split
        frames.append(df)
        print(f"  [csv  ] {split}: {len(df)} rows")
    if not frames:
        raise RuntimeError("no label CSVs could be fetched")
    return pd.concat(frames, ignore_index=True)


def clone(dest: Path) -> Path:
    repo_dir = dest / "DeepDRiD"
    if (repo_dir / "regular_fundus_images").is_dir():
        n = len(list(repo_dir.rglob("*.jpg")))
        print(f"  [skip ] already cloned ({n} jpg)")
        return repo_dir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  [clone] {REPO} -> {repo_dir}  (~1.3 GB, shallow)")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", REPO, str(repo_dir)],
        check=True,
    )
    return repo_dir


def build_metadata(labels: pd.DataFrame, repo_dir: Path) -> pd.DataFrame:
    index = {p.name: p for p in repo_dir.rglob("*.jpg")}
    print(f"  [index] {len(index)} image files on disk")

    rows = []
    for _, r in labels.iterrows():
        iid = str(r["image_id"])
        m = re.fullmatch(r"(\d+)_([lr])(\d+)", iid)
        if not m:
            continue
        pid, side, field = m.group(1), m.group(2), int(m.group(3))
        eye = "OS" if side == "l" else "OD"
        grade = r["left_eye_DR_Level"] if eye == "OS" else r["right_eye_DR_Level"]
        path = index.get(f"{iid}.jpg")
        if path is None:
            continue
        rows.append({
            "patient_id": int(pid), "eye": eye, "field": field,
            "image_id": iid, "filename": f"{iid}.jpg",
            "path": str(path.relative_to(PATHS.raw)).replace("\\", "/"),
            "dr_grade": grade, "patient_dr": r.get("patient_DR_Level"),
            "quality": r.get("Overall quality"), "clarity": r.get("Clarity"),
            "src_split": r["src_split"],
        })
    meta = pd.DataFrame(rows)
    print(f"  [meta ] {len(meta)} images matched to labels")
    return meta


def report(meta: pd.DataFrame) -> None:
    per = meta.groupby("patient_id")["eye"].nunique()
    print(f"\n  patients={len(per)}  with both eyes={int((per == 2).sum())}")
    print(f"  fields per eye: "
          f"{meta.groupby(['patient_id','eye'])['field'].nunique().value_counts().to_dict()}")
    print(f"  DR grade distribution: "
          f"{meta.dr_grade.value_counts().sort_index().to_dict()}")

    eye_grade = (meta.dropna(subset=["dr_grade"])
                     .groupby(["patient_id", "eye"])["dr_grade"].first().unstack())
    if {"OD", "OS"}.issubset(eye_grade.columns):
        w = eye_grade.dropna()
        exact = (w["OD"] != w["OS"]).mean()
        binary = ((w["OD"] > 0) != (w["OS"] > 0)).mean()
        print(f"\n  BILATERAL DISCORDANCE over {len(w)} pairs:")
        print(f"    exact DR grade differs : {100*exact:.1f}%")
        print(f"    referable (grade>0)    : {100*binary:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    out = PATHS.raw / "deepdrid"
    print("DeepDRiD download")
    print("=" * 60)

    labels = fetch_labels()
    if args.inspect:
        print(labels.head().to_string())
        return 0

    repo_dir = clone(out)
    meta = build_metadata(labels, repo_dir)
    meta.to_csv(out / "deepdrid_metadata.csv", index=False)
    report(meta)
    print(f"\nDone -> {out / 'deepdrid_metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
