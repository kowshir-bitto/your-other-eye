
from __future__ import annotations

import argparse
import io
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

REPO = "bumbledeep/odir"
PARQUET = "data/train-00000-of-00001.parquet"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{PARQUET}"
CHUNK = 1 << 20


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1e8:
        print(f"  [skip ] {dest.name} present ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    tmp = dest.with_suffix(".part")
    print(f"  [get  ] {URL}")
    with requests.get(URL, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for c in r.iter_content(CHUNK):
                fh.write(c)
                done += len(c)
                if total:
                    print(f"\r  [dl   ] {done/1e6:8.1f} / {total/1e6:.1f} MB "
                          f"({100*done/total:5.1f}%)", end="", flush=True)
        print()
    tmp.replace(dest)
    return dest


def eye_from_path(p: str | None) -> str | None:
    if not p:
        return None
    m = re.search(r"_(left|right)\b", str(p), flags=re.I)
    if not m:
        return None
    return "OS" if m.group(1).lower() == "left" else "OD"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N rows")
    args = ap.parse_args()

    root = PATHS.odir
    root.mkdir(parents=True, exist_ok=True)
    print("ODIR-5K download (Hugging Face mirror)")
    print("=" * 60)

    pq = download(root / "odir_train.parquet")

    print("  [read ] parquet")
    df = pd.read_parquet(pq)
    print(f"  [read ] {len(df)} rows, columns: {list(df.columns)}")

    sample = df["image"].iloc[0]
    keys = list(sample.keys()) if isinstance(sample, dict) else type(sample)
    print(f"  [image] keys: {keys}")
    paths = df["image"].map(lambda d: d.get("path") if isinstance(d, dict) else None)
    print(f"  [image] example paths: {[p for p in paths.head(4)]}")

    eyes = paths.map(eye_from_path)
    n_known = int(eyes.notna().sum())
    print(f"  [eye  ] recovered laterality for {n_known}/{len(df)} rows")
    if n_known < 0.99 * len(df):
        print("!! laterality could not be recovered from image paths.")
        print("   A bilateral study cannot proceed on guessed sides -- stopping.")
        print(f"   sample paths: {[p for p in paths.head(10)]}")
        return 1

    df = df.assign(eye=eyes, src_path=paths)
    print(f"  [label] {dict(Counter(df['label']).most_common())}")
    per_pat = df.groupby("patient_id")["eye"].nunique()
    print(f"  [pairs] patients={len(per_pat)}  "
          f"with both eyes={int((per_pat == 2).sum())}  "
          f"single-eye={int((per_pat == 1).sum())}")

    if args.inspect:
        return 0

    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(df) if not args.limit else min(args.limit, len(df))
    written = skipped = 0
    for i in range(n):
        r = df.iloc[i]
        eye = r["eye"]
        name = f"{int(r['patient_id']):04d}_{eye}.jpg"
        out = img_dir / name
        if not out.is_file():
            blob = r["image"]
            data = blob.get("bytes") if isinstance(blob, dict) else None
            if data is None:
                skipped += 1
                continue
            out.write_bytes(data)
            written += 1
        rows.append({
            "patient_id": int(r["patient_id"]), "eye": eye,
            "filename": name, "age": r.get("age"), "sex": r.get("sex"),
            "label": r.get("label"), "label_code": r.get("label_code"),
            "src_path": r.get("src_path"),
        })
        if (i + 1) % 500 == 0:
            print(f"\r  [write] {i+1}/{n}", end="", flush=True)
    print(f"\r  [write] {n}/{n}  ({written} new, {skipped} without bytes)")

    meta = pd.DataFrame(rows)
    meta_csv = root / "odir_metadata.csv"
    meta.to_csv(meta_csv, index=False)
    print(f"  [meta ] {meta_csv}  ({len(meta)} rows)")

    on_disk = len(list(img_dir.glob("*.jpg")))
    print(f"\nDone -> {img_dir}  ({on_disk} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
