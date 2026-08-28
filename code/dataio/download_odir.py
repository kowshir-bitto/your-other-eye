
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

DEFAULT_SLUG = "andrewmvd/ocular-disease-recognition-odir5k"
API = "https://www.kaggle.com/api/v1"
CHUNK = 1 << 20

SETUP_HELP = """
  Kaggle credentials not found.

  1. Sign in at https://www.kaggle.com  and open  https://www.kaggle.com/settings
  2. Scroll to the "API" section and click  [Create New Token]
     -> your browser downloads  kaggle.json
  3. Move that file to:
        {dest}
  4. Re-run:  python code/dataio/download_odir.py

  (Alternatively export KAGGLE_USERNAME and KAGGLE_KEY as environment
   variables and re-run.)
"""


def load_credentials() -> tuple[str, str] | None:
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key

    for cand in (
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"
        if os.environ.get("KAGGLE_CONFIG_DIR") else None,
        Path.home() / ".kaggle" / "kaggle.json",
        PATHS.root / "kaggle.json",
    ):
        if cand and cand.is_file():
            try:
                blob = json.loads(cand.read_text(encoding="utf-8"))
                if blob.get("username") and blob.get("key"):
                    print(f"  [auth ] using {cand}")
                    return blob["username"], blob["key"]
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  [warn ] could not read {cand}: {exc}")
    return None


def verify(auth: HTTPBasicAuth, slug: str) -> dict | None:
    owner, name = slug.split("/", 1)
    r = requests.get(f"{API}/datasets/list", params={"search": name},
                     auth=auth, timeout=60)
    if r.status_code == 401:
        print("!! 401 Unauthorized - the API token is invalid or expired.")
        return None
    r.raise_for_status()

    r = requests.get(f"{API}/datasets/view/{owner}/{name}", auth=auth, timeout=60)
    if r.status_code == 403:
        print(f"!! 403 Forbidden for {slug}")
        print(f"   Open https://www.kaggle.com/datasets/{slug} in a browser and")
        print("   accept the dataset rules once, then re-run this script.")
        return None
    if r.status_code == 404:
        print(f"!! 404 - dataset {slug} not found. Pass a different --slug.")
        return None
    r.raise_for_status()
    meta = r.json()
    print(f"  title  : {meta.get('title')}")
    print(f"  size   : {meta.get('totalBytes', 0)/1e9:.2f} GB")
    print(f"  license: {meta.get('licenseName')}")
    return meta


def download_dataset(auth: HTTPBasicAuth, slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{slug.split('/')[1]}.zip"
    tmp = zip_path.with_suffix(".zip.part")

    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"  [skip ] {zip_path.name} present ({zip_path.stat().st_size/1e9:.2f} GB)")
        return zip_path

    url = f"{API}/datasets/download/{slug}"
    print(f"  [get  ] {url}")
    with requests.get(url, auth=auth, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  [dl   ] {done/1e6:9.1f} / {total/1e6:.1f} MB "
                          f"({100*done/total:5.1f}%)", end="", flush=True)
                elif done % (50 << 20) < CHUNK:
                    print(f"\r  [dl   ] {done/1e6:9.1f} MB", end="", flush=True)
        print()
    tmp.replace(zip_path)
    return zip_path


def unpack(zip_path: Path, out_dir: Path) -> None:
    print(f"  [unzip] -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"  [unzip] {len(names)} entries")
        for i, n in enumerate(names, 1):
            zf.extract(n, out_dir)
            if i % 500 == 0:
                print(f"\r  [unzip] {i}/{len(names)}", end="", flush=True)
    print(f"\r  [unzip] {len(names)}/{len(names)} done")


def summarise(root: Path) -> None:
    exts = {".jpg", ".jpeg", ".png"}
    imgs, sheets = [], []
    for p in root.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in exts:
                imgs.append(p)
            elif p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                sheets.append(p)
    print(f"\n  images={len(imgs)}  tabular={len(sheets)}")
    for s in sheets:
        print(f"    tabular: {s.relative_to(root)}")
    dirs = sorted({p.parent.relative_to(root) for p in imgs})
    for d in dirs[:10]:
        n = len([p for p in imgs if p.parent.relative_to(root) == d])
        print(f"    {d}/  [{n} images]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--check", action="store_true", help="verify credentials only")
    ap.add_argument("--keep-zip", action="store_true", default=True)
    args = ap.parse_args()

    print("ODIR-5K download")
    print("=" * 60)

    creds = load_credentials()
    if creds is None:
        print(SETUP_HELP.format(dest=Path.home() / ".kaggle" / "kaggle.json"))
        return 2

    auth = HTTPBasicAuth(*creds)
    meta = verify(auth, args.slug)
    if meta is None:
        return 1
    if args.check:
        print("\n  credentials OK, dataset reachable.")
        return 0

    out = PATHS.odir
    zip_path = download_dataset(auth, args.slug, out)
    unpack(zip_path, out)
    if not args.keep_zip:
        zip_path.unlink()
    summarise(out)
    print("\nDone ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
