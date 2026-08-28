
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

FIGSHARE_ARTICLE = 14798004
CHUNK = 1 << 20


def figshare_files(article_id: int) -> list[dict]:
    r = requests.get(f"https://api.figshare.com/v2/articles/{article_id}", timeout=60)
    r.raise_for_status()
    meta = r.json()
    print(f"  title  : {meta.get('title')}")
    print(f"  doi    : {meta.get('doi')}")
    print(f"  license: {(meta.get('license') or {}).get('name')}")
    return meta.get("files", [])


def download(url: str, dest: Path, expected_size: int | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return dest

    print(f"  [get ] {url}")
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", expected_size or 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  [dl  ] {done/1e6:8.1f} / {total/1e6:.1f} MB  ({pct:5.1f}%)",
                          end="", flush=True)
        print()
    tmp.replace(dest)
    return dest


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [unzip] -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        print(f"  [unzip] {len(members)} entries")
        zf.extractall(out_dir)

    entries = [p for p in out_dir.iterdir() if p.name != "PAPILA.zip"]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        print(f"  [flat ] hoisting {inner.name}/*")
        for child in list(inner.iterdir()):
            target = out_dir / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))
        inner.rmdir()


def summarise(root: Path) -> None:
    print("\n  Layout:")
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            n_img = len([f for f in p.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".bmp"}])
            rel = p.relative_to(root)
            depth = len(rel.parts)
            if depth <= 2:
                extra = f"  [{n_img} images]" if n_img else ""
                print(f"    {'  '*depth}{rel}/{extra}")
    files = [p for p in root.rglob("*") if p.is_file()]
    imgs = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".bmp"}]
    sheets = [p for p in files if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".txt"}]
    print(f"\n  files={len(files)}  images={len(imgs)}  tabular={len(sheets)}")
    for s in sheets[:12]:
        print(f"    tabular: {s.relative_to(root)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--keep-zip", action="store_true", default=True)
    args = ap.parse_args()

    out = PATHS.papila
    out.mkdir(parents=True, exist_ok=True)
    print("PAPILA download")
    print("=" * 60)

    files = figshare_files(FIGSHARE_ARTICLE)
    if not files:
        print("!! no files returned by figshare API")
        return 1

    for f in files:
        zip_path = out / f["name"]
        if args.force and zip_path.exists():
            zip_path.unlink()
        download(f["download_url"], zip_path, expected_size=f.get("size"))

        want = f.get("supplied_md5") or f.get("computed_md5")
        if want:
            got = md5(zip_path)
            ok = "OK" if got == want else "MISMATCH"
            print(f"  [md5 ] {got}  ({ok})")
            if got != want:
                print("!! checksum mismatch - delete the file and retry")
                return 1

        if zip_path.suffix.lower() == ".zip":
            unpack(zip_path, out)
            if not args.keep_zip:
                zip_path.unlink()

    summarise(out)
    print("\nDone ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
