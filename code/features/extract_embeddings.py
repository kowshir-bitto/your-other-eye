
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BACKBONES, DEFAULT_BACKBONE, EXTRACT_BATCH, PATHS, TORCH_THREADS, VARIANTS
from dataio.preprocess import load_fundus


class FundusSet(Dataset):

    def __init__(self, df: pd.DataFrame, variant: str, size: int,
                 norm: tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
        self.paths = [PATHS.raw / p for p in df["image_path"]]
        self.eyes = df["eye"].tolist()
        self.ids = df["image_id"].tolist()
        self.opts = VARIANTS[variant]
        self.size = size
        self.mean, self.std = norm

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        arr = load_fundus(self.paths[i], eye=self.eyes[i], size=self.size,
                          mean=self.mean, std=self.std, **self.opts)
        return torch.from_numpy(arr), i


def build_backbone(key: str) -> tuple[torch.nn.Module, int, tuple]:
    spec = BACKBONES[key]

    if spec.get("hf_repo"):
        from huggingface_hub import hf_hub_download

        model = timm.create_model(spec["timm_name"], pretrained=False,
                                  num_classes=0, **spec.get("kwargs", {}))
        ckpt_path = hf_hub_download(spec["hf_repo"], spec["hf_file"])
        sd = torch.load(ckpt_path, map_location="cpu")
        for k in ("model", "state_dict"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if not k.startswith("head.")]
        if real_missing:
            raise RuntimeError(
                f"{key}: {len(real_missing)} weights missing from the checkpoint, "
                f"e.g. {real_missing[:4]}")
        print(f"  [ckpt ] {spec['hf_repo']}  loaded "
              f"({len(sd)} tensors, {len(unexpected)} unused)")
    else:
        model = timm.create_model(spec["timm_name"], pretrained=True,
                                  num_classes=0, **spec.get("kwargs", {}))

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, spec["size"], spec.get(
        "norm", ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))


@torch.no_grad()
def extract(df: pd.DataFrame, backbone_key: str, variant: str,
            batch_size: int = EXTRACT_BATCH, workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    model, size, norm = build_backbone(backbone_key)
    ds = FundusSet(df, variant, size, norm=norm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=False)

    feats: list[np.ndarray] = []
    order: list[int] = []
    n = len(ds)
    t0 = time.time()
    seen = 0
    for x, idx in dl:
        f = model(x)
        feats.append(f.numpy().astype(np.float32))
        order.extend(idx.tolist())
        seen += len(idx)
        el = time.time() - t0
        rate = seen / max(el, 1e-9)
        eta = (n - seen) / max(rate, 1e-9)
        print(f"\r    {seen:>6}/{n}  {rate:5.2f} img/s  elapsed {el/60:5.1f}m  eta {eta/60:5.1f}m",
              end="", flush=True)
    print()

    feat = np.concatenate(feats, axis=0)
    ids = np.array([ds.ids[i] for i in order], dtype=object)
    return ids, feat


def out_path(dataset: str, backbone: str, variant: str) -> Path:
    return PATHS.embeddings / f"{dataset}__{backbone}__{variant}.npz"


def run_one(dataset: str, backbone: str, variant: str, force: bool, workers: int) -> None:
    dst = out_path(dataset, backbone, variant)
    man = PATHS.manifests / f"{dataset}.csv"
    if not man.is_file():
        print(f"  [skip ] no manifest {man}; run build_manifest.py")
        return
    df = pd.read_csv(man)

    if dst.is_file() and not force:
        cached = np.load(dst, allow_pickle=True)
        if len(cached["image_id"]) == len(df):
            print(f"  [skip ] {dst.name} present ({cached['feat'].shape})")
            return
        print(f"  [stale] {dst.name} has {len(cached['image_id'])} rows, manifest has {len(df)}")

    print(f"  [run  ] {dataset} | {backbone} | {variant} | {len(df)} images")
    ids, feat = extract(df, backbone, variant, workers=workers)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        image_id=ids,
        feat=feat,
        meta=json.dumps({
            "dataset": dataset,
            "backbone": backbone,
            "timm_name": BACKBONES[backbone]["timm_name"],
            "variant": variant,
            "variant_opts": VARIANTS[variant],
            "dim": int(feat.shape[1]),
            "n": int(feat.shape[0]),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }),
    )
    mb = dst.stat().st_size / 1e6
    print(f"  [done ] {dst.name}  shape={feat.shape}  {mb:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["papila"],
                    help="papila / odir / all")
    ap.add_argument("--backbone", nargs="+", default=[DEFAULT_BACKBONE],
                    choices=list(BACKBONES) + ["all"])
    ap.add_argument("--variant", nargs="+", default=["canon"],
                    choices=list(VARIANTS) + ["all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=TORCH_THREADS)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    datasets = (["papila", "odir", "deepdrid_fields"] if "all" in args.dataset
                else args.dataset)
    backbones = (list(BACKBONES) if "all" in args.backbone else args.backbone)
    variants = (list(VARIANTS) if "all" in args.variant else args.variant)

    print("Extract embeddings")
    print("=" * 60)
    print(f"  threads={args.threads}  batch={EXTRACT_BATCH}")
    print(f"  datasets={datasets}  backbones={backbones}  variants={variants}")

    t0 = time.time()
    for d in datasets:
        for b in backbones:
            for v in variants:
                run_one(d, b, v, args.force, args.workers)
    print(f"\nTotal {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
