
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    data: Path = ROOT / "data"
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    manifests: Path = ROOT / "data" / "manifests"

    papila: Path = ROOT / "data" / "raw" / "papila"
    odir: Path = ROOT / "data" / "raw" / "odir5k"

    outputs: Path = ROOT / "outputs"
    embeddings: Path = ROOT / "outputs" / "embeddings"
    checkpoints: Path = ROOT / "outputs" / "checkpoints"
    figures: Path = ROOT / "outputs" / "figures"
    tables: Path = ROOT / "outputs" / "tables"
    logs: Path = ROOT / "outputs" / "logs"
    results: Path = ROOT / "outputs" / "results"

    paper: Path = ROOT / "paper"

    def ensure(self) -> None:
        for f in self.__dataclass_fields__:
            p = getattr(self, f)
            if isinstance(p, Path):
                p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

SEED = 1337

DEVICE = "cuda" if os.environ.get("Q1_FORCE_CUDA") else "cpu"
NUM_WORKERS = 4
TORCH_THREADS = 10
EXTRACT_BATCH = 16

IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

BACKBONES = {
    "dinov2_vitb14": {"timm_name": "vit_base_patch14_dinov2.lvd142m", "dim": 768,
                      "size": 224, "kwargs": {"img_size": 224},
                      "norm": IMAGENET_NORM},
    "dinov2_vits14": {"timm_name": "vit_small_patch14_dinov2.lvd142m", "dim": 384,
                      "size": 224, "kwargs": {"img_size": 224},
                      "norm": IMAGENET_NORM},
    "resnet50_in1k": {"timm_name": "resnet50.a1_in1k", "dim": 2048,
                      "size": 224, "kwargs": {}, "norm": IMAGENET_NORM},
    "retfound_cfp": {"timm_name": "vit_large_patch16_224", "dim": 1024,
                     "size": 224, "kwargs": {},
                     "norm": HALF_NORM,
                     "hf_repo": "bitfount/RETFound_MAE",
                     "hf_file": "pytorch_model.bin"},
}
DEFAULT_BACKBONE = "dinov2_vitb14"

VARIANTS = {
    "raw":     {"do_circle_crop": False, "do_laterality": False, "do_ben_graham": False},
    "crop":    {"do_circle_crop": True,  "do_laterality": False, "do_ben_graham": False},
    "canon":   {"do_circle_crop": True,  "do_laterality": True,  "do_ben_graham": False},
    "canonbg": {"do_circle_crop": True,  "do_laterality": True,  "do_ben_graham": True},
}
DEFAULT_VARIANT = "canon"

IMG_SIZE = 224
CIRCLE_CROP = True
LATERALITY_CANONICAL = True


if __name__ == "__main__":
    PATHS.ensure()
    print(f"ROOT      : {PATHS.root}")
    print(f"DEVICE    : {DEVICE}")
    print(f"BACKBONE  : {DEFAULT_BACKBONE} ({BACKBONES[DEFAULT_BACKBONE]['dim']}-d)")
    for f in Paths.__dataclass_fields__:
        p = getattr(PATHS, f)
        print(f"  {f:<12} {p}  {'OK' if p.exists() else 'MISSING'}")
