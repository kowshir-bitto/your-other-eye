
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS, SEED


@dataclass
class EmbeddingStore:

    df: pd.DataFrame
    feat: np.ndarray
    meta: dict

    @classmethod
    def load(cls, dataset: str, backbone: str, variant: str = "canon") -> "EmbeddingStore":
        npz_path = PATHS.embeddings / f"{dataset}__{backbone}__{variant}.npz"
        man_path = PATHS.manifests / f"{dataset}.csv"
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"{npz_path} missing; run:\n"
                f"  python code/features/extract_embeddings.py "
                f"--dataset {dataset} --backbone {backbone} --variant {variant}")
        if not man_path.is_file():
            raise FileNotFoundError(f"{man_path} missing; run build_manifest.py")

        blob = np.load(npz_path, allow_pickle=True)
        ids = [str(i) for i in blob["image_id"]]
        feat = blob["feat"].astype(np.float32)
        meta = json.loads(str(blob["meta"]))

        df = pd.read_csv(man_path)
        pos = {i: k for k, i in enumerate(ids)}
        missing = set(df["image_id"]) - set(pos)
        if missing:
            raise ValueError(f"{len(missing)} manifest rows have no embedding, "
                             f"e.g. {sorted(missing)[:3]}")
        df = df.copy()
        df["_row"] = df["image_id"].map(pos)
        df = df.sort_values("_row").reset_index(drop=True)
        feat = feat[df["_row"].to_numpy()]
        df = df.drop(columns=["_row"])
        return cls(df=df, feat=feat, meta=meta)

    @property
    def dim(self) -> int:
        return int(self.feat.shape[1])

    def subset(self, mask: np.ndarray | pd.Series) -> "EmbeddingStore":
        m = np.asarray(mask, dtype=bool)
        return EmbeddingStore(df=self.df[m].reset_index(drop=True),
                              feat=self.feat[m], meta=self.meta)

    def split(self, name: str, col: str = "split") -> "EmbeddingStore":
        return self.subset(self.df[col].to_numpy() == name)

    def standardise(self, mean: np.ndarray | None = None, std: np.ndarray | None = None,
                    rel_floor: float = 1e-3):
        if mean is None:
            mean = self.feat.mean(0, keepdims=True)
        if std is None:
            raw = self.feat.std(0, keepdims=True)
            floor = rel_floor * float(np.median(raw))
            std = np.where(raw > floor, raw, np.inf)
        z = (self.feat - mean) / std
        self.feat = np.clip(z, -8.0, 8.0).astype(np.float32)
        return mean, std

    def labels(self, col: str) -> np.ndarray:
        return pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=np.float32)

    def pair_index(self) -> np.ndarray:
        codes, _ = pd.factorize(self.df["pair_id"])
        return codes.astype(np.int64)

    def patient_index(self) -> np.ndarray:
        codes, _ = pd.factorize(self.df["patient_id"])
        return codes.astype(np.int64)

    def complete_pairs_only(self) -> "EmbeddingStore":
        n = self.df.groupby("pair_id")["eye"].transform("nunique")
        return self.subset((n == 2).to_numpy())

    def as_tensors(self, label_cols: list[str] | None = None) -> dict[str, torch.Tensor]:
        out = {
            "x": torch.from_numpy(self.feat),
            "pair_idx": torch.from_numpy(self.pair_index()),
            "patient_idx": torch.from_numpy(self.patient_index()),
        }
        if label_cols:
            y = np.stack([self.labels(c) for c in label_cols], axis=1)
            out["y"] = torch.from_numpy(y)
        return out

    def describe(self) -> str:
        return (f"{self.meta.get('dataset')}/{self.meta.get('backbone')}/"
                f"{self.meta.get('variant')}  n={len(self.df)}  d={self.dim}  "
                f"patients={self.df['patient_id'].nunique()}")


class PairBatchSampler(Sampler[list[int]]):

    def __init__(self, pair_idx: np.ndarray, batch_pairs: int = 32,
                 shuffle: bool = True, drop_last: bool = False, seed: int = SEED):
        self.batch_pairs = batch_pairs
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        order: dict[int, list[int]] = {}
        for row, p in enumerate(pair_idx):
            order.setdefault(int(p), []).append(row)
        self.groups = list(order.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = np.arange(len(self.groups))
        if self.shuffle:
            rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_pairs):
            chunk = idx[start:start + self.batch_pairs]
            if self.drop_last and len(chunk) < self.batch_pairs:
                continue
            batch: list[int] = []
            for g in chunk:
                batch.extend(self.groups[g])
            if len(batch) > 1:
                yield batch

    def __len__(self) -> int:
        n = len(self.groups)
        return n // self.batch_pairs if self.drop_last else (n + self.batch_pairs - 1) // self.batch_pairs


def label_subsample(df: pd.DataFrame, frac: float, seed: int = SEED,
                    group: str = "patient_id") -> np.ndarray:
    if frac >= 1.0:
        return np.ones(len(df), dtype=bool)
    rng = np.random.default_rng(seed)
    pats = df[group].unique()
    keep = set(rng.choice(pats, size=max(1, int(round(frac * len(pats)))), replace=False))
    return df[group].isin(keep).to_numpy()
