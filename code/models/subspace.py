
from __future__ import annotations

import numpy as np
import pandas as pd


class ConditionalPairMeanProjection:

    def __init__(self, rank: int = 16, energy: float | None = None,
                 condition: bool = True):
        self.rank = rank
        self.energy = energy
        self.condition = condition
        self.U_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_: np.ndarray | None = None

    def fit(self, z: np.ndarray, pair_id: np.ndarray, y: np.ndarray | None = None
            ) -> "ConditionalPairMeanProjection":
        z = np.asarray(z, dtype=np.float64)
        self.mean_ = z.mean(0, keepdims=True)
        zc = z - self.mean_

        df = pd.DataFrame({"pair": np.asarray(pair_id)})
        df["row"] = np.arange(len(df))

        means, labs = [], []
        for pid, grp in df.groupby("pair"):
            rows = grp["row"].to_numpy()
            if len(rows) < 2:
                continue
            means.append(zc[rows].mean(0))
            if y is not None:
                vals = np.asarray(y, dtype=float)[rows]
                vals = vals[np.isfinite(vals)]
                labs.append(vals[0] if len(vals) and np.all(vals == vals[0]) else np.nan)
        if not means:
            raise ValueError("no complete pairs to fit the projection on")

        M = np.stack(means, 0)

        if self.condition and y is not None and labs:
            labs = np.asarray(labs, dtype=float)
            for v in np.unique(labs[np.isfinite(labs)]):
                m = labs == v
                if m.sum() >= 2:
                    M[m] -= M[m].mean(0, keepdims=True)
            unl = ~np.isfinite(labs)
            if unl.any():
                M[unl] -= M[unl].mean(0, keepdims=True)

        _, s, vt = np.linalg.svd(M, full_matrices=False)
        var = s ** 2
        self.explained_ = var / max(var.sum(), 1e-12)

        k = self.rank
        if self.energy is not None:
            k = int(np.searchsorted(np.cumsum(self.explained_), self.energy) + 1)
        k = max(1, min(k, vt.shape[0], z.shape[1]))
        self.rank_used_ = k
        self.U_ = vt[:k].T
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        if self.U_ is None:
            raise RuntimeError("call fit() first")
        zc = np.asarray(z, dtype=np.float64) - self.mean_
        return (zc - (zc @ self.U_) @ self.U_.T).astype(np.float32)

    def fit_transform(self, z, pair_id, y=None) -> np.ndarray:
        return self.fit(z, pair_id, y).transform(z)

    def __repr__(self) -> str:
        k = getattr(self, "rank_used_", self.rank)
        return (f"CPMP(rank={k}, conditional={self.condition}"
                + (f", energy={self.energy}" if self.energy else "") + ")")


class NullspaceProjection:

    def __init__(self, n_iter: int = 8, seed: int = 0):
        self.n_iter = n_iter
        self.seed = seed
        self.P_: np.ndarray | None = None

    def fit(self, z: np.ndarray, nuisance: np.ndarray) -> "NullspaceProjection":
        from sklearn.linear_model import LogisticRegression

        z = np.asarray(z, dtype=np.float64)
        d = z.shape[1]
        P = np.eye(d)
        zc = z.copy()
        for _ in range(self.n_iter):
            if np.unique(nuisance).size < 2:
                break
            clf = LogisticRegression(max_iter=500, random_state=self.seed)
            try:
                clf.fit(zc, nuisance)
            except ValueError:
                break
            W = np.atleast_2d(clf.coef_)
            q, _ = np.linalg.qr(W.T)
            Pi = np.eye(d) - q @ q.T
            P = Pi @ P
            zc = z @ P.T
        self.P_ = P
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        if self.P_ is None:
            raise RuntimeError("call fit() first")
        return (np.asarray(z, dtype=np.float64) @ self.P_.T).astype(np.float32)
