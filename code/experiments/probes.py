
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-1, 5, 25)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def bootstrap_ci(fn, *arrays, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0, groups: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = []

    try:
        point = float(fn(*arrays))
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    if n_boot <= 0:
        return point, float("nan"), float("nan")

    if groups is not None:
        uniq = np.unique(groups)
        index_of = {g: np.flatnonzero(groups == g) for g in uniq}

    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            picked = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([index_of[g] for g in picked])
        try:
            stats.append(fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue

    if not stats:
        return point, float("nan"), float("nan")
    stats = np.asarray(stats, dtype=float)
    return point, float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def fellow_eye_retrieval(z: np.ndarray, pair_idx: np.ndarray,
                         ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    z = _l2(z.astype(np.float64))
    n = len(z)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)

    same = pair_idx[:, None] == pair_idx[None, :]
    np.fill_diagonal(same, False)
    has_fellow = same.any(axis=1)
    if not has_fellow.any():
        return {"n": 0}

    order = np.argsort(-sim, axis=1)
    hit_rank = np.full(n, np.inf)
    for i in np.flatnonzero(has_fellow):
        ranked_same = same[i][order[i]]
        pos = np.flatnonzero(ranked_same)
        if pos.size:
            hit_rank[i] = pos[0] + 1

    valid = np.isfinite(hit_rank) & has_fellow
    out: dict[str, float] = {"n": int(valid.sum())}
    for k in ks:
        out[f"R@{k}"] = float((hit_rank[valid] <= k).mean())
    out["MRR"] = float((1.0 / hit_rank[valid]).mean())
    for k in ks:
        out[f"chance_R@{k}"] = float(min(1.0, k / (n - 1)))
    out["lift_R@1"] = (out["R@1"] / out["chance_R@1"]) if out["chance_R@1"] > 0 else float("nan")
    return out


def attribute_probe(z_tr: np.ndarray, a_tr: np.ndarray,
                    z_te: np.ndarray, a_te: np.ndarray,
                    kind: str = "auto", seed: int = 0) -> dict[str, float]:
    m_tr = np.isfinite(a_tr)
    m_te = np.isfinite(a_te)
    if m_tr.sum() < 10 or m_te.sum() < 10:
        return {"metric": "n/a", "value": float("nan"), "n_test": int(m_te.sum())}

    z_tr, a_tr = z_tr[m_tr], a_tr[m_tr]
    z_te, a_te = z_te[m_te], a_te[m_te]

    if kind == "auto":
        kind = "binary" if np.unique(a_tr).size <= 2 else "continuous"

    sc = StandardScaler().fit(z_tr)
    Xtr, Xte = sc.transform(z_tr), sc.transform(z_te)

    if kind == "binary":
        if np.unique(a_tr).size < 2 or np.unique(a_te).size < 2:
            return {"metric": "auc", "value": float("nan"), "n_test": len(a_te)}
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xtr, a_tr)
        p = clf.predict_proba(Xte)[:, 1]
        return {"metric": "auc", "value": float(roc_auc_score(a_te, p)),
                "chance": 0.5, "n_test": len(a_te)}

    reg = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xtr, a_tr)
    return {"metric": "r2", "value": float(r2_score(a_te, reg.predict(Xte))),
            "chance": 0.0, "n_test": len(a_te), "alpha": float(reg.alpha_)}


def patient_id_probe(z: np.ndarray, patient_idx: np.ndarray,
                     seed: int = 0) -> dict[str, float]:
    uniq, counts = np.unique(patient_idx, return_counts=True)
    keep = set(uniq[counts == 2].tolist())
    mask = np.array([p in keep for p in patient_idx])
    if mask.sum() < 20:
        return {"acc": float("nan"), "chance": float("nan"), "n": int(mask.sum())}

    z, pid = z[mask], patient_idx[mask]
    tr_rows, te_rows = [], []
    seen: set[int] = set()
    for i, p in enumerate(pid):
        (tr_rows if p not in seen else te_rows).append(i)
        seen.add(int(p))

    sc = StandardScaler().fit(z[tr_rows])
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                             random_state=seed).fit(sc.transform(z[tr_rows]), pid[tr_rows])
    acc = float(clf.score(sc.transform(z[te_rows]), pid[te_rows]))
    n_cls = len(np.unique(pid))
    return {"acc": acc, "chance": 1.0 / n_cls, "lift": acc * n_cls,
            "n_patients": n_cls, "n": int(mask.sum())}


def downstream_auc(z_tr: np.ndarray, y_tr: np.ndarray,
                   z_te: np.ndarray, y_te: np.ndarray,
                   groups_te: np.ndarray | None = None,
                   C: float = 1.0, seed: int = 0, n_boot: int = 500) -> dict[str, float]:
    m_tr, m_te = np.isfinite(y_tr), np.isfinite(y_te)
    z_tr, y_tr, z_te, y_te = z_tr[m_tr], y_tr[m_tr], z_te[m_te], y_te[m_te]
    if groups_te is not None:
        groups_te = groups_te[m_te]
    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_test": len(y_te)}

    sc = StandardScaler().fit(z_tr)
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced",
                             random_state=seed).fit(sc.transform(z_tr), y_tr)
    p = clf.predict_proba(sc.transform(z_te))[:, 1]

    auc, lo, hi = bootstrap_ci(roc_auc_score, y_te, p, n_boot=n_boot,
                               seed=seed, groups=groups_te)
    return {"auc": auc, "lo": lo, "hi": hi, "n_test": int(len(y_te)),
            "pos_rate": float(np.mean(y_te))}


def audit_representation(name: str,
                         z_tr: np.ndarray, z_te: np.ndarray,
                         df_tr, df_te,
                         label_col: str,
                         attrs: tuple[str, ...] = ("a_age", "a_gender"),
                         seed: int = 0) -> dict:
    pair_te = df_te["pair_id"].factorize()[0]
    pid_te = df_te["patient_id"].factorize()[0]

    row: dict = {"representation": name, "dim": int(z_te.shape[1])}

    ret = fellow_eye_retrieval(z_te, pair_te)
    row.update({f"ret_{k}": v for k, v in ret.items()})

    pid = patient_id_probe(z_te, pid_te, seed=seed)
    row.update({f"pid_{k}": v for k, v in pid.items()})

    for a in attrs:
        if a in df_tr.columns:
            r = attribute_probe(z_tr, df_tr[a].to_numpy(dtype=float),
                                z_te, df_te[a].to_numpy(dtype=float), seed=seed)
            row[f"attr_{a}"] = r["value"]
            row[f"attr_{a}_metric"] = r["metric"]

    d = downstream_auc(z_tr, df_tr[label_col].to_numpy(dtype=float),
                       z_te, df_te[label_col].to_numpy(dtype=float),
                       groups_te=df_te["patient_id"].to_numpy(), seed=seed)
    row.update({"task_auc": d["auc"], "task_lo": d["lo"], "task_hi": d["hi"]})
    return row
