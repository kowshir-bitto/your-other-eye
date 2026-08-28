
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BACKBONES, PATHS, SEED, VARIANTS
from dataio.embeddings import EmbeddingStore, label_subsample
from experiments.probes import (attribute_probe, downstream_auc,
                                fellow_eye_retrieval, patient_id_probe)
from models.subspace import ConditionalPairMeanProjection as CPMP

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_LABEL = {"papila": "y_glaucoma", "odir": "y_abnormal",
                 "deepdrid": "y_referable", "deepdrid_fields": "y_referable"}
LABEL = "y_glaucoma"


def set_label(dataset: str, override: str = "") -> str:
    global LABEL
    LABEL = override or DEFAULT_LABEL.get(dataset, "y_abnormal")
    return LABEL


def available(dataset: str = "papila") -> list[tuple[str, str]]:
    out = []
    for b, v in itertools.product(BACKBONES, VARIANTS):
        if (PATHS.embeddings / f"{dataset}__{b}__{v}.npz").is_file():
            out.append((b, v))
    return out


def _prep(dataset, backbone, variant, split_col="split"):
    s = EmbeddingStore.load(dataset, backbone, variant).complete_pairs_only()
    tr = s.subset(s.df[split_col].to_numpy() == "train")
    te = s.subset(s.df[split_col].to_numpy() == "test")
    m, sd = tr.standardise()
    te.standardise(m, sd)
    return tr, te


def e1_leakage_audit(dataset="papila") -> pd.DataFrame:
    rows = []
    for backbone, variant in available(dataset):
        tr, te = _prep(dataset, backbone, variant)
        pair = te.pair_index()
        ret = fellow_eye_retrieval(te.feat, pair)
        pid = patient_id_probe(te.feat, te.df["patient_id"].factorize()[0])
        auc = downstream_auc(tr.feat, tr.labels(LABEL), te.feat, te.labels(LABEL),
                             groups_te=te.df["patient_id"].to_numpy())
        row = {
            "dataset": dataset, "backbone": backbone, "variant": variant,
            "dim": te.dim, "n_test": len(te.df),
            "n_test_patients": te.df["patient_id"].nunique(),
            "R@1": ret["R@1"], "R@5": ret["R@5"], "MRR": ret["MRR"],
            "chance_R@1": ret["chance_R@1"], "lift_R@1": ret["lift_R@1"],
            "pid_acc": pid["acc"], "pid_chance": pid["chance"],
            "task_auc": auc["auc"], "task_lo": auc["lo"], "task_hi": auc["hi"],
        }
        for a in ("a_age", "a_gender"):
            if a not in tr.df.columns or a not in te.df.columns:
                row[f"probe_{a}"] = np.nan
                row[f"probe_{a}_metric"] = "n/a"
                continue
            r = attribute_probe(tr.feat, tr.df[a].to_numpy(float),
                                te.feat, te.df[a].to_numpy(float))
            row[f"probe_{a}"] = r["value"]
            row[f"probe_{a}_metric"] = r["metric"]
        rows.append(row)
        print(f"  {backbone:<16} {variant:<8} R@1={row['R@1']:.3f} "
              f"({row['lift_R@1']:.0f}x)  AUC={row['task_auc']:.3f}")
    return pd.DataFrame(rows)


def e2_split_protocol(dataset="papila", n_seeds=10) -> pd.DataFrame:
    rows = []
    for backbone, variant in available(dataset):
        for split_col in ("split", "split_image"):
            for seed in range(n_seeds):
                s = EmbeddingStore.load(dataset, backbone, variant).complete_pairs_only()
                from dataio.build_manifest import assign_splits
                strat = next((c for c in ("y_nonhealthy", "y_abnormal", LABEL)
                              if c in s.df.columns), LABEL)
                s.df = assign_splits(s.df.drop(columns=["split", "split_image"]),
                                     strat, seed=SEED + seed)
                tr = s.subset(s.df[split_col].to_numpy() == "train")
                te = s.subset(s.df[split_col].to_numpy() == "test")
                m, sd = tr.standardise()
                te.standardise(m, sd)
                auc = downstream_auc(tr.feat, tr.labels(LABEL), te.feat, te.labels(LABEL),
                                     groups_te=te.df["patient_id"].to_numpy(), n_boot=0)
                tr_pat = set(tr.df["patient_id"])
                leaked = te.df["patient_id"].isin(tr_pat).mean()
                rows.append({
                    "dataset": dataset, "backbone": backbone, "variant": variant,
                    "protocol": "image-level" if split_col == "split_image" else "patient-level",
                    "seed": seed, "auc": auc["auc"],
                    "frac_test_with_fellow_in_train": float(leaked),
                })
        sub = pd.DataFrame(rows)
        sub = sub[(sub.backbone == backbone) & (sub.variant == variant)]
        g = sub.groupby("protocol")["auc"].mean()
        if len(g) == 2:
            print(f"  {backbone:<16} {variant:<8} "
                  f"patient={g.get('patient-level', np.nan):.3f}  "
                  f"image={g.get('image-level', np.nan):.3f}  "
                  f"gap={g.get('image-level', np.nan)-g.get('patient-level', np.nan):+.3f}")
    return pd.DataFrame(rows)


def e3_mitigation(dataset="papila", backbone="dinov2_vitb14", variant="canon") -> pd.DataFrame:
    tr, te = _prep(dataset, backbone, variant)
    ytr, yte = tr.labels(LABEL), te.labels(LABEL)
    g = te.df["patient_id"].to_numpy()
    pair_te = te.pair_index()
    n_pairs_tr = tr.df["pair_id"].nunique()

    def score(name, ztr, zte, extra=None):
        r = fellow_eye_retrieval(zte, pair_te)
        a = downstream_auc(ztr, ytr, zte, yte, groups_te=g, n_boot=300)
        row = {"method": name, "R@1": r["R@1"], "lift_R@1": r["lift_R@1"],
               "MRR": r["MRR"], "auc": a["auc"], "lo": a["lo"], "hi": a["hi"],
               "dim": zte.shape[1]}
        for at in ("a_age", "a_gender"):
            p = attribute_probe(ztr, tr.df[at].to_numpy(float),
                                zte, te.df[at].to_numpy(float))
            row[f"probe_{at}"] = p["value"]
        if extra:
            row.update(extra)
        return row

    rows = [score("none (frozen)", tr.feat, te.feat, {"rank": 0, "conditional": None})]

    ranks = [r for r in (4, 8, 16, 32, 64, 96, 128) if r <= n_pairs_tr]
    for cond in (True, False):
        for rank in ranks:
            p = CPMP(rank=rank, condition=cond).fit(tr.feat, tr.pair_index(), ytr)
            rows.append(score(f"CPMP-{'cond' if cond else 'uncond'} r={rank}",
                              p.transform(tr.feat), p.transform(te.feat),
                              {"rank": rank, "conditional": cond,
                               "family": f"CPMP-{'cond' if cond else 'uncond'}"}))
            print(f"  CPMP {'cond  ' if cond else 'uncond'} r={rank:<4} "
                  f"R@1={rows[-1]['R@1']:.3f}  AUC={rows[-1]['auc']:.3f}")

    df = pd.DataFrame(rows)
    df["max_estimable_rank"] = n_pairs_tr
    return df


def e4_dimensionality(dataset="papila", backbone="dinov2_vitb14",
                      variant="canon") -> pd.DataFrame:
    tr, te = _prep(dataset, backbone, variant)
    ytr, yte = tr.labels(LABEL), te.labels(LABEL)
    g = te.df["patient_id"].to_numpy()
    pair_te = te.pair_index()
    n_pairs_tr = tr.df["pair_id"].nunique()

    rows = []
    for rank in [r for r in (0, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 144) if r <= n_pairs_tr]:
        if rank == 0:
            ztr, zte = tr.feat, te.feat
        else:
            p = CPMP(rank=rank, condition=True).fit(tr.feat, tr.pair_index(), ytr)
            ztr, zte = p.transform(tr.feat), p.transform(te.feat)
        r = fellow_eye_retrieval(zte, pair_te)
        a = downstream_auc(ztr, ytr, zte, yte, groups_te=g, n_boot=0)
        rows.append({"sweep": "remove_top_k_nuisance", "k": rank,
                     "R@1": r["R@1"], "chance": r["chance_R@1"], "auc": a["auc"],
                     "max_estimable_rank": n_pairs_tr})

    from sklearn.decomposition import PCA
    for k in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        if k >= min(tr.feat.shape):
            continue
        pca = PCA(n_components=k, random_state=SEED).fit(tr.feat)
        ztr, zte = pca.transform(tr.feat), pca.transform(te.feat)
        r = fellow_eye_retrieval(zte, pair_te)
        a = downstream_auc(ztr, ytr, zte, yte, groups_te=g, n_boot=0)
        rows.append({"sweep": "keep_top_k_pca", "k": k,
                     "R@1": r["R@1"], "chance": r["chance_R@1"], "auc": a["auc"],
                     "max_estimable_rank": n_pairs_tr})
    return pd.DataFrame(rows)


def e5_low_label(dataset="papila", backbone="dinov2_vitb14", variant="canon",
                 fracs=(0.05, 0.1, 0.25, 0.5, 1.0), n_seeds=10) -> pd.DataFrame:
    tr, te = _prep(dataset, backbone, variant)
    yte = te.labels(LABEL)
    g = te.df["patient_id"].to_numpy()
    rows = []
    for frac in fracs:
        for seed in range(n_seeds):
            keep = label_subsample(tr.df, frac, seed=SEED + seed)
            ytr = tr.labels(LABEL)
            if np.unique(ytr[keep]).size < 2:
                continue
            a = downstream_auc(tr.feat[keep], ytr[keep], te.feat, yte,
                               groups_te=g, n_boot=0)
            rows.append({"label_frac": frac, "seed": seed, "auc": a["auc"],
                         "n_labelled_patients": int(tr.df[keep]["patient_id"].nunique()),
                         "n_labelled_images": int(keep.sum())})
    df = pd.DataFrame(rows)
    if not df.empty:
        for f, grp in df.groupby("label_frac"):
            print(f"  frac={f:<5} n_pat={grp.n_labelled_patients.iloc[0]:>4}  "
                  f"AUC={grp.auc.mean():.3f} +/- {grp.auc.std():.3f}")
    return df


def e9_same_eye_control(dataset="deepdrid_fields", backbone="dinov2_vitb14",
                        variant="canon") -> pd.DataFrame:
    import cv2
    from dataio.preprocess import load_fundus_rgb

    store = EmbeddingStore.load(dataset, backbone, variant)
    te = store.subset(store.df["split"].to_numpy() == "test")
    df = te.df.reset_index(drop=True)
    if "field" not in df.columns:
        raise ValueError("needs the both-fields manifest")

    eye_key = (df["patient_id"] + "_" + df["eye"]).factorize()[0]
    pat_key = df["patient_id"].factorize()[0]

    n_same_eye = int(pd.Series(eye_key).value_counts().gt(1).sum())
    print(f"  test images={len(df)}  eyes with 2 fields={n_same_eye}  "
          f"patients={df['patient_id'].nunique()}")

    fld = df["field"].to_numpy()
    same_eye_m = eye_key[:, None] == eye_key[None, :]
    same_pat_m = pat_key[:, None] == pat_key[None, :]
    same_fld_m = fld[:, None] == fld[None, :]

    CONDITIONS = {
        "same eye, diff field": lambda: same_eye_m & ~same_fld_m,
        "fellow eye, same field": lambda: same_pat_m & ~same_eye_m & same_fld_m,
        "fellow eye, diff field": lambda: same_pat_m & ~same_eye_m & ~same_fld_m,
    }

    def retrieve(z, pos_mask):
        zz = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        sim = zz @ zz.T
        np.fill_diagonal(sim, -np.inf)
        pos = pos_mask.copy()
        np.fill_diagonal(pos, False)
        return _rank_metrics(sim, pos)

    paths = [PATHS.raw / p for p in df["image_path"]]
    eyes = df["eye"].tolist()
    reps = {f"{backbone} (learned)": te.feat}

    rows_px = []
    for p, e in zip(paths, eyes):
        rows_px.append(load_fundus_rgb(p, e, size=32).astype(np.float32).ravel() / 255.0)
    reps["raw pixels 32x32"] = np.stack(rows_px)

    rows_h = []
    for p, e in zip(paths, eyes):
        im = load_fundus_rgb(p, e, size=224)
        h = np.concatenate([np.histogram(im[..., c], bins=32, range=(0, 255))[0]
                            for c in range(3)]).astype(np.float32)
        rows_h.append(h / max(h.sum(), 1))
    reps["colour histogram"] = np.stack(rows_h)

    out = []
    for name, z in reps.items():
        res = {c: retrieve(z, fn()) for c, fn in CONDITIONS.items()}
        row = {"representation": name, "dim": int(z.shape[1]),
               "n_test": len(df), "dataset": dataset, "variant": variant}
        for c, r in res.items():
            key = c.replace(", ", "_").replace(" ", "_")
            row[f"{key}_R@1"] = r["R@1"]
            row[f"{key}_MRR"] = r["MRR"]
        se = res["same eye, diff field"]["R@1"]
        fe = res["fellow eye, diff field"]["R@1"]
        row["ratio_sameeye_over_fellow_matched_field"] = (se / fe) if fe else np.nan
        out.append(row)
        print(f"  {name}")
        for c, r in res.items():
            print(f"      {c:<26} R@1={r['R@1']:.3f}  MRR={r['MRR']:.3f}  n={r['n']}")
        print(f"      {'-> same/fellow (field matched)':<26} "
              f"{row['ratio_sameeye_over_fellow_matched_field']:.2f}x")
    return pd.DataFrame(out)


def _rank_metrics(sim: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    has = pos.any(axis=1)
    if not has.any():
        return {"R@1": float("nan"), "MRR": float("nan"), "n": 0}
    order = np.argsort(-sim, axis=1)
    ranks = np.full(len(sim), np.inf)
    for i in np.flatnonzero(has):
        hit = np.flatnonzero(pos[i][order[i]])
        if hit.size:
            ranks[i] = hit[0] + 1
    ok = np.isfinite(ranks)
    return {"R@1": float((ranks[ok] <= 1).mean()),
            "MRR": float((1.0 / ranks[ok]).mean()), "n": int(ok.sum())}


def e8_asymmetry(dataset="odir", backbone="dinov2_vitb14", variant="canon",
                 n_seeds=5) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    store = EmbeddingStore.load(dataset, backbone, variant).complete_pairs_only()
    df = store.df
    if LABEL not in df.columns:
        raise ValueError(f"{LABEL} not in manifest")

    wide = df.pivot_table(index="pair_id", columns="eye", values=LABEL, aggfunc="first")
    if not {"OD", "OS"}.issubset(wide.columns):
        raise ValueError("need both eyes per pair")
    wide = wide.dropna()
    disc = wide[wide["OD"] != wide["OS"]]
    print(f"  discordant pairs: {len(disc)} of {len(wide)} "
          f"({100*len(disc)/max(len(wide),1):.1f}%)")
    if len(disc) < 40:
        raise ValueError(f"only {len(disc)} discordant pairs; not enough")

    row_of = {(p, e): i for i, (p, e) in
              enumerate(zip(df["pair_id"], df["eye"]))}
    split_of = dict(zip(df["pair_id"], df["split"]))

    pairs, targets, splits = [], [], []
    for pid, r in disc.iterrows():
        i_od, i_os = row_of.get((pid, "OD")), row_of.get((pid, "OS"))
        if i_od is None or i_os is None:
            continue
        pairs.append((i_od, i_os))
        targets.append(1 if r["OD"] > r["OS"] else 0)
        splits.append(split_of.get(pid, "train"))
    pairs = np.array(pairs)
    y = np.array(targets)
    splits = np.array(splits)

    tr_m = splits == "train"
    te_m = splits == "test"
    if te_m.sum() < 20:
        tr_m = np.isin(splits, ["train", "val"])
    print(f"  train pairs={int(tr_m.sum())}  test pairs={int(te_m.sum())}")

    feat = store.feat.copy()
    tr_rows = np.unique(pairs[tr_m].ravel())
    mu = feat[tr_rows].mean(0, keepdims=True)
    sd = feat[tr_rows].std(0, keepdims=True)
    sd = np.where(sd > 1e-3 * np.median(sd), sd, np.inf)
    feat = np.clip((feat - mu) / sd, -8, 8)

    def symmetrise(idx, yy):
        a = feat[pairs[idx][:, 0]]
        b = feat[pairs[idx][:, 1]]
        return (np.concatenate([a, b]), np.concatenate([b, a]),
                np.concatenate([yy[idx], 1 - yy[idx]]))

    rows = []
    tr_idx_all = np.flatnonzero(tr_m)
    te_idx_all = np.flatnonzero(te_m)
    for seed in range(n_seeds):
        rng = np.random.default_rng(SEED + seed)
        boot = (tr_idx_all if seed == 0
                else rng.choice(tr_idx_all, size=len(tr_idx_all), replace=True))
        A_tr, B_tr, y_tr = symmetrise(boot, y)
        A_te, B_te, y_te = symmetrise(te_idx_all, y)

        yy = pd.to_numeric(df[LABEL], errors="coerce").to_numpy()
        ok = np.flatnonzero((df["split"] == "train").to_numpy() & np.isfinite(yy))
        ok_b = ok if seed == 0 else rng.choice(ok, size=len(ok), replace=True)
        clf_eye = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=seed).fit(feat[ok_b], yy[ok_b])
        s_a = clf_eye.decision_function(A_te)
        s_b = clf_eye.decision_function(B_te)
        pred = (s_a > s_b).astype(int)
        rows.append({"method": "per-eye (all training images)", "seed": seed,
                     "acc": float((pred == y_te).mean()),
                     "auc": float(roc_auc_score(y_te, s_a - s_b)),
                     "n_test": len(y_te)})

        eyes_b = np.unique(pairs[boot].ravel())
        yb = yy[eyes_b]
        if np.unique(yb[np.isfinite(yb)]).size > 1:
            ok2 = eyes_b[np.isfinite(yb)]
            clf_m = LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed).fit(feat[ok2], yy[ok2])
            m_a = clf_m.decision_function(A_te)
            m_b = clf_m.decision_function(B_te)
            rows.append({"method": "per-eye (matched data)", "seed": seed,
                         "acc": float(((m_a > m_b).astype(int) == y_te).mean()),
                         "auc": float(roc_auc_score(y_te, m_a - m_b)),
                         "n_test": len(y_te)})

        for name, Xtr, Xte in (
                ("difference  z(A) - z(B)", A_tr - B_tr, A_te - B_te),
                ("concatenation [z(A); z(B)]",
                 np.hstack([A_tr, B_tr]), np.hstack([A_te, B_te]))):
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, C=0.1,
                                     random_state=seed).fit(sc.transform(Xtr), y_tr)
            p = clf.predict_proba(sc.transform(Xte))[:, 1]
            rows.append({"method": name, "seed": seed,
                         "acc": float(((p > 0.5).astype(int) == y_te).mean()),
                         "auc": float(roc_auc_score(y_te, p)),
                         "n_test": len(y_te)})

    out = pd.DataFrame(rows)
    out["dataset"] = dataset
    out["label"] = LABEL
    out["n_discordant"] = len(disc)
    for m, g in out.groupby("method"):
        print(f"  {m:<28} acc={g.acc.mean():.3f}+/-{g.acc.std():.3f}  "
              f"auc={g.auc.mean():.3f}")
    return out


def e7_cv_split(dataset="papila", backbone="dinov2_vitb14", variant="canon",
                n_folds=5, n_repeats=10) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    store = EmbeddingStore.load(dataset, backbone, variant).complete_pairs_only()
    X = store.feat
    y = store.labels(LABEL)
    groups = store.df["patient_id"].to_numpy()
    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], groups[ok]

    rows = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(SEED + rep)
        perm = rng.permutation(len(X))
        Xp, yp, gp = X[perm], y[perm], groups[perm]

        for protocol, splitter in (
                ("patient-level", StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                                       random_state=SEED + rep)),
                ("image-level", KFold(n_splits=n_folds, shuffle=True,
                                      random_state=SEED + rep))):
            oof = np.full(len(yp), np.nan)
            gen = (splitter.split(Xp, yp.astype(int), gp) if protocol == "patient-level"
                   else splitter.split(Xp))
            for tr_i, te_i in gen:
                if np.unique(yp[tr_i]).size < 2:
                    continue
                sc = StandardScaler().fit(Xp[tr_i])
                clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=SEED).fit(sc.transform(Xp[tr_i]),
                                                                yp[tr_i])
                oof[te_i] = clf.predict_proba(sc.transform(Xp[te_i]))[:, 1]

            m = np.isfinite(oof)
            if m.sum() and np.unique(yp[m]).size > 1:
                rows.append({"dataset": dataset, "backbone": backbone,
                             "variant": variant, "protocol": protocol,
                             "repeat": rep, "n_folds": n_folds,
                             "auc": float(roc_auc_score(yp[m], oof[m])),
                             "n_scored": int(m.sum())})

    df = pd.DataFrame(rows)
    if not df.empty:
        g = df.groupby("protocol").auc.agg(["mean", "std"])
        for p, r in g.iterrows():
            print(f"  {p:<14} AUC={r['mean']:.4f} +/- {r['std']:.4f}")
        if {"patient-level", "image-level"}.issubset(g.index):
            gap = g.loc["image-level", "mean"] - g.loc["patient-level", "mean"]
            piv = df.pivot_table(index="repeat", columns="protocol", values="auc")
            d = piv["image-level"] - piv["patient-level"]
            print(f"  INFLATION = {gap:+.4f}  "
                  f"(paired mean {d.mean():+.4f} +/- {d.std():.4f}, "
                  f"{int((d > 0).sum())}/{len(d)} repeats positive)")
    return df


def e6_pixel_baseline(dataset="papila", backbone="dinov2_vitb14",
                      variant="canon") -> pd.DataFrame:
    import cv2
    from dataio.preprocess import load_fundus_rgb

    store = EmbeddingStore.load(dataset, backbone, variant).complete_pairs_only()
    is_te = store.df["split"].to_numpy() == "test"
    pair_te = store.subset(is_te).pair_index()

    paths = [PATHS.raw / p for p in store.df["image_path"]]
    eyes = store.df["eye"].tolist()

    def standardise(z):
        tr = z[~is_te]
        mu = tr.mean(0, keepdims=True)
        sd = tr.std(0, keepdims=True)
        sd = np.where(sd > 1e-3 * max(np.median(sd), 1e-12), sd, np.inf)
        return np.clip((z - mu) / sd, -8, 8)[is_te]

    acc: dict[str, list] = {f"raw pixels {s}x{s}": [] for s in (16, 32, 64)}
    acc["raw pixels 32x32 gray"] = []
    acc["colour histogram"] = []

    t0 = time.time()
    for i, (p, e) in enumerate(zip(paths, eyes)):
        img = load_fundus_rgb(p, e, size=224)
        for s in (16, 32, 64):
            small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
            acc[f"raw pixels {s}x{s}"].append(
                small.astype(np.float32).ravel() / 255.0)
        g32 = cv2.cvtColor(cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_RGB2GRAY)
        acc["raw pixels 32x32 gray"].append(g32.astype(np.float32).ravel() / 255.0)
        h = np.concatenate([np.histogram(img[..., c], bins=32, range=(0, 255))[0]
                            for c in range(3)]).astype(np.float32)
        acc["colour histogram"].append(h / max(h.sum(), 1))
        if (i + 1) % 500 == 0:
            print(f"\r  [pixels] {i+1}/{len(paths)}  "
                  f"{(i+1)/(time.time()-t0):.0f} img/s", end="", flush=True)
    if len(paths) >= 500:
        print()

    reps: dict[str, np.ndarray] = {k: np.stack(v) for k, v in acc.items()}
    reps[f"{backbone} (learned)"] = store.feat

    rows_out = []
    for name, z in reps.items():
        r = fellow_eye_retrieval(standardise(z), pair_te)
        rows_out.append({
            "representation": name, "dim": int(z.shape[1]),
            "R@1": r["R@1"], "R@5": r["R@5"], "MRR": r["MRR"],
            "chance_R@1": r["chance_R@1"], "lift_R@1": r["lift_R@1"],
            "n_test": int(is_te.sum()), "dataset": dataset, "variant": variant,
        })
        print(f"  {name:<28} R@1={r['R@1']:.3f}  ({r['lift_R@1']:5.1f}x)  MRR={r['MRR']:.3f}")
    return pd.DataFrame(rows_out)


EXPERIMENTS = {
    "E1": ("leakage_audit", e1_leakage_audit),
    "E2": ("split_protocol", e2_split_protocol),
    "E3": ("mitigation", e3_mitigation),
    "E4": ("dimensionality", e4_dimensionality),
    "E5": ("low_label", e5_low_label),
    "E6": ("pixel_baseline", e6_pixel_baseline),
    "E7": ("cv_split", e7_cv_split),
    "E8": ("asymmetry", e8_asymmetry),
    "E9": ("same_eye_control", e9_same_eye_control),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["all"],
                    choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--dataset", default="papila")
    ap.add_argument("--backbone", default="dinov2_vitb14", choices=list(BACKBONES),
                    help="applies to E3-E9; E1 always sweeps every backbone present")
    ap.add_argument("--label", default="", help="override the per-dataset default task")
    ap.add_argument("--suffix", default="", help="appended to output filenames")
    args = ap.parse_args()

    todo = list(EXPERIMENTS) if "all" in args.exp else args.exp
    PATHS.results.mkdir(parents=True, exist_ok=True)

    avail = available(args.dataset)
    if not avail:
        print(f"!! no embeddings for {args.dataset}; run extract_embeddings.py")
        return 1
    lab = set_label(args.dataset, args.label)
    print(f"dataset={args.dataset}  task={lab}")
    print(f"available (backbone, variant): {avail}\n")

    for key in todo:
        name, fn = EXPERIMENTS[key]
        print(f"[{key}] {name}")
        kw = {"dataset": args.dataset}
        if key not in ("E1", "E2"):
            kw["backbone"] = args.backbone
        try:
            df = fn(**kw)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"  !! {type(exc).__name__}: {exc}")
            continue
        sfx = args.suffix or ("" if args.dataset == "papila" else f"_{args.dataset}")
        if args.backbone != "dinov2_vitb14" and key not in ("E1", "E2"):
            sfx += f"_{args.backbone}"
        out = PATHS.results / f"{key.lower()}_{name}{sfx}.csv"
        df.to_csv(out, index=False)
        print(f"  -> {out}  ({len(df)} rows)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
