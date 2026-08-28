
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BACKBONES, DEFAULT_BACKBONE, PATHS, SEED, TORCH_THREADS
from dataio.embeddings import EmbeddingStore, PairBatchSampler, label_subsample
from experiments.probes import audit_representation
from models.bilateral import BilateralAdapter, BilateralLoss, MLPBaseline


class TensorRows(Dataset):

    def __init__(self, x, y, pair_idx, patient_idx, labelled):
        self.x, self.y = x, y
        self.pair_idx, self.patient_idx = pair_idx, patient_idx
        self.labelled = labelled

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {
            "x": self.x[i], "y": self.y[i],
            "pair_idx": self.pair_idx[i], "patient_idx": self.patient_idx[i],
            "labelled": self.labelled[i],
        }


def collate(rows):
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def grl_schedule(step: int, total: int, max_lambda: float = 1.0) -> float:
    p = step / max(total, 1)
    return max_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


@torch.no_grad()
def encode_all(model, x: torch.Tensor, batch: int = 512) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {}
    for i in range(0, len(x), batch):
        o = model(x[i:i + batch]) if not isinstance(model, BilateralAdapter) \
            else model(x[i:i + batch], grl_lambda=0.0)
        for k in ("z_pat", "z_path", "logits"):
            if k in o:
                outs.setdefault(k, []).append(o[k].numpy())
    return {k: np.concatenate(v, 0) for k, v in outs.items()}


def run_once(args, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    store = EmbeddingStore.load(args.dataset, args.backbone, args.variant).complete_pairs_only()
    split_col = "split_image" if args.leaky_split else "split"

    tr = store.subset(store.df[split_col].to_numpy() == "train")
    va = store.subset(store.df[split_col].to_numpy() == "val")
    te = store.subset(store.df[split_col].to_numpy() == "test")

    mean, std = tr.standardise()
    va.standardise(mean, std)
    te.standardise(mean, std)

    labels = args.label if isinstance(args.label, list) else [args.label]

    def tensors(s: EmbeddingStore):
        y = np.stack([s.labels(c) for c in labels], 1)
        return (torch.from_numpy(s.feat),
                torch.from_numpy(y),
                torch.from_numpy(s.pair_index()),
                torch.from_numpy(s.patient_index()))

    x_tr, y_tr, pair_tr, pid_tr = tensors(tr)
    x_va, y_va, _, _ = tensors(va)
    x_te, y_te, _, _ = tensors(te)

    keep = label_subsample(tr.df, args.label_frac, seed=seed)
    labelled = torch.from_numpy(keep.astype(np.float32))
    y_tr_masked = y_tr.clone()
    y_tr_masked[~torch.from_numpy(keep)] = float("nan")

    n_patients = int(pid_tr.max().item()) + 1
    in_dim = tr.dim

    if args.model == "mlp":
        model = MLPBaseline(in_dim, n_classes=len(labels), hidden=args.hidden,
                            d_out=args.d_path, dropout=args.dropout)
        crit = None
    else:
        model = BilateralAdapter(
            in_dim, d_pat=args.d_pat, d_path=args.d_path, n_classes=len(labels),
            n_patients=n_patients if args.w_advid > 0 else 0,
            hidden=args.hidden, dropout=args.dropout)
        crit = BilateralLoss(w_cls=args.w_cls, w_nce=args.w_nce, w_adv=args.w_adv,
                             w_dec=args.w_dec, w_rec=args.w_rec, w_advid=args.w_advid,
                             temperature=args.temperature, multilabel=True,
                             adv_mode=args.adv_mode)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ds = TensorRows(x_tr, y_tr_masked, pair_tr, pid_tr, labelled)
    sampler = PairBatchSampler(tr.pair_index(), batch_pairs=args.batch_pairs, seed=seed)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)
    total_steps = args.epochs * max(len(sampler), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best = {"val": -np.inf, "state": None, "epoch": -1}
    hist = []
    step = 0
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        sampler.set_epoch(ep)
        agg: dict[str, float] = {}
        for batch in dl:
            lam = grl_schedule(step, total_steps, args.grl_max)
            if args.model == "mlp":
                out = model(batch["x"])
                m = ~torch.isnan(batch["y"])
                loss = (F.binary_cross_entropy_with_logits(out["logits"][m], batch["y"][m])
                        if m.any() else out["logits"].sum() * 0.0)
                parts = {"cls": float(loss)}
            else:
                out = model(batch["x"], grl_lambda=lam)
                loss, parts = crit(out, batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v

        enc = encode_all(model, x_va)
        from sklearn.metrics import roc_auc_score
        yv = y_va.numpy()[:, 0]
        pv = enc["logits"][:, 0]
        m = np.isfinite(yv)
        val_auc = (float(roc_auc_score(yv[m], pv[m]))
                   if m.sum() and np.unique(yv[m]).size > 1 else float("nan"))

        n_b = max(len(sampler), 1)
        rec = {"epoch": ep, "val_auc": val_auc, "grl": lam,
               **{f"loss_{k}": v / n_b for k, v in agg.items()}}
        hist.append(rec)
        if np.isfinite(val_auc) and val_auc > best["val"]:
            best = {"val": val_auc, "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if args.verbose and (ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1):
            losses = "  ".join(f"{k}={v/n_b:.3f}" for k, v in agg.items())
            print(f"    ep {ep:>3}  val_auc {val_auc:.4f}  grl {lam:.2f}  {losses}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    train_time = time.time() - t0

    enc_tr = encode_all(model, x_tr)
    enc_te = encode_all(model, x_te)

    rows = []
    reps = {"frozen": (tr.feat, te.feat)}
    if "z_path" in enc_te:
        reps["z_path"] = (enc_tr["z_path"], enc_te["z_path"])
    if "z_pat" in enc_te:
        reps["z_pat"] = (enc_tr["z_pat"], enc_te["z_pat"])
    for name, (ztr, zte) in reps.items():
        r = audit_representation(name, ztr, zte, tr.df, te.df, labels[0], seed=seed)
        r.update({"seed": seed, "model": args.model, "tag": args.tag,
                  "label_frac": args.label_frac, "split": split_col,
                  "backbone": args.backbone, "variant": args.variant,
                  "dataset": args.dataset, "label": labels[0],
                  "best_epoch": best["epoch"], "best_val_auc": best["val"],
                  "train_s": round(train_time, 2)})
        rows.append(r)

    from sklearn.metrics import roc_auc_score
    yt = y_te.numpy()[:, 0]
    pt = enc_te["logits"][:, 0]
    m = np.isfinite(yt)
    head_auc = (float(roc_auc_score(yt[m], pt[m]))
                if m.sum() and np.unique(yt[m]).size > 1 else float("nan"))
    for r in rows:
        r["head_auc"] = head_auc

    return {"rows": rows, "history": hist, "state": model.state_dict()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="papila")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=list(BACKBONES))
    ap.add_argument("--variant", default="canon")
    ap.add_argument("--label", default="y_glaucoma")
    ap.add_argument("--model", default="bilateral", choices=["bilateral", "mlp"])

    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--w-adv", type=float, default=1.0)
    ap.add_argument("--w-dec", type=float, default=0.1)
    ap.add_argument("--w-rec", type=float, default=0.1)
    ap.add_argument("--w-advid", type=float, default=0.0,
                    help="discrete patient-ID adversary; ablation only")
    ap.add_argument("--adv-mode", default="conditional", choices=["conditional", "uncond"])
    ap.add_argument("--grl-max", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.1)

    ap.add_argument("--d-pat", type=int, default=128)
    ap.add_argument("--d-path", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-pairs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--label-frac", type=float, default=1.0)
    ap.add_argument("--leaky-split", action="store_true",
                    help="use the image-level split to measure fellow-eye leakage")
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--threads", type=int, default=TORCH_THREADS)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tag = args.tag or f"{args.model}_{args.label}_lf{args.label_frac}"
    print(f"train_adapter | {args.dataset} | {args.backbone}/{args.variant} | "
          f"{args.model} | label={args.label} | frac={args.label_frac} | "
          f"split={'image(leaky)' if args.leaky_split else 'patient'}")

    all_rows, all_hist = [], []
    for s in args.seeds:
        print(f"  seed {s}")
        res = run_once(args, s)
        all_rows.extend(res["rows"])
        for h in res["history"]:
            h["seed"] = s
        all_hist.extend(res["history"])
        ck = PATHS.checkpoints / f"{tag}_seed{s}.pt"
        torch.save(res["state"], ck)

    df = pd.DataFrame(all_rows)
    out = Path(args.out) if args.out else PATHS.results / f"{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.DataFrame(all_hist).to_csv(PATHS.logs / f"{tag}_history.csv", index=False)

    cols = ["representation", "ret_R@1", "ret_lift_R@1", "pid_acc",
            "attr_a_age", "attr_a_gender", "task_auc", "head_auc"]
    summ = (df.groupby("representation")[
                [c for c in cols if c != "representation" and c in df]]
              .agg(["mean", "std"]).round(4))
    print("\n" + "=" * 78)
    print(summ.to_string())
    print("=" * 78)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
