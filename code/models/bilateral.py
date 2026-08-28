
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def mlp(dims: list[int], dropout: float = 0.0, norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class BilateralAdapter(nn.Module):

    def __init__(
        self,
        in_dim: int,
        d_pat: int = 128,
        d_path: int = 128,
        n_classes: int = 1,
        n_patients: int = 0,
        hidden: int = 512,
        dropout: float = 0.1,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.in_dim, self.d_pat, self.d_path = in_dim, d_pat, d_path

        self.trunk = mlp([in_dim, hidden], dropout=dropout)
        self.to_pat = mlp([hidden, d_pat])
        self.to_path = mlp([hidden, d_path])

        self.proj_pat = mlp([d_pat, proj_dim])

        self.proj_path = mlp([d_path, proj_dim])

        self.head_cls = mlp([d_path, hidden // 2, n_classes], dropout=dropout)

        self.head_adv = (mlp([d_path, hidden // 2, n_patients], dropout=dropout)
                         if n_patients > 0 else None)

        self.decoder = nn.Linear(d_pat + d_path, in_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.to_pat(h), self.to_path(h)

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0) -> dict[str, torch.Tensor]:
        z_pat, z_path = self.encode(x)
        out = {
            "z_pat": z_pat,
            "z_path": z_path,
            "p_pat": F.normalize(self.proj_pat(z_pat), dim=-1),
            "p_path_adv": F.normalize(
                self.proj_path(grad_reverse(z_path, grl_lambda)), dim=-1),
            "logits": self.head_cls(z_path),
            "recon": self.decoder(torch.cat([z_pat, z_path], dim=-1)),
        }
        if self.head_adv is not None:
            out["adv_logits"] = self.head_adv(grad_reverse(z_path, grl_lambda))
        return out


def fellow_eye_nce(p: torch.Tensor, pair_idx: torch.Tensor, temperature: float = 0.1,
                   candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    b = p.shape[0]
    NEG_INF = torch.finfo(p.dtype).min
    sim = (p @ p.t()) / temperature

    self_mask = torch.eye(b, dtype=torch.bool, device=p.device)
    pos = (pair_idx[:, None] == pair_idx[None, :]) & ~self_mask

    allowed = ~self_mask
    if candidate_mask is not None:
        allowed = allowed & (candidate_mask | pos)

    sim = sim.masked_fill(~allowed, NEG_INF)

    valid = pos.any(dim=1) & (allowed.sum(dim=1) > 1)
    if not valid.any():
        return p.new_zeros(())

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob = torch.where(pos, log_prob, torch.zeros_like(log_prob))
    pos_lp = log_prob.sum(dim=1) / pos.sum(dim=1).clamp(min=1)
    return -pos_lp[valid].mean()


def conditional_fellow_eye_nce(p: torch.Tensor, pair_idx: torch.Tensor,
                               y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if y.dim() > 1:
        y = y[:, 0]
    known = torch.isfinite(y)
    lab = torch.where(known, y, torch.full_like(y, -1.0))
    same_label = (lab[:, None] == lab[None, :]) & known[:, None] & known[None, :]
    return fellow_eye_nce(p, pair_idx, temperature, candidate_mask=same_label)


def decorrelation(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    a = z_a - z_a.mean(0, keepdim=True)
    b = z_b - z_b.mean(0, keepdim=True)
    a = a / (a.std(0, keepdim=True) + 1e-6)
    b = b / (b.std(0, keepdim=True) + 1e-6)
    c = (a.t() @ b) / max(z_a.shape[0] - 1, 1)
    return (c ** 2).mean()


def bilateral_consistency(z_path: torch.Tensor, pair_idx: torch.Tensor,
                          same_label: torch.Tensor | None = None) -> torch.Tensor:
    if same_label is None:
        return z_path.new_zeros(())
    b = z_path.shape[0]
    eye = torch.eye(b, dtype=torch.bool, device=z_path.device)
    pos = (pair_idx[:, None] == pair_idx[None, :]) & ~eye & same_label
    if not pos.any():
        return z_path.new_zeros(())
    zi = F.normalize(z_path, dim=-1)
    d = 1.0 - (zi @ zi.t())
    return d[pos].mean()


class BilateralLoss(nn.Module):

    def __init__(self, w_cls=1.0, w_nce=1.0, w_adv=1.0, w_dec=0.1, w_rec=0.1,
                 w_advid=0.0, temperature=0.1, multilabel=False,
                 adv_mode="conditional"):
        super().__init__()
        self.w = dict(cls=w_cls, nce=w_nce, adv=w_adv, dec=w_dec, rec=w_rec,
                      advid=w_advid)
        self.temperature = temperature
        self.multilabel = multilabel
        self.adv_mode = adv_mode

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]]:
        parts: dict[str, torch.Tensor] = {}

        y = batch.get("y")
        if y is not None and self.w["cls"]:
            logits = out["logits"]
            if self.multilabel:
                m = ~torch.isnan(y)
                parts["cls"] = (F.binary_cross_entropy_with_logits(logits[m], y[m])
                                if m.any() else logits.new_zeros(()))
            else:
                m = y >= 0
                parts["cls"] = (F.cross_entropy(logits[m], y[m].long())
                                if m.any() else logits.new_zeros(()))

        if self.w["nce"]:
            parts["nce"] = fellow_eye_nce(out["p_pat"], batch["pair_idx"], self.temperature)

        if self.w["adv"] and "p_path_adv" in out:
            if self.adv_mode == "conditional" and y is not None:
                parts["adv"] = conditional_fellow_eye_nce(
                    out["p_path_adv"], batch["pair_idx"], y, self.temperature)
            else:
                parts["adv"] = fellow_eye_nce(
                    out["p_path_adv"], batch["pair_idx"], self.temperature)

        if self.w["advid"] and "adv_logits" in out and "patient_idx" in batch:
            parts["advid"] = F.cross_entropy(out["adv_logits"], batch["patient_idx"])

        if self.w["dec"]:
            parts["dec"] = decorrelation(out["z_pat"], out["z_path"])
        if self.w["rec"]:
            parts["rec"] = F.mse_loss(out["recon"], batch["x"])

        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}


class LinearProbe(nn.Module):

    def __init__(self, in_dim: int, n_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return {"logits": self.fc(x)}


class MLPBaseline(nn.Module):

    def __init__(self, in_dim: int, n_classes: int = 1, hidden: int = 512,
                 d_out: int = 128, dropout: float = 0.1):
        super().__init__()
        self.trunk = mlp([in_dim, hidden], dropout=dropout)
        self.to_z = mlp([hidden, d_out])
        self.head = mlp([d_out, hidden // 2, n_classes], dropout=dropout)

    def forward(self, x):
        z = self.to_z(self.trunk(x))
        return {"logits": self.head(z), "z_path": z}
