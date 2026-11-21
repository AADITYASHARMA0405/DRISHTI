"""Training losses. All take L2-normalised features (B, D) and person ids (B,).

sdm_loss              : Similarity Distribution Matching (Jiang & Ye, IRRA, CVPR 2023).
robust_triplet_loss   : hardest-negative triplet loss whose margin shrinks with the number of
                        corrupted attributes:  m_i = m0 * (1 - k_i / A_i).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sdm_loss(img: torch.Tensor, txt: torch.Tensor, pid: torch.Tensor, scale: torch.Tensor,
             eps: float = 1e-8) -> torch.Tensor:
    """KL divergence between the softmax of scaled similarities and the true matching distribution,
    in both directions. Captions of the same person are all treated as positives."""
    labels = (pid.view(-1, 1) == pid.view(1, -1)).float()
    target = labels / labels.sum(dim=1, keepdim=True)
    sim_t2i = scale * txt @ img.t()
    loss = 0.0
    for logits in (sim_t2i, sim_t2i.t()):
        pred = F.softmax(logits, dim=1)
        loss = loss + (pred * (F.log_softmax(logits, dim=1) - torch.log(target + eps))).sum(1).mean()
    return loss


def robust_triplet_loss(img: torch.Tensor, txt: torch.Tensor, pid: torch.Tensor, k: torch.Tensor,
                        A: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Bidirectional hardest-negative triplet loss with a per-sample soft margin.

    A caption with k of its A attributes corrupted is still a positive, but it should not be forced as
    far from the negatives as a clean caption, so its margin is reduced in proportion to k / A.
    """
    sim = txt @ img.t()                                   # (B, B), row = caption, col = image
    pos = sim.diag()
    neg_mask = pid.view(-1, 1) != pid.view(1, -1)
    m = margin * (1.0 - (k / A.clamp(min=1.0)).clamp(0.0, 1.0))
    very_low = torch.finfo(sim.dtype).min
    hard_t2i = sim.masked_fill(~neg_mask, very_low).max(dim=1).values
    hard_i2t = sim.t().masked_fill(~neg_mask, very_low).max(dim=1).values
    valid = neg_mask.any(dim=1).float()
    loss = F.relu(m - pos + hard_t2i) * valid + F.relu(m - pos + hard_i2t) * valid
    return loss.sum() / valid.sum().clamp(min=1.0)


def total_loss(model, batch: dict, img: torch.Tensor, txt: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    lc = cfg["loss"]
    parts = {"sdm": sdm_loss(img, txt, batch["pid"], model.scale)}
    loss = float(lc.get("sdm", 1.0)) * parts["sdm"]
    if float(lc.get("robust_triplet", 0.0)) > 0:
        parts["robust_triplet"] = robust_triplet_loss(img, txt, batch["pid"], batch["k"], batch["A"],
                                                      float(lc.get("margin", 0.2)))
        loss = loss + float(lc["robust_triplet"]) * parts["robust_triplet"]
    return loss, {k: float(v.detach()) for k, v in parts.items()}
