"""Retrieval metrics for text-to-image person search: Rank-k (CMC), mAP and mINP.

Follows the standard TBPS protocol (as in IRRA): for every text query, rank all gallery images by
similarity; a gallery image is relevant when its person id equals the query's person id.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def retrieval_metrics(sim: torch.Tensor, q_pids: torch.Tensor, g_pids: torch.Tensor,
                      ks=(1, 5, 10), chunk: int = 2048) -> dict:
    """sim: (Q, G) similarity matrix. Returns percentages: R1, R5, R10, mAP, mINP."""
    q_pids, g_pids = q_pids.to(sim.device), g_pids.to(sim.device)
    G = sim.shape[1]
    pos = torch.arange(1, G + 1, device=sim.device, dtype=torch.float)
    hits = {k: 0.0 for k in ks}
    ap_sum = inp_sum = 0.0
    n = 0
    for s in range(0, sim.shape[0], chunk):
        order = sim[s:s + chunk].argsort(dim=1, descending=True)
        match = (g_pids[order] == q_pids[s:s + chunk].view(-1, 1)).float()   # (q, G)
        num_rel = match.sum(1)
        keep = num_rel > 0
        match, num_rel = match[keep], num_rel[keep]
        for k in ks:
            hits[k] += (match[:, :k].sum(1) > 0).float().sum().item()
        cum = match.cumsum(1)
        ap_sum += ((cum / pos) * match).sum(1).div(num_rel).sum().item()
        last_hit = (match * pos).max(dim=1).values                              # rank of the hardest positive
        inp_sum += (num_rel / last_hit).sum().item()
        n += int(keep.sum())
    if n == 0:
        raise ValueError("No query has a matching gallery image.")
    out = {f"R{k}": 100.0 * hits[k] / n for k in ks}
    out.update({"mAP": 100.0 * ap_sum / n, "mINP": 100.0 * inp_sum / n, "n_queries": n})
    return out
