from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    squared: bool = True,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Batched Chamfer distance between two point sets.

    pred:   (B, N, 3)
    target: (B, M, 3)

    Returns:
      - if reduction='none': (B,)
      - else scalar
    """
    if pred.dim() != 3 or target.dim() != 3 or pred.size(-1) != 3 or target.size(-1) != 3:
        raise ValueError(f"Expected pred/target of shape (B,N,3)/(B,M,3), got {tuple(pred.shape)} and {tuple(target.shape)}")

    # (B, N, M)
    dists = torch.cdist(pred, target, p=2)
    if squared:
        dists = dists**2

    min_pred_to_tgt = dists.min(dim=2).values  # (B, N)
    min_tgt_to_pred = dists.min(dim=1).values  # (B, M)

    per_batch = min_pred_to_tgt.mean(dim=1) + min_tgt_to_pred.mean(dim=1)  # (B,)

    if reduction == "none":
        return per_batch
    if reduction == "sum":
        return per_batch.sum()
    if reduction == "mean":
        return per_batch.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def _random_subsample(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, N, 3)
    return: (B, k, 3) (with replacement if N < k)
    """
    b, n, _ = x.shape
    if n == k:
        return x
    if n > k:
        idx = torch.randint(low=0, high=n, size=(b, k), device=x.device)
        return x.gather(1, idx[..., None].expand(-1, -1, 3))
    # n < k
    idx = torch.randint(low=0, high=n, size=(b, k), device=x.device)
    return x.gather(1, idx[..., None].expand(-1, -1, 3))


def emd_hungarian(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    points: int = 512,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Approximate Earth Mover's Distance via Hungarian assignment on a random subsample.

    Notes:
    - Uses scipy if available (recommended). Falls back to a very slow pure-numpy O(n^3)
      implementation only if scipy is missing.
    - Intended primarily for evaluation (not every training step).

    pred/target: (B, N, 3) / (B, M, 3)
    points: number of points to subsample per cloud before assignment
    Returns per-batch mean L2 cost.
    """
    if pred.dim() != 3 or target.dim() != 3 or pred.size(-1) != 3 or target.size(-1) != 3:
        raise ValueError(f"Expected pred/target of shape (B,N,3)/(B,M,3), got {tuple(pred.shape)} and {tuple(target.shape)}")

    pred_s = _random_subsample(pred, points).detach().cpu().numpy()
    tgt_s = _random_subsample(target, points).detach().cpu().numpy()

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        costs = []
        for b in range(pred_s.shape[0]):
            # (P, P)
            c = np.linalg.norm(pred_s[b, :, None, :] - tgt_s[b, None, :, :], axis=-1)
            row_ind, col_ind = linear_sum_assignment(c)
            costs.append(float(c[row_ind, col_ind].mean()))
        out = torch.tensor(costs, dtype=torch.float32, device=pred.device)
    except Exception as e:
        raise RuntimeError(
            "EMD requires scipy (scipy.optimize.linear_sum_assignment). "
            "Install scipy or run with --emd_weight 0 to disable EMD."
        ) from e

    if reduction == "none":
        return out
    if reduction == "sum":
        return out.sum()
    if reduction == "mean":
        return out.mean()
    raise ValueError(f"Unknown reduction: {reduction}")

