from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import PointCloudDataset
from losses.metrics import chamfer_distance, emd_hungarian
from models.cvae_pointnet import ConditionalVAEPointNet


def _subsample_partial(partial: torch.Tensor, k: int) -> torch.Tensor:
    """
    partial: (B, N, C)
    returns: (B, k, C) with replacement if needed.
    """
    b, n, c = partial.shape
    if n == k:
        return partial
    idx = torch.randint(low=0, high=n, size=(b, k), device=partial.device)
    return partial.gather(1, idx[..., None].expand(-1, -1, c))


@torch.no_grad()
def evaluate(
    *,
    model: ConditionalVAEPointNet,
    loader: DataLoader,
    variant: Literal["xyz", "xyzfeat"],
    device: str,
    sparsity_k: int | None,
    emd_points: int,
) -> dict[str, float]:
    model.eval()

    cd_vals: list[float] = []
    emd_vals: list[float] = []

    for batch in loader:
        partial_xyz = batch["partial_xyz"].to(device)
        partial_feat = batch["partial_feat"].to(device)
        full_xyz = batch["full_xyz"].to(device)

        if variant == "xyz":
            partial = partial_xyz
        else:
            partial = torch.cat([partial_xyz, partial_feat], dim=-1)

        if sparsity_k is not None:
            partial = _subsample_partial(partial, sparsity_k)

        out = model(partial=partial, full_xyz=None)

        cd = chamfer_distance(out.pred_full_xyz, full_xyz, reduction="none")
        emd = emd_hungarian(out.pred_full_xyz, full_xyz, points=emd_points, reduction="none")
        cd_vals.extend(cd.detach().cpu().tolist())
        emd_vals.extend(emd.detach().cpu().tolist())

    return {
        "cd": float(np.mean(cd_vals)) if cd_vals else float("nan"),
        "emd": float(np.mean(emd_vals)) if emd_vals else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["xyz", "xyzfeat"], default="xyzfeat")
    p.add_argument("--test_root", default="./data/chairs_processed/test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--pooling", choices=["max", "mean"], default="max")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt or best_model.pt")
    p.add_argument("--out_dir", default="./eval")
    p.add_argument("--emd_points", type=int, default=256)
    p.add_argument("--sweep", default="1000,500,250", help="Comma-separated partial point counts to evaluate")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PointCloudDataset(args.test_root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    in_ch = 3 if args.variant == "xyz" else 6
    model = ConditionalVAEPointNet(
        partial_in_channels=in_ch,
        cond_dim=args.cond_dim,
        latent_dim=args.latent_dim,
        pooling=args.pooling,
        dropout=args.dropout,
    ).to(args.device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.name.endswith(".pt"):
        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state" in payload:
            model.load_state_dict(payload["model_state"])
        else:
            model.load_state_dict(payload)
    else:
        raise ValueError("checkpoint must be a .pt file")

    sweep_ks = []
    for part in args.sweep.split(","):
        part = part.strip()
        if not part:
            continue
        sweep_ks.append(int(part))

    results = {
        "variant": args.variant,
        "checkpoint": str(ckpt_path),
        "emd_points": args.emd_points,
        "sweep": {},
    }

    rows = []
    for k in sweep_ks:
        m = evaluate(
            model=model,
            loader=loader,
            variant=args.variant,
            device=args.device,
            sparsity_k=k,
            emd_points=args.emd_points,
        )
        results["sweep"][str(k)] = m
        rows.append({"partial_points": k, **m})

    # Also evaluate the default (no extra subsampling)
    m_full = evaluate(
        model=model,
        loader=loader,
        variant=args.variant,
        device=args.device,
        sparsity_k=None,
        emd_points=args.emd_points,
    )
    results["sweep"]["default"] = m_full
    rows.append({"partial_points": -1, **m_full})

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["partial_points", "cd", "emd"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_dir/'results.json'} and {out_dir/'results.csv'}")


if __name__ == "__main__":
    main()

