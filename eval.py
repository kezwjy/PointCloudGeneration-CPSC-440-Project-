from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import PointCloudDataset
from losses.metrics import chamfer_distance, emd_hungarian
from models.cvae_pointnet import ConditionalVAEPointNet
from models.pcn_completion import PCNCompletionNet


def _load_state_dict_and_cfg(path: Path) -> tuple[dict, dict | None]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"], payload.get("cfg")
    return payload, None


def _build_model_for_eval(
    *,
    variant: Literal["xyz", "xyzfeat"],
    architecture: Literal["cvae", "pcn"],
    cond_dim: int,
    latent_dim: int,
    pooling: str,
    dropout: float,
    device: str,
) -> nn.Module:
    in_ch = 3 if variant == "xyz" else 6
    if architecture == "pcn":
        return PCNCompletionNet(
            partial_in_channels=in_ch,
            cond_dim=cond_dim,
            pooling=pooling,
            dropout=dropout,
        ).to(device)
    return ConditionalVAEPointNet(
        partial_in_channels=in_ch,
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        pooling=pooling,
        dropout=dropout,
    ).to(device)


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
    model: nn.Module,
    architecture: Literal["cvae", "pcn"],
    loader: DataLoader,
    variant: Literal["xyz", "xyzfeat"],
    device: str,
    sparsity_k: int | None,
    emd_points: int,
    inference_latent: str | None,
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

        if architecture == "pcn":
            out = model(partial=partial)
        else:
            out = model(partial=partial, full_xyz=None, inference_latent=inference_latent)

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
    p.add_argument(
        "--inference_latent",
        default="posterior_mean",
        choices=["posterior_mean", "posterior_sample", "prior_sample", "zero"],
        help="CVAE only: how to form z at inference (ignored for --architecture pcn)",
    )
    p.add_argument(
        "--architecture",
        choices=["cvae", "pcn"],
        default=None,
        help="Override model type; default is read from checkpoint cfg (falls back to cvae)",
    )
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

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.name.endswith(".pt"):
        raise ValueError("checkpoint must be a .pt file")

    state_dict, cfg_dict = _load_state_dict_and_cfg(ckpt_path)
    architecture: Literal["cvae", "pcn"] = (
        args.architecture
        if args.architecture is not None
        else (cfg_dict.get("architecture", "cvae") if cfg_dict else "cvae")
    )
    variant: Literal["xyz", "xyzfeat"] = cfg_dict.get("variant", args.variant) if cfg_dict else args.variant
    cond_dim = cfg_dict.get("cond_dim", args.cond_dim) if cfg_dict else args.cond_dim
    latent_dim = cfg_dict.get("latent_dim", args.latent_dim) if cfg_dict else args.latent_dim
    pooling = cfg_dict.get("pooling", args.pooling) if cfg_dict else args.pooling
    dropout = cfg_dict.get("dropout", args.dropout) if cfg_dict else args.dropout

    model = _build_model_for_eval(
        variant=variant,
        architecture=architecture,
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        pooling=pooling,
        dropout=dropout,
        device=args.device,
    )
    model.load_state_dict(state_dict, strict=True)

    sweep_ks = []
    for part in args.sweep.split(","):
        part = part.strip()
        if not part:
            continue
        sweep_ks.append(int(part))

    results = {
        "variant": variant,
        "architecture": architecture,
        "checkpoint": str(ckpt_path),
        "emd_points": args.emd_points,
        "inference_latent": None if architecture == "pcn" else args.inference_latent,
        "sweep": {},
    }

    rows = []
    for k in sweep_ks:
        m = evaluate(
            model=model,
            architecture=architecture,
            loader=loader,
            variant=variant,
            device=args.device,
            sparsity_k=k,
            emd_points=args.emd_points,
            inference_latent=args.inference_latent,
        )
        results["sweep"][str(k)] = m
        rows.append({"partial_points": k, **m})

    # Also evaluate the default (no extra subsampling)
    m_full = evaluate(
        model=model,
        architecture=architecture,
        loader=loader,
        variant=variant,
        device=args.device,
        sparsity_k=None,
        emd_points=args.emd_points,
        inference_latent=args.inference_latent,
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

