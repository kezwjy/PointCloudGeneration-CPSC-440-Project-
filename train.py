from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import PointCloudDataset
from losses.metrics import chamfer_distance, emd_hungarian
from models.cvae_pointnet import ConditionalVAEPointNet, kl_normal


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl_beta(step: int, *, warmup_steps: int, max_beta: float) -> float:
    if warmup_steps <= 0:
        return max_beta
    return max_beta * min(1.0, step / warmup_steps)


@dataclass
class TrainConfig:
    variant: Literal["xyz", "xyzfeat"]
    train_root: str
    val_root: str
    out_dir: str
    seed: int = 42
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    cond_dim: int = 256
    latent_dim: int = 128
    dropout: float = 0.0
    pooling: str = "max"

    # loss
    cd_weight: float = 1.0
    emd_weight: float = 0.0
    emd_points: int = 256
    kl_max_beta: float = 0.01
    kl_warmup_steps: int = 2000

    # logging / eval cadence
    log_every: int = 50
    val_every_epochs: int = 1


def make_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = PointCloudDataset(cfg.train_root)
    val_ds = PointCloudDataset(cfg.val_root)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=(cfg.device.startswith("cuda")),
                              drop_last=True)
    
    val_loader = DataLoader(val_ds,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            pin_memory=(cfg.device.startswith("cuda")),
                            drop_last=False)
    
    return train_loader, val_loader


def batch_to_inputs(batch: dict, *, variant: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    partial_xyz = batch["partial_xyz"].to(device)  # (B,1000,3)
    partial_feat = batch["partial_feat"].to(device)  # (B,1000,3)
    full_xyz = batch["full_xyz"].to(device)  # (B,3000,3)

    if variant == "xyz":
        partial = partial_xyz
    elif variant == "xyzfeat":
        partial = torch.cat([partial_xyz, partial_feat], dim=-1)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return partial, full_xyz


@torch.no_grad()
def run_validation(model: ConditionalVAEPointNet, 
                   loader: DataLoader,
                   *,
                   cfg: TrainConfig) -> dict[str, float]:
    model.eval()
    cd_vals: list[float] = []
    emd_vals: list[float] = []
    kl_vals: list[float] = []

    for batch in loader:
        partial, full_xyz = batch_to_inputs(batch, variant=cfg.variant, device=cfg.device)
        out = model(partial=partial, full_xyz=full_xyz)

        cd = chamfer_distance(out.pred_full_xyz, full_xyz, reduction="none")
        kl = kl_normal(out.mu, out.logvar)
        cd_vals.extend(cd.detach().cpu().tolist())
        kl_vals.extend(kl.detach().cpu().tolist())

        if cfg.emd_weight > 0:
            emd = emd_hungarian(out.pred_full_xyz, full_xyz, points=cfg.emd_points, reduction="none")
            emd_vals.extend(emd.detach().cpu().tolist())

    metrics = {
        "val_cd": float(np.mean(cd_vals)) if cd_vals else float("nan"),
        "val_kl": float(np.mean(kl_vals)) if kl_vals else float("nan"),
    }

    if cfg.emd_weight > 0:
        metrics["val_emd"] = float(np.mean(emd_vals)) if emd_vals else float("nan")

    return metrics


def save_checkpoint(out_dir: Path,
                    *,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    step: int,
                    cfg: TrainConfig,
                    extra_metrics: dict[str, float] | None = None) -> Path:
    
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    payload = {
        "epoch": epoch,
        "step": step,
        "cfg": asdict(cfg),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics": extra_metrics or {},
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["xyz", "xyzfeat"], default="xyzfeat")
    p.add_argument("--train_root", default="./data/chairs_processed/train")
    p.add_argument("--val_root", default="./data/chairs_processed/test")
    p.add_argument("--out_dir", default="./runs")
    p.add_argument("--run_name", default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--pooling", choices=["max", "mean"], default="max")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--cd_weight", type=float, default=1.0)
    p.add_argument("--emd_weight", type=float, default=0.0)
    p.add_argument("--emd_points", type=int, default=256)
    p.add_argument("--kl_max_beta", type=float, default=0.01)
    p.add_argument("--kl_warmup_steps", type=int, default=2000)

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every_epochs", type=int, default=1)
    args = p.parse_args()

    run_name = args.run_name or f"{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) / run_name

    cfg = TrainConfig(variant=args.variant,
                      train_root=args.train_root,
                      val_root=args.val_root,
                      out_dir=str(out_dir),
                      seed=args.seed,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      weight_decay=args.weight_decay,
                      num_workers=args.num_workers,
                      cond_dim=args.cond_dim,
                      latent_dim=args.latent_dim,
                      pooling=args.pooling,
                      dropout=args.dropout,
                      cd_weight=args.cd_weight,
                      emd_weight=args.emd_weight,
                      emd_points=args.emd_points,
                      kl_max_beta=args.kl_max_beta,
                      kl_warmup_steps=args.kl_warmup_steps,
                      log_every=args.log_every,
                      val_every_epochs=args.val_every_epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    seed_everything(cfg.seed)

    in_ch = 3 if cfg.variant == "xyz" else 6
    
    model = ConditionalVAEPointNet(partial_in_channels=in_ch,
                                   cond_dim=cfg.cond_dim,
                                   latent_dim=cfg.latent_dim,
                                   pooling=cfg.pooling,
                                   dropout=cfg.dropout).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader, val_loader = make_dataloaders(cfg)

    step = 0
    best_val_cd = float("inf")

    log_path = out_dir / "metrics.jsonl"
    with log_path.open("a") as f_log:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            for batch in train_loader:
                partial, full_xyz = batch_to_inputs(batch, variant=cfg.variant, device=cfg.device)

                out = model(partial=partial, full_xyz=full_xyz)
                cd = chamfer_distance(out.pred_full_xyz, full_xyz, reduction="mean")
                kl = kl_normal(out.mu, out.logvar).mean()
                beta = kl_beta(step, warmup_steps=cfg.kl_warmup_steps, max_beta=cfg.kl_max_beta)

                loss = cfg.cd_weight * cd + beta * kl
                if cfg.emd_weight > 0:
                    emd = emd_hungarian(out.pred_full_xyz, full_xyz, points=cfg.emd_points, reduction="mean")
                    loss = loss + cfg.emd_weight * emd
                else:
                    emd = None

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if step % cfg.log_every == 0:
                    rec = {
                        "epoch": epoch,
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                        "train_cd": float(cd.detach().cpu()),
                        "train_kl": float(kl.detach().cpu()),
                        "kl_beta": float(beta),
                    }
                    if emd is not None:
                        rec["train_emd"] = float(emd.detach().cpu())
                    f_log.write(json.dumps(rec) + "\n")
                    f_log.flush()

                step += 1

            if epoch % cfg.val_every_epochs == 0:
                metrics = run_validation(model, val_loader, cfg=cfg)
                metrics = {"epoch": epoch, "step": step, **metrics}
                f_log.write(json.dumps(metrics) + "\n")
                f_log.flush()

                val_cd = float(metrics["val_cd"])
                save_checkpoint(out_dir, model=model, optimizer=optimizer, epoch=epoch, step=step, cfg=cfg, extra_metrics=metrics)
                if val_cd < best_val_cd:
                    best_val_cd = val_cd
                    torch.save(model.state_dict(), out_dir / "best_model.pt")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

