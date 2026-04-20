"""
Visualize reconstruction(s): partial input, predicted full cloud, ground-truth full cloud.

Examples:
  python3 scripts/visualize_reconstruction.py \\
    --variant xyzfeat \\
    --checkpoint runs/run_xyzfeat_5ep/best_model.pt \\
    --out recon.png

  # Multiple samples in one figure (rows x 3 columns), distinct chairs:
  python3 scripts/visualize_reconstruction.py \\
    --variant xyzfeat \\
    --checkpoint runs/run_xyzfeat_5ep/best_model.pt \\
    --distinct_chairs 5 \\
    --out recon_multi.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Headless-safe plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Ensure repo root is importable when run as `python3 scripts/...`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loader import PointCloudDataset, first_n_distinct_chair_indices
from models.cvae_pointnet import ConditionalVAEPointNet
from models.pcn_completion import PCNCompletionNet


def _subsample_partial(partial: torch.Tensor, k: int) -> torch.Tensor:
    b, n, c = partial.shape
    if n == k:
        return partial
    idx = torch.randint(low=0, high=n, size=(b, k), device=partial.device)
    return partial.gather(1, idx[..., None].expand(-1, -1, c))


def _load_state_dict_and_cfg(path: Path) -> tuple[dict, dict | None]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"], payload.get("cfg")
    return payload, None


def load_model(
    path: str,
    *,
    variant: str,
    cond_dim: int,
    latent_dim: int,
    pooling: str,
    dropout: float,
    device: str,
    architecture: str | None = None,
):
    ckpt = Path(path)
    state_dict, cfg_dict = _load_state_dict_and_cfg(ckpt)
    arch = (
        architecture
        if architecture is not None
        else (cfg_dict.get("architecture", "cvae") if cfg_dict else "cvae")
    )
    if cfg_dict:
        variant = cfg_dict.get("variant", variant)
        cond_dim = cfg_dict.get("cond_dim", cond_dim)
        latent_dim = cfg_dict.get("latent_dim", latent_dim)
        pooling = cfg_dict.get("pooling", pooling)
        dropout = cfg_dict.get("dropout", dropout)

    in_ch = 3 if variant == "xyz" else 6
    if arch == "pcn":
        model = PCNCompletionNet(
            partial_in_channels=in_ch,
            cond_dim=cond_dim,
            pooling=pooling,
            dropout=dropout,
        ).to(device)
    else:
        model = ConditionalVAEPointNet(
            partial_in_channels=in_ch,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            pooling=pooling,
            dropout=dropout,
        ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, arch, variant


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["xyz", "xyzfeat"], default="xyzfeat")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_root", default="./data/chairs_processed/test")
    p.add_argument("--index", type=int, default=0, help="Dataset index when --indices is not set")
    p.add_argument(
        "--indices",
        default=None,
        help="Comma-separated dataset indices for a multi-row figure (overrides --index). Example: 0,2,4,6,8",
    )
    p.add_argument(
        "--distinct_chairs",
        type=int,
        default=None,
        metavar="N",
        help="Use the first N unique chair folders (one partial each); mutually exclusive with --indices",
    )
    p.add_argument(
        "--inference_latent",
        default="posterior_mean",
        choices=["posterior_mean", "posterior_sample", "prior_sample", "zero"],
        help="CVAE only: latent at inference (ignored for PCN)",
    )
    p.add_argument(
        "--architecture",
        choices=["cvae", "pcn"],
        default=None,
        help="Override model type; default from checkpoint cfg (else cvae)",
    )
    p.add_argument(
        "--partial_points",
        type=int,
        default=None,
        help="Subsample partial to this many points (default: use all 1000)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--pooling", choices=["max", "mean"], default="max")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--out", default="reconstruction.png", help="Output PNG path")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = PointCloudDataset(args.test_root)

    if args.distinct_chairs is not None and args.indices is not None:
        raise SystemExit("Use either --indices or --distinct_chairs, not both")

    if args.distinct_chairs is not None:
        idx_list = first_n_distinct_chair_indices(ds, args.distinct_chairs)
    elif args.indices is not None:
        idx_list = []
        for part in args.indices.split(","):
            part = part.strip()
            if not part:
                continue
            idx_list.append(int(part))
        if not idx_list:
            raise SystemExit("--indices must contain at least one integer")
    else:
        idx_list = [args.index]

    for idx in idx_list:
        if idx < 0 or idx >= len(ds):
            raise SystemExit(f"index must be in [0, {len(ds) - 1}], got {idx}")

    model, arch, variant = load_model(
        args.checkpoint,
        variant=args.variant,
        cond_dim=args.cond_dim,
        latent_dim=args.latent_dim,
        pooling=args.pooling,
        dropout=args.dropout,
        device=args.device,
        architecture=args.architecture,
    )

    n_rows = len(idx_list)
    fig = plt.figure(figsize=(14, max(4.0, 3.8 * n_rows)))

    def scatter(ax, pts, title, color, part_vis, pred, gt):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        lim = np.vstack([part_vis, pred, gt])
        c = lim.mean(axis=0)
        r = np.abs(lim - c).max() + 1e-6
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)

    for row, sample_idx in enumerate(idx_list):
        sample = ds[sample_idx]
        partial_xyz = sample["partial_xyz"].unsqueeze(0).to(args.device)
        partial_feat = sample["partial_feat"].unsqueeze(0).to(args.device)
        full_xyz = sample["full_xyz"].unsqueeze(0).to(args.device)

        if variant == "xyz":
            partial = partial_xyz
        else:
            partial = torch.cat([partial_xyz, partial_feat], dim=-1)

        if args.partial_points is not None:
            partial = _subsample_partial(partial, args.partial_points)

        part_vis = partial[0, :, :3].detach().cpu().numpy()

        with torch.no_grad():
            if arch == "pcn":
                out = model(partial=partial)
            else:
                out = model(
                    partial=partial,
                    full_xyz=None,
                    inference_latent=args.inference_latent,
                )
            pred = out.pred_full_xyz[0].cpu().numpy()
        gt = full_xyz[0].cpu().numpy()

        label = f"[{sample_idx}] "
        ax1 = fig.add_subplot(n_rows, 3, row * 3 + 1, projection="3d")
        ax2 = fig.add_subplot(n_rows, 3, row * 3 + 2, projection="3d")
        ax3 = fig.add_subplot(n_rows, 3, row * 3 + 3, projection="3d")
        scatter(ax1, part_vis, label + "Partial input (xyz)", "tab:blue", part_vis, pred, gt)
        scatter(ax2, pred, label + "Reconstructed", "tab:orange", part_vis, pred, gt)
        scatter(ax3, gt, label + "Ground truth", "tab:green", part_vis, pred, gt)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
