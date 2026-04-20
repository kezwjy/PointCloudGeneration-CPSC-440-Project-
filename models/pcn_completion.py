"""
PCN-style deterministic point cloud completion (coarse + folding stages).

Inspired by Yuan et al., "PCN: Point Completion Network", CVPR 2018.
Adapted to a fixed full resolution (default 3000) via two folding steps that
double the point count: Nc -> 2Nc -> 4Nc = full_points (choose Nc = full_points // 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .pointnet_blocks import PointNetConditionEncoder, SharedMLP


@dataclass(frozen=True)
class PCNOutput:
    pred_full_xyz: torch.Tensor  # (B, full_points, 3)
    cond: torch.Tensor           # (B, cond_dim)


class FoldingBlock(nn.Module):
    """
    Doubles the number of points: (B, N, 3) + global cond -> (B, 2N, 3).
    """

    def __init__(
        self,
        cond_dim: int,
        *,
        channels: Sequence[int] = (512, 512, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_ch = 3 + cond_dim
        self.point_mlp = SharedMLP(
            in_channels=in_ch,
            channels=list(channels),
            dropout=dropout,
        )
        last_c = int(channels[-1])
        self.out_conv = nn.Conv1d(last_c, 6, kernel_size=1)

    def forward(self, pts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, 3)
        cond: (B, C)
        returns: (B, 2N, 3)
        """
        b, n, _ = pts.shape
        cexp = cond.unsqueeze(1).expand(-1, n, -1)
        x = torch.cat([pts, cexp], dim=-1)
        x = x.transpose(1, 2).contiguous()
        h = self.point_mlp(x)
        h = self.out_conv(h)
        return h.transpose(1, 2).contiguous().reshape(b, n * 2, 3)


class PCNCompletionNet(nn.Module):
    """
    Encoder: PointNet on partial -> global cond.
    Decoder: FC coarse cloud -> two folding stages doubling count to full_points.
    """

    def __init__(
        self,
        *,
        partial_in_channels: int,
        partial_points: int = 1000,
        full_points: int = 3000,
        cond_dim: int = 256,
        n_coarse: int | None = None,
        pooling: str = "max",
        dropout: float = 0.0,
        point_mlp_channels: Sequence[int] = (64, 128, 256, 512),
        folding_channels: Sequence[int] = (512, 512, 512),
    ) -> None:
        super().__init__()
        if full_points % 4 != 0:
            raise ValueError(f"full_points must be divisible by 4, got {full_points}")
        self.n_coarse = n_coarse if n_coarse is not None else full_points // 4
        if self.n_coarse * 4 != full_points:
            raise ValueError(
                f"n_coarse={self.n_coarse} must satisfy n_coarse*4==full_points ({full_points})"
            )
        self.full_points = full_points
        self.cond_dim = cond_dim

        self.cond_enc = PointNetConditionEncoder(
            in_channels=partial_in_channels,
            pooling=pooling,
            out_dim=cond_dim,
            dropout=dropout,
            point_mlp_channels=point_mlp_channels,
        )

        hidden = max(256, cond_dim)
        self.coarse_fc = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_coarse * 3),
        )

        self.fold1 = FoldingBlock(cond_dim, channels=folding_channels, dropout=dropout)
        self.fold2 = FoldingBlock(cond_dim, channels=folding_channels, dropout=dropout)

    def encode_condition(self, partial: torch.Tensor) -> torch.Tensor:
        return self.cond_enc(partial)

    def forward(self, partial: torch.Tensor, **_kwargs) -> PCNOutput:
        """
        partial: (B, Np, C)
        """
        cond = self.encode_condition(partial)
        b = partial.shape[0]
        coarse = self.coarse_fc(cond).view(b, self.n_coarse, 3)
        pts = self.fold1(coarse, cond)
        pts = self.fold2(pts, cond)
        return PCNOutput(pred_full_xyz=pts, cond=cond)
