from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _make_mlp(in_dim: int,
              layer_dims: Sequence[int],
              *,
              activation: nn.Module | None = None,
              use_batchnorm: bool = True,
              dropout: float = 0.0) -> nn.Sequential:
    
    if activation is None:
        activation = nn.ReLU(inplace=True)

    layers: list[nn.Module] = []
    d = in_dim
    for i, out_dim in enumerate(layer_dims):
        layers.append(nn.Linear(d, out_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        d = out_dim

    return nn.Sequential(*layers)


class SharedMLP(nn.Module):
    """
    Per-point MLP implemented as 1x1 conv over (B, C, N).

    Input:  (B, in_channels, N)
    Output: (B, out_channels, N)
    """

    def __init__(self, 
                 in_channels: int, 
                 channels: Sequence[int], 
                 *, 
                 activation: nn.Module | None = None, 
                 use_batchnorm: bool = True, 
                 dropout: float = 0.0,) -> None:
        
        super().__init__()

        if activation is None:
            activation = nn.ReLU(inplace=True)

        layers: list[nn.Module] = []
        c = in_channels
        for out_c in channels:
            layers.append(nn.Conv1d(c, out_c, kernel_size=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_c))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            c = out_c

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNetConditionEncoder(nn.Module):
    """
    PointNet-style encoder that maps a partial point cloud with per-point features
    into a single global condition vector.
    """

    def __init__(self,
                 in_channels: int,
                 *,
                 point_mlp_channels: Sequence[int] = (64, 128, 256),
                 pooling: str = "mean",
                 out_dim: int = 256,
                 dropout: float = 0.0) -> None:
        
        super().__init__()

        if pooling not in {"max", "mean"}:
            raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}")

        self.pooling = pooling

        self.point_mlp = SharedMLP(in_channels=in_channels,
                                   channels=point_mlp_channels,
                                   dropout=dropout)

        last_c = int(point_mlp_channels[-1]) if len(point_mlp_channels) else in_channels

        self.proj = nn.Sequential(nn.Linear(last_c, out_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C) float
        returns: (B, out_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape (B,N,C), got {tuple(x.shape)}")

        x = x.transpose(1, 2).contiguous()  # (B, C, N)
        
        feat = self.point_mlp(x)  # (B, C', N)

        if self.pooling == "max":
            pooled = torch.max(feat, dim=-1).values  # (B, C')
        else:
            pooled = torch.mean(feat, dim=-1)  # (B, C')

        return self.proj(pooled)


class MLPDecoder(nn.Module):
    """
    Simple MLP decoder that outputs a fixed-size point set (B, out_points, 3).
    """

    def __init__(self,
                 in_dim: int,
                 *,
                 hidden_dims: Sequence[int] = (512, 512, 1024),
                 out_points: int = 3000,
                 dropout: float = 0.0) -> None:
        
        super().__init__()

        self.out_points = out_points
        self.mlp = _make_mlp(in_dim,
                             list(hidden_dims),
                             use_batchnorm=True,
                             dropout=dropout)
        
        self.out = nn.Linear(int(hidden_dims[-1]) if hidden_dims else in_dim, out_points * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        y = self.out(h)
        return y.view(x.shape[0], self.out_points, 3)

