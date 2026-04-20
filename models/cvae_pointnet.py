from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn

from .pointnet_blocks import MLPDecoder, PointNetConditionEncoder

InferenceLatentMode = Literal["posterior_mean", "posterior_sample", "prior_sample", "zero"]


@dataclass(frozen=True)
class CVAEOutput:
    pred_full_xyz: torch.Tensor  # (B, 3000, 3)
    mu: torch.Tensor            # (B, latent_dim)
    logvar: torch.Tensor        # (B, latent_dim)
    cond: torch.Tensor          # (B, cond_dim)


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu, sigma^2) || N(0, I) ) per batch element.
    Returns: (B,)
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1)


class ConditionalVAEPointNet(nn.Module):
    """
    Conditional VAE for point cloud completion.

    Condition is a PointNet global feature from the partial input.
    The approximate posterior is **q(z | partial)** only (amortized via the same
    global embedding used for decoding), so training and inference use the same
    latent path. Optional ``inference_latent`` controls how ``z`` is chosen when
    ``z`` is not provided (e.g. prior sample for ablations).
    """

    def __init__(
        self,
        *,
        partial_in_channels: int,
        partial_points: int = 1000,
        full_points: int = 3000,
        cond_dim: int = 256,
        latent_dim: int = 128,
        pooling: str = "max",
        dropout: float = 0.0,
        point_mlp_channels: Sequence[int] = (64, 128, 256, 512),
        decoder_hidden_dims: Sequence[int] = (512, 768, 1024),
    ) -> None:
        super().__init__()
        self.partial_points = partial_points
        self.full_points = full_points
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.cond_enc = PointNetConditionEncoder(
            in_channels=partial_in_channels,
            pooling=pooling,
            out_dim=cond_dim,
            dropout=dropout,
            point_mlp_channels=point_mlp_channels,
        )

        self.posterior = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

        self.decoder = MLPDecoder(
            in_dim=cond_dim + latent_dim,
            hidden_dims=tuple(decoder_hidden_dims),
            out_points=full_points,
            dropout=dropout,
        )

    def encode_condition(self, partial: torch.Tensor) -> torch.Tensor:
        return self.cond_enc(partial)

    def encode_latent(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q(z | partial): amortized as MLP(cond) with cond = encoder(partial)."""
        h = self.posterior(cond)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def sample_z(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        inference_latent: InferenceLatentMode | None = None,
    ) -> torch.Tensor:
        if inference_latent == "zero":
            return torch.zeros_like(mu)
        if inference_latent == "prior_sample":
            return torch.randn_like(mu)
        if inference_latent == "posterior_mean":
            return mu
        if inference_latent == "posterior_sample":
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, *, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([cond, z], dim=-1))

    def forward(
        self,
        *,
        partial: torch.Tensor,
        full_xyz: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        inference_latent: InferenceLatentMode | str | None = None,
    ) -> CVAEOutput:
        """
        partial: (B, Np, C) where C is 3 (xyz) or 6 (xyz+features)
        full_xyz: optional, kept for API compatibility with training scripts (unused).
        z: optional latent override
        inference_latent: when ``z`` is None, how to form ``z`` from ``q(z|partial)``
            (ignored in training except explicit modes). Default: train=reparam,
            eval=posterior mean.
        """
        _ = full_xyz  # q(z | partial) only; do not condition on ground-truth full cloud
        cond = self.encode_condition(partial)

        if z is None:
            mu, logvar = self.encode_latent(cond)
            z = self.sample_z(mu, logvar, inference_latent=inference_latent)
        else:
            mu = torch.zeros(partial.shape[0], self.latent_dim, device=partial.device)
            logvar = torch.zeros_like(mu)

        pred = self.decode(cond=cond, z=z)
        return CVAEOutput(pred_full_xyz=pred, mu=mu, logvar=logvar, cond=cond)
