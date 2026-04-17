from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .pointnet_blocks import MLPDecoder, PointNetConditionEncoder


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

    Condition is a PointNet global feature from partial input.
    During training, posterior uses (condition, target_summary) to produce q(z|x,y).
    During inference, z ~ N(0,I) (or user-provided).
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
        )

        # Summarize the target full cloud (xyz only) during training for posterior.
        self.full_enc = PointNetConditionEncoder(
            in_channels=3,
            pooling=pooling,
            out_dim=cond_dim,
            dropout=dropout,
        )

        self.posterior = nn.Sequential(
            nn.Linear(cond_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

        self.decoder = MLPDecoder(
            in_dim=cond_dim + latent_dim,
            hidden_dims=(512, 512, 1024),
            out_points=full_points,
            dropout=dropout,
        )

    def encode_condition(self, partial: torch.Tensor) -> torch.Tensor:
        return self.cond_enc(partial)

    def encode_posterior(
        self,
        *,
        cond: torch.Tensor,
        full_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_sum = self.full_enc(full_xyz)
        h = self.posterior(torch.cat([cond, full_sum], dim=-1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
    ) -> CVAEOutput:
        """
        partial: (B, Np, C) where C is 3 (xyz) or 6 (xyz+features)
        full_xyz: (B, Nf, 3) for training (enables posterior)
        z: optional latent override
        """
        cond = self.encode_condition(partial)

        if z is None:
            if full_xyz is None:
                mu = torch.zeros(partial.shape[0], self.latent_dim, device=partial.device)
                logvar = torch.zeros_like(mu)
                z = torch.randn_like(mu)
            else:
                mu, logvar = self.encode_posterior(cond=cond, full_xyz=full_xyz)
                z = self.reparameterize(mu, logvar)
        else:
            mu = torch.zeros(partial.shape[0], self.latent_dim, device=partial.device)
            logvar = torch.zeros_like(mu)

        pred = self.decode(cond=cond, z=z)
        return CVAEOutput(pred_full_xyz=pred, mu=mu, logvar=logvar, cond=cond)

