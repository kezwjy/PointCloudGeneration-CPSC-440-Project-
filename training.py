import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from point_cloud_dataset import PointCloudDataset
from models.cvae_pointnet import ConditionalVAEPointNet, kl_normal
    
def chamfer_distance(x, y):
    """
    x: (B, N, 3)
    y: (B, M, 3)
    """
    x = x.unsqueeze(2)  # (B, N, 1, 3)
    y = y.unsqueeze(1)  # (B, 1, M, 3)

    dist = torch.norm(x - y, dim=-1)  # (B, N, M)

    cd1 = dist.min(dim=2)[0].mean(dim=1)  # x -> y
    cd2 = dist.min(dim=1)[0].mean(dim=1)  # y -> x

    return cd1 + cd2

def train(root_dir="./data/chairs_processed/train",
          batch_size=16,
          epochs=100,
          lr=1e-4,
          device="cuda" if torch.cuda.is_available() else "cpu"):

    dataset = PointCloudDataset(root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ConditionalVAEPointNet(partial_in_channels=3,   # xyz + 3 features
                                   partial_points=1000,
                                   full_points=3000,
                                   cond_dim=256,
                                   latent_dim=128).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for partial, full in loader:
            partial = partial.to(device)
            full = full.to(device)

            optimizer.zero_grad()

            out = model(partial=partial, full_xyz=full)

            pred = out.pred_full_xyz
            mu = out.mu
            logvar = out.logvar

            # reconstruction loss
            recon_loss = chamfer_distance(pred, full).mean()

            # KL loss (VERY IMPORTANT scaling term)
            kl_loss = kl_normal(mu, logvar).mean()

            # beta-VAE style weighting (important for stability)
            beta = 0.001  # you can tune this
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), "cvae_pointcloud_onlypoints.pth")

if __name__ == "__main__":
    train()