import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from point_cloud_dataset import PointCloudDataset
from models.cvae_pointnet import ConditionalVAEPointNet, kl_normal

# reuse your dataset
# from dataset import ChairCompletionDataset
# from model import ConditionalVAEPointNet


def save_ply(points, filename):
    """
    points: (N, 3)
    """
    points = points.cpu().numpy()

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def chamfer_distance(x, y):
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    dist = torch.norm(x - y, dim=-1)

    cd1 = dist.min(dim=2)[0].mean(dim=1)
    cd2 = dist.min(dim=1)[0].mean(dim=1)

    return cd1 + cd2

def evaluate(model_path="cvae_pointcloud_deletelater.pth",
             root_dir="./data/chairs_processed/test",
             save_dir="./eval/eval_results",
             batch_size=16,
             device="cuda" if torch.cuda.is_available() else "cpu"):
    
    os.makedirs(save_dir, exist_ok=True)

    dataset = PointCloudDataset(root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ConditionalVAEPointNet(partial_in_channels=3,
                                   partial_points=2000,
                                   full_points=5000,
                                   cond_dim=256,
                                   latent_dim=128).to(device).float()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_cd = 0
    count = 0

    with torch.no_grad():
        for batch_idx, (partial, full) in enumerate(loader):
            partial = partial.to(device)
            full = full.to(device)

            # inference (NO full_xyz!)
            out = model(partial=partial)

            pred = out.pred_full_xyz  # (B, 3000, 3)

            # metric
            cd = chamfer_distance(pred, full).mean()
            total_cd += cd.item()
            count += 1




            cond = model.encode_condition(partial)
            print("cond std:", cond.std().item())

            z = torch.randn(partial.shape[0], model.latent_dim, device=partial.device)
            print("z std:", z.std().item())

            out1 = model.decode(cond=cond, z=z)
            out2 = model.decode(cond=cond, z=torch.randn_like(z))

            print("pred diff:", (out1 - out2).abs().mean().item())





            # save outputs
            for i in range(pred.shape[0]):
                idx = batch_idx * batch_size + i

                pred_points = pred[i]
                full_points = full[i]
                partial_points = partial[i][:, :3]  # only xyz

                save_ply(pred_points, os.path.join(save_dir, f"{idx}_pred.ply"))
                save_ply(full_points, os.path.join(save_dir, f"{idx}_gt.ply"))
                save_ply(partial_points, os.path.join(save_dir, f"{idx}_partial.ply"))

    print(f"\nEvaluation Chamfer Distance: {total_cd / count:.6f}")

if __name__ == "__main__":
    evaluate(batch_size=2)