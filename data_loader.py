import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for obj in sorted(os.listdir(root_dir)):
            obj_path = os.path.join(root_dir, obj)

            full_path = os.path.join(obj_path, 'full.npy')
            p1_path = os.path.join(obj_path, 'partial1.npy')
            p2_path = os.path.join(obj_path, 'partial2.npy')

            if os.path.exists(full_path):
                self.samples.append((p1_path, full_path))
                self.samples.append((p2_path, full_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        partial_path, full_path = self.samples[idx]

        partial = np.load(partial_path)   # (Np, F)
        full = np.load(full_path)         # (Nf, F or 3)

        # split features
        partial_xyz = partial[:, :3]
        partial_feat = partial[:, 3:]

        full_xyz = full[:, :3]

        return {
            "partial_xyz": torch.tensor(partial_xyz, dtype=torch.float32),
            "partial_feat": torch.tensor(partial_feat, dtype=torch.float32),
            "full_xyz": torch.tensor(full_xyz, dtype=torch.float32),
        }


def distinct_chair_start_indices(dataset: PointCloudDataset) -> list[int]:
    """
    First dataset index for each unique object folder (e.g. chair_0909).

    The dataset stores two rows per chair (partial1 and partial2); consecutive
    indices often share the same ground-truth full cloud. Use this list to pick
    one row per chair for diversity comparisons.
    """
    seen: set[str] = set()
    out: list[int] = []
    for i, (partial_path, _) in enumerate(dataset.samples):
        cid = Path(partial_path).parent.name
        if cid not in seen:
            seen.add(cid)
            out.append(i)
    return out


def first_n_distinct_chair_indices(dataset: PointCloudDataset, n: int) -> list[int]:
    """First ``n`` indices from :func:`distinct_chair_start_indices`."""
    idx = distinct_chair_start_indices(dataset)
    if n > len(idx):
        raise ValueError(f"Requested {n} distinct chairs but dataset has {len(idx)}")
    return idx[:n]


if __name__=="__main__":
    # Example usage
    from torch.utils.data import DataLoader

    dataset = PointCloudDataset('./data/chairs_processed/train')

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )