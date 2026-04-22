import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for obj in os.listdir(root_dir):
            if obj.endswith('_0003'): break # ************* remove ***************
                
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

        return torch.from_numpy(partial_xyz).float(), torch.from_numpy(full_xyz).float()
    