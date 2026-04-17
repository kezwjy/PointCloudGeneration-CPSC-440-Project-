# Conditional VAE for point cloud completion

This project trains and evaluates a **PointNet-style conditional VAE** for completing full point clouds from **partial point clouds**.

We compare two variants:
- **xyz-only**: condition on partial xyz coordinates only
- **xyz+features**: condition on partial xyz plus explicit geometric features (surface variation, normal alignment, local density)

The dataset in this repo is already processed under `data/chairs_processed/`.

## Data format
See `data/README.txt` for details. In short:
- `full.npy`: `(3000, 3)` xyz
- `partial1.npy`, `partial2.npy`: `(1000, 6)` where columns are `[x,y,z, surface_variation, alignment, density]`

## Setup
This repo assumes Python 3 and PyTorch are installed. EMD evaluation uses `scipy` (already present on many setups).

## Train
Train the **xyz-only** baseline:

```bash
python3 train.py --variant xyz --out_dir runs --epochs 20
```

Train the **xyz+features** model:

```bash
python3 train.py --variant xyzfeat --out_dir runs --epochs 20
```

Notes:
- Training loss uses **Chamfer Distance + KL** by default. You can optionally add EMD into the training objective (slow) via `--emd_weight`.
- KL annealing is controlled by `--kl_warmup_steps` and `--kl_max_beta`.

Example with EMD enabled (evaluation-style approximation):

```bash
python3 train.py --variant xyzfeat --emd_weight 0.1 --emd_points 256 --out_dir runs --epochs 20
```

## Evaluate (CD + EMD) with sparsity sweep
After training, evaluate a checkpoint on the test set across different partial-point counts.

Example (use `best_model.pt` or `checkpoint.pt` from the run folder):

```bash
python3 eval.py \
  --variant xyzfeat \
  --checkpoint runs/<run_name>/best_model.pt \
  --out_dir eval/xyzfeat \
  --sweep 1000,500,250 \
  --emd_points 256
```

Outputs:
- `eval/.../results.json`
- `eval/.../results.csv`

## Visualize one reconstruction (PNG)
Save a side-by-side figure: partial input (xyz), model prediction, ground-truth full cloud.

```bash
python3 scripts/visualize_reconstruction.py \
  --variant xyzfeat \
  --checkpoint runs/<run_name>/best_model.pt \
  --test_root ./data/chairs_processed/test \
  --index 0 \
  --out reconstruction.png
```

Optional: `--partial_points 500` to match a sparser partial at eval time.

## Repo layout
- `models/cvae_pointnet.py`: conditional VAE model
- `models/pointnet_blocks.py`: PointNet building blocks
- `losses/metrics.py`: Chamfer Distance and EMD (Hungarian on a subsample)
- `train.py`: training loop + checkpointing
- `eval.py`: test evaluation + sparsity sweep
- `scripts/visualize_reconstruction.py`: one-sample PNG visualization
