# Point cloud completion (CVAE and PCN-style)

This project trains and evaluates models that complete **full** point clouds from **partial** inputs on the chairs dataset.

We compare **input variants**:
- **xyz-only**: partial xyz coordinates only
- **xyz+features**: partial xyz plus geometric features (surface variation, normal alignment, local density)

We support **two architectures**:
- **`cvae`** (default): PointNet conditional **VAE** with **q(z | partial)** and Chamfer + KL (see [`models/cvae_pointnet.py`](models/cvae_pointnet.py)).
- **`pcn`**: Deterministic **PCN-style** coarse-to-fine completion (encoder + coarse FC + two folding stages), Chamfer only (no KL). Inspired by Yuan et al., *PCN: Point Completion Network*, CVPR 2018 ([`models/pcn_completion.py`](models/pcn_completion.py)).

The dataset is under `data/chairs_processed/`.

## Model (CVAE latent)

For **`--architecture cvae`**, the approximate posterior is **q(z | partial)** only (amortized from the same PointNet embedding used for decoding). Training and evaluation both use this path.

**Checkpoints from older commits** (posterior used both partial and full-cloud encoders) **do not load** into the current CVAE; retrain after pulling those changes.

## Checkpoints

New runs save `best_model.pt` as `{"model_state", "cfg"}` so `eval.py` and `visualize_reconstruction.py` can read `architecture`, `variant`, and hyperparameters. Raw state dict-only files still work if you pass matching `--variant` / `--architecture` where needed.

## Data format
See `data/README.txt` for details. In short:
- `full.npy`: `(3000, 3)` xyz
- `partial1.npy`, `partial2.npy`: `(1000, 6)` where columns are `[x,y,z, surface_variation, alignment, density]`

Each chair folder contributes **two** dataset rows (`partial1` and `partial2` with the same `full.npy`). Consecutive indices often refer to the **same** object. For diversity in figures, use `scripts/visualize_reconstruction.py --distinct_chairs N` or pick non-consecutive indices (e.g. `0,2,4,6,8`). Helpers: `data_loader.distinct_chair_start_indices`, `data_loader.first_n_distinct_chair_indices`.

## Setup
This repo assumes Python 3 and PyTorch are installed. EMD evaluation uses `scipy` (already present on many setups).

## Train
Default settings use **30 epochs**, **KL warmup 3000** steps, and **max KL weight 0.02** for the CVAE (see `train.py`).

**Conditional VAE** (xyz + features):

```bash
python3 train.py --variant xyzfeat --architecture cvae --out_dir runs --epochs 30
```

**PCN-style** (same data, Chamfer + optional EMD, no KL):

```bash
python3 train.py --variant xyzfeat --architecture pcn --out_dir runs --epochs 30
```

Notes:
- **`--architecture`**: `cvae` (default) or `pcn`.
- CVAE loss: **Chamfer + KL**; optional **EMD** via `--emd_weight`. PCN loss: **Chamfer** (+ optional EMD); KL flags are ignored.
- KL annealing (`--kl_warmup_steps`, `--kl_max_beta`) applies to the CVAE only.
- Validation logs **`val_cond_mean_cos`**: mean pairwise cosine similarity of normalized condition vectors (high values suggest similar encodings within a batch). CVAE runs also log **`val_kl`**.

Example with EMD in the training objective:

```bash
python3 train.py --variant xyzfeat --architecture pcn --emd_weight 0.05 --emd_points 256 --out_dir runs --epochs 40
```

## Evaluate (CD + EMD) with sparsity sweep
After training, evaluate a checkpoint on the test set across different partial-point counts.

```bash
python3 eval.py \
  --checkpoint runs/<run_name>/best_model.pt \
  --out_dir eval/<run_name> \
  --sweep 1000,500,250 \
  --emd_points 256
```

`variant`, `architecture`, and hyperparameters are read from the checkpoint `cfg` when present. You can override **`--architecture`** if needed.

For **CVAE** checkpoints, **`--inference_latent`** applies: **`posterior_mean`** (default), **`posterior_sample`**, **`prior_sample`**, or **`zero`**. For **PCN**, this flag is ignored (see `results.json`: `inference_latent` is `null`).

Outputs:
- `eval/.../results.json`
- `eval/.../results.csv`

## Visualize one or more reconstructions (PNG)
Save side-by-side figures: partial input (xyz), model prediction, ground-truth full cloud.

```bash
python3 scripts/visualize_reconstruction.py \
  --checkpoint runs/<run_name>/best_model.pt \
  --test_root ./data/chairs_processed/test \
  --index 0 \
  --out reconstruction.png
```

**Five different chairs** (one row per object):

```bash
python3 scripts/visualize_reconstruction.py \
  --checkpoint runs/<run_name>/best_model.pt \
  --distinct_chairs 5 \
  --out reconstruction_5chairs.png
```

Optional: `--partial_points 500` for a sparser partial. **`--inference_latent`** applies only to the CVAE.

## Repo layout
- `models/cvae_pointnet.py`: conditional VAE
- `models/pcn_completion.py`: PCN-style coarse + folding decoder
- `models/pointnet_blocks.py`: PointNet building blocks
- `losses/metrics.py`: Chamfer Distance and EMD (Hungarian on a subsample)
- `train.py`: training loop + checkpointing
- `eval.py`: test evaluation + sparsity sweep
- `scripts/visualize_reconstruction.py`: PNG visualization
- `data_loader.py`: dataset + distinct-chair index helpers
