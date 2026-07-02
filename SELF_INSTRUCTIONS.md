<<<<<<< HEAD
# OLMo-1B Replication — Setup Guide

Two paths: **Local** (any Linux machine with an NVIDIA GPU) and **Frontier** (OLCF cluster, AMD MI250X).

---

## Path A: Local / NVIDIA GPU

### Step 1 — Clone and install

```bash
git clone <repo-url> OLMo_training
cd OLMo_training
conda create -n olmo_pretraining python=3.11 -y
conda activate olmo_pretraining
pip install -e .[all]
```

### Step 2 — Prepare sample data (~2 min)

```bash
python scripts/create_sample_data.py \
    --output /disk/u/kerem.sahin/data/olmo1b_sample.npy \
    --num-docs 10000
```

This streams 10k documents from `allenai/c4`, tokenizes them, and saves a ~9MB `.npy` file.
Increase `--num-docs` for larger runs (e.g. `--num-docs 15000000` for ~30B tokens).

### Step 3 — Run training

```bash
# Check a GPU is free first:
nvidia-smi

# Single GPU (~20-25 GB needed):
conda run -n olmo_pretraining python -m torch.distributed.run \
    --nproc_per_node=1 scripts/train.py configs/olmo1b-sample-run.yaml

# Multi-GPU (e.g. 4 GPUs) — also enable fsdp in the config:
conda run -n olmo_pretraining python -m torch.distributed.run \
    --nproc_per_node=4 scripts/train.py configs/olmo1b-sample-run.yaml
```

Checkpoints saved to `/disk/u/kerem.sahin/data/checkpoints/olmo1b-sample-run/`.
Re-running the same command resumes automatically from the latest checkpoint.

---

## Path B: Frontier (OLCF, AMD MI250X)

### Step 1 — First-time account setup (one-off, takes days)

1. Apply at https://my.olcf.ornl.gov — get added to project `lrn089` by the PI.
2. Set up RSA token (physical keyfob or MobilePASS+). Login password = `PIN + 6-digit-code`.
3. SSH in: `ssh <username>@frontier.olcf.ornl.gov`

### Step 2 — Environment setup (once per account)

```bash
# On the Frontier login node:
module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a

conda create -p /ccs/home/<username>/.conda/envs/olmo_pretraining python=3.11 -y
conda activate /ccs/home/<username>/.conda/envs/olmo_pretraining

# ROCm build of PyTorch — required for MI250X:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2

# Clone repo to Lustre scratch (NOT home — home is slow and small):
cd /lustre/orion/lrn089/scratch/<username>
git clone <repo-url> OLMo_training
cd OLMo_training
pip install -e .[all]

# Sanity check:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: True 8
```

### Step 3 — Prepare data

```bash
# Run on a compute node (not login node) via interactive allocation:
salloc -A lrn089 -N 1 --gpus-per-node=8 -t 00:30:00 -p batch

# Then inside the allocation:
module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a
conda activate /ccs/home/<username>/.conda/envs/olmo_pretraining
cd /lustre/orion/lrn089/scratch/<username>/OLMo_training

python scripts/create_sample_data.py \
    --output /lustre/orion/lrn089/scratch/<username>/data/olmo1b_sample.npy \
    --num-docs 10000
```

### Step 4 — Edit config and submit

```bash
# Replace <username> in both files:
sed -i 's/<username>/YOUR_USERNAME/g' configs/olmo1b-frontier.yaml scripts/frontier_run.sh

# Create log directory:
mkdir -p /lustre/orion/lrn089/scratch/<username>/logs

# Submit:
sbatch scripts/frontier_run.sh

# Monitor:
squeue -u <username>
tail -f /lustre/orion/lrn089/scratch/<username>/logs/olmo1b-train-<jobid>.out
```

Checkpoints saved to `/lustre/orion/lrn089/scratch/<username>/checkpoints/olmo1b-frontier/`.
Re-submitting the job resumes automatically from the latest checkpoint.

---

## Scaling up to 30B tokens

1. Regenerate data with more docs:
   ```bash
   python scripts/create_sample_data.py --output <path>/data30b.npy --num-docs 15000000
   ```
2. Update `data.paths` in the config to point at the new file.
3. In the config, revert all lines marked `# SCALE-UP:` — specifically:
   - `t_warmup: 2000`
   - `max_duration: ceil(30e9 / (global_train_batch_size * 2048))`
   - `save_overwrite: false`
   - Re-enable evaluators and wandb
=======
# Setup

1. ```bash
   conda create -n olmo_pretraining python=3.11
   conda activate olmo_pretraining
   python -m pip install -e .[all]
   ```

2. **CUDA/driver check (important).** Step 1 installs the latest PyTorch, which may be built
   against a newer CUDA version than the GPU driver on this node supports (570.86.15 -> CUDA
   12.8 max). If `torch.cuda.is_available()` returns `False`, training silently falls back to
   CPU and looks like a hang (high CPU usage, GPU stuck at 0% util, no errors). Fix:
   ```bash
   conda run -n olmo_pretraining pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
   conda run -n olmo_pretraining python -c "import torch; print(torch.cuda.is_available())"  # must print True
   ```

3. When launching training in the background, prefer `conda run --no-capture-output -n ...` --
   plain `conda run` buffers stdout/stderr and only releases it when the subprocess exits, which
   makes a perfectly healthy run look stuck.

# Data

There are two datasets in this repo, for two different purposes:
>>>>>>> 283fb54862e7ee831b306bddc17dbd13386a799f

## 1. Small local sample (smoke testing / pipeline verification)

`scripts/create_sample_data.py` streams documents from `allenai/c4`, tokenizes them with the
OLMo tokenizer, and writes a raw uint16 token memmap (no `.npy` header -- `MemMapDataset` reads
these files as raw bytes, so don't switch this to `np.save`).

```bash
conda run -n olmo_pretraining python scripts/create_sample_data.py \
    --output /disk/u/kerem.sahin/data/olmo1b_sample.npy \
    --num-docs 10000
```

Used by `configs/olmo1b-sample-run.yaml`, a single-GPU, reduced-batch-size config for quickly
checking that the training loop runs end to end.

## 2. Official olmo-mix v1.5 shards (the real run)

`configs/data/olmo-mix-50b-shards.yaml` already records a ~64.6B-token selection (4 shards,
randomly sampled with seed 6198 from the 249-shard `configs/official-0724/OLMo-1B.yaml` manifest)
used by `configs/olmo1b-50b-run.yaml`. To regenerate the selection with a different size/seed:

```bash
conda run -n olmo_pretraining python scripts/select_shards.py \
    --target-tokens 50_000_000_000 --seed 6198
```

`MemMapDataset` can read these shards directly over HTTP (`data.paths` can be `https://...`
URLs -- this is how the official config does it, no special flag needed), which is fine for
quick checks. For an actual full-scale run, download them locally first -- otherwise every
training instance is its own HTTP range request, which won't keep up with GPU throughput and
puts sustained load on `olmo-data.org`:

```bash
conda run -n olmo_pretraining python scripts/download_shards.py \
    --shard-list configs/data/olmo-mix-50b-shards.yaml \
    --output-dir /disk/u/kerem.sahin/data/olmo-mix-50b \
    --update-config configs/olmo1b-50b-run.yaml
```

This downloads (resumable) the selected shards and rewrites `configs/olmo1b-50b-run.yaml`'s
`data.paths` to point at the local copies.

# Training

```bash
# smoke test (single GPU, small batch, local sample data)
conda run --no-capture-output -n olmo_pretraining python -m torch.distributed.run \
    --nproc_per_node=1 scripts/train.py configs/olmo1b-sample-run.yaml

# real run (after downloading shards above; --nproc_per_node=N for multi-GPU)
conda run --no-capture-output -n olmo_pretraining python -m torch.distributed.run \
    --nproc_per_node=N scripts/train.py configs/olmo1b-50b-run.yaml
```

## Repeated-token loss masking

`data.mask_repeated_tokens` (null by default) excludes induction/copy-shortcut positions from
the loss. Given a context `abcdefah` (the second `a` at index 6 repeats the one at index 0):

- `repeat`: masks the repeated token itself (index 6).
- `offset`: masks the prediction made right after a repeat (index 7, `h`) -- the position where
  an induction-style copy shortcut could influence the prediction.
- `both`: masks both.

Set via config or CLI override, e.g. `--data.mask_repeated_tokens=offset`.

## Prefix-matching (induction head) metric

Every eval step, all attention heads are scored for "prefix matching" behavior (Olsson et al.,
2022) on a synthetic repeated-random-token batch, independent of the real eval data. The top-3
heads by score are printed to the console and logged to W&B as
`eval/prefix_matching_top{1,2,3}_{score,layer,head}`.
