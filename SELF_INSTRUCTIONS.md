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

---

# Session Handoff — OLMo-1B Replication Progress

## What has been done

### 1. Data preparation
- Created `scripts/create_sample_data.py` — streams N docs from `allenai/c4`, tokenizes with the OLMo-1B tokenizer, saves as a uint16 numpy memmap.
- Sample output already generated: `/disk/u/kerem.sahin/data/olmo1b_sample.npy` (~4.7M tokens, 9MB, 10k docs)
- Tokenizer used: `olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json` (already in repo)

To regenerate:
```bash
conda run -n olmo_pretraining python scripts/create_sample_data.py \
    --output /disk/u/kerem.sahin/data/olmo1b_sample.npy \
    --num-docs 10000
```

### 2. Training config
- Created `configs/olmo1b-sample-run.yaml` — smoke test config based on `configs/official-0724/OLMo-1B.yaml`
- Model architecture is **identical** to OLMo-1B
- Differences from original are marked with `# SCALE-UP:` comments in the file:
  - `data.paths` points to the local `.npy` file instead of production URLs
  - `max_duration: 1000` steps (original: 739,328)
  - `global_train_batch_size: 32` (original: 2048)
  - `scheduler.t_warmup: 100` (original: 2000)
  - No evaluators, no wandb, no FSDP

### 3. Training command
```bash
conda run -n olmo_pretraining python -m torch.distributed.run \
    --nproc_per_node=1 scripts/train.py configs/olmo1b-sample-run.yaml
```

Use `--nproc_per_node=N` for multi-GPU.

## Status
Training has NOT been run yet — all 8 A100s were occupied. Ready to run on a free GPU (~20-25GB needed).

## Next steps
1. Run the smoke test
2. If it passes, scale up:
   - Prepare a larger dataset (~30B tokens) by increasing `--num-docs` in the data prep script
   - Revert the `# SCALE-UP:` settings in the config
   - Set `max_duration = ceil(30e9 / (global_train_batch_size * 2048))`
