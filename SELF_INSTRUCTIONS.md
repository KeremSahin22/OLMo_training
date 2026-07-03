# OLMo-1B Replication on Frontier (OLCF)

This document is a complete guide to replicating OLMo-1B training on Frontier (AMD MI250X).
Starting from a fresh clone, follow these steps in order.

---

# 1. Repository

```bash
cd /lustre/orion/lrn089/scratch/kerem.sahin
git clone <repo-url> OLMo_training
cd OLMo_training
```

Always work from Lustre scratch, not home — home is slow, small, and not suitable for training I/O.

---

# 2. Environment setup (one-time)

```bash
module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a

conda create -p /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining python=3.11 -y
conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining

cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training
pip install -e .[all]

# Install ROCm PyTorch AFTER pip install -e .[all] — the above overwrites it with a CUDA build.
# If you do this in the wrong order, torch.cuda.is_available() returns False and training
# silently runs on CPU (looks like a hang: high CPU, GPU at 0%).
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2

# Verify — must print True 8:
python -c "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count())"
```

---

# 3. W&B setup (one-time)

Compute nodes on Frontier block all outbound internet, so wandb runs in offline mode and is
synced to the cloud from the login node. Credentials stored in `~/.netrc` are accessible from
compute nodes via NFS, but the actual API connection must happen from the login node.

```bash
# On the login node (modules already loaded, conda activated):
wandb login  # enter API key from wandb.ai/settings
```

---

# 4. Prepare data

## What data

We use 4 shards from the official OLMo-mix v1.5 dataset, randomly sampled with seed 6198 from
the 249-shard manifest in `configs/official-0724/OLMo-1B.yaml`. Total: ~64.6B tokens.

The shard selection is recorded in `configs/data/olmo-mix-50b-shards.yaml`. To regenerate
with a different size or seed:

```bash
python scripts/select_shards.py --target-tokens 50_000_000_000 --seed 6198
```

## Downloading

Login nodes and compute nodes on Frontier cannot reach `olmo-data.org` (connection times out).
Use the DTN node instead, which has unrestricted internet access:

```bash
ssh kerem.sahin@dtn.olcf.ornl.gov
```

The DTN has no module system, so use `wget` directly. IPv6 is broken — always pass `-4`:

```bash
# On the DTN, download each shard manually (get URLs from configs/data/olmo-mix-50b-shards.yaml):
mkdir -p /lustre/orion/lrn089/scratch/kerem.sahin/data/olmo-mix-50b
wget -4 -P /lustre/orion/lrn089/scratch/kerem.sahin/data/olmo-mix-50b <shard-url>
# Repeat for each shard. wget is resumable — re-run if interrupted.
```

Alternatively, submit `scripts/prepare_data.sh` as a Slurm batch job from the login node
(uses `download_shards.py` which handles resumption automatically). Edit the username
placeholder before submitting:

```bash
mkdir -p /lustre/orion/lrn089/scratch/kerem.sahin/logs
sbatch scripts/prepare_data.sh
tail -f /lustre/orion/lrn089/scratch/kerem.sahin/logs/prepare-data-JOBID.out
```

Downloaded shards go to:
```
/lustre/orion/lrn089/scratch/kerem.sahin/data/olmo-mix-50b/
  part-035-00000.npy
  part-057-00000.npy
  part-062-00000.npy
  part-008-00000.npy
```

These paths are already set in `configs/olmo1b-frontier.yaml`.

---

# 5. Config overview (`configs/olmo1b-frontier.yaml`)

Key settings and why they are set the way they are:

| Setting | Value | Reason |
|---|---|---|
| `model.*` | OLMo-1B architecture | Matches original paper |
| `precision` | `amp_bf16` | fp16 causes NaNs on MI250X |
| `flash_attention` | `false` | Not supported on MI250X |
| `compile` | `null` | torch.compile unstable on ROCm |
| `distributed_strategy` | `fsdp` | Required for 4-node training |
| `global_train_batch_size` | `2048` | Matches original OLMo-1B |
| `device_train_microbatch_size` | `4` | Fits in 64 GB HBM2e per GCD |
| `max_duration` | `"50e9T"` | 50B total tokens budget |
| `mask_repeated_tokens` | `offset` | Masks copy-shortcut positions from loss |
| `save_folder` | Lustre path without job ID | Stable path enables checkpoint resuming |
| `try_load_latest_save` | `true` | Auto-resume from latest checkpoint on resubmit |
| `wandb.entity` | `null` | Uses personal wandb account (not ai2-llm org) |

`global_train_batch_size` must be updated when changing node count:
```
global_train_batch_size = num_nodes × 8 GPUs × device_train_microbatch_size
e.g. 4 nodes: 4 × 8 × 4 = 128   (smoketest uses 8 to run fast)
```

---

# 6. Smoketest (before every full run)

Runs 100 steps on 1 node with batch size 8 to verify the full pipeline end-to-end.

```bash
# Delete old checkpoints so it starts clean (important — see note below)
rm -rf /lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/olmo1b-frontier/
mkdir -p /lustre/orion/lrn089/scratch/kerem.sahin/logs

cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training
git pull
sbatch scripts/frontier_smoketest.sh
```

> **Why delete checkpoints:** `max_duration=100` in the smoketest is an integer step count.
> If a checkpoint from a previous run exists (e.g. step110), OLMo loads it and immediately
> exits because it's already past step 100. Always start the smoketest clean.
> (The full run uses `"50e9T"` which is token-based and handles restarts correctly.)

Watch logs:
```bash
tail -f /lustre/orion/lrn089/scratch/kerem.sahin/logs/olmo1b-smoketest-JOBID.out
```

What to verify:
- Steps print every ~0.7s (GPU is being used)
- `train/CrossEntropyLoss` (masked) and `train/CrossEntropyLoss_all_tokens` (unmasked) both appear and differ
- `throughput/effective_tokens` is noticeably less than `throughput/total_tokens` (masking active)
- Prefix-matching scores printed at end: `Top-3 prefix-matching (induction) heads: ...`
- W&B sync works (see section 7)

---

# 7. W&B sync

Wandb is configured with `WANDB_MODE=offline` in both run scripts (compute nodes have no internet).
Offline run files are written to:
```
/lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/olmo1b-frontier/wandb/wandb/offline-run-*/
```

Sync to the cloud from the login node while the job runs or after it finishes:

```bash
# In a separate terminal on the login node:
module load miniforge3/23.11.0-0
conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training
bash scripts/wandb_sync.sh
```

The script syncs every 60 seconds. Each iteration uploads new chunks for in-progress runs,
and skips runs that already have a `.synced` marker (fully uploaded). W&B project
`olmo-1b-replication` and run `olmo1b-frontier` are created automatically on first sync.

To sync a specific run manually:
```bash
wandb sync /lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/olmo1b-frontier/wandb/wandb/offline-run-YYYYMMDD_HHMMSS-RUNID
```

---

# 8. Full training run (4 nodes, 50B tokens)

```bash
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training
git pull
sbatch scripts/frontier_run.sh
```

Monitor:
```bash
tail -f /lustre/orion/lrn089/scratch/kerem.sahin/logs/olmo1b-train-JOBID.out
bash scripts/wandb_sync.sh   # in a separate terminal
```

**Resuming after job ends:** just resubmit `sbatch scripts/frontier_run.sh`. The trainer
automatically loads the latest checkpoint and continues toward the 50B token budget.

**Scaling to more nodes:** change `-N` in `frontier_run.sh` and update
`global_train_batch_size` in `configs/olmo1b-frontier.yaml` accordingly.

---

# 9. Metrics logged to W&B

**Every training step:**
- `train/CrossEntropyLoss` — masked loss (actual training signal)
- `train/CrossEntropyLoss_all_tokens` — unmasked loss (monitoring only, no gradient)
- `train/Perplexity`, `train/Perplexity_all_tokens`
- `train/lr`, `train/global_grad_norm`
- `throughput/total_tokens`, `throughput/effective_tokens`
- `throughput/device/tokens_per_second`
- `System/Peak GPU Memory (MB)`

**Every eval step** (every 1000 training steps):
- `{label}/CrossEntropyLoss`, `{label}/CrossEntropyLoss_all_tokens`
- `eval/prefix_matching_top{1,2,3}_{score,layer,head}` — induction head diagnostic

**W&B x-axes:**
- `train/global_train_tokens_seen` — all tokens seen (masked + unmasked), governs budget
- `train/global_effective_tokens_seen` — non-masked tokens only

---

# 10. Repeated-token loss masking

`data.mask_repeated_tokens: offset` excludes positions where an induction head could trivially
copy from the loss. Given sequence `abcdefah` (second `a` at index 6 repeats index 0):

- `repeat`: masks the repeated token itself (index 6)
- `offset`: masks the prediction right after the repeat (index 7, `h`) — **this is what we use**
- `both`: masks both

The masked loss is the training signal. The unmasked loss is computed in a separate
`torch.no_grad()` pass using the same logits (no extra forward pass) and logged for monitoring.

---

# 11. Non-Frontier local setup (development / debugging)

```bash
conda create -n olmo_pretraining python=3.11
conda activate olmo_pretraining
pip install -e .[all]

# Fix PyTorch version if torch.cuda.is_available() is False:
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"  # must be True
```

Generate a small local dataset (requires internet to HuggingFace):
```bash
python scripts/create_sample_data.py \
    --output /disk/u/kerem.sahin/data/olmo1b_sample.npy \
    --num-docs 10000
```

Run a single-GPU smoke test locally:
```bash
conda run --no-capture-output -n olmo_pretraining python -m torch.distributed.run \
    --nproc_per_node=1 scripts/train.py configs/olmo1b-sample-run.yaml
```
