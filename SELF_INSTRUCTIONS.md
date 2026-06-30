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
