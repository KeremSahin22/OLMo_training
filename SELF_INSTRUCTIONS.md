# Create Conda Environment

1. ```conda create -n olmo_pretraining python=3.11```

2. ```conda activate olmo_pretraining```

3. ```python -m pip install -e .[all]```

# Select Config

We will be using ```configs/official-0724/OLMo-1B.yaml```.

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
