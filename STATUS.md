# Project Status — OLMo-1B Replication on Frontier

Last updated: 2026-07-03 (W&B sync fix verified)

---

## Current state

- Environment is set up on Frontier (`olmo_pretraining` conda env on `/ccs/home/kerem.sahin/.conda/envs/`)
- All 4 OLMo-mix shards (~64.6B tokens) are downloaded to Lustre scratch
- Training config (`configs/olmo1b-frontier.yaml`) is finalized for 4-node FSDP run
- Smoketest (`scripts/frontier_smoketest.sh`) has been validated:
  - FSDP on 8 MI250X GCDs works correctly
  - `mask_repeated_tokens: offset` is active — effective tokens (~40%) visibly less than total tokens
  - Both masked (`train/CrossEntropyLoss`) and unmasked (`train/CrossEntropyLoss_all_tokens`) losses log correctly at every step
  - Prefix-matching (induction head) scores computed and printed at end of run
  - Checkpoint save and resume work correctly
  - W&B offline mode works; sync from login node via `bash scripts/wandb_sync.sh` works
- W&B credentials configured on login node (`~/.netrc`)
- `scripts/wandb_sync.sh` fixed and verified end-to-end: the glob `offline-run-*/` was passing
  `wandb sync` a path with a trailing slash, which caused it to report "nothing to sync" even on
  never-synced runs. Stripped the trailing slash (`run_dir="${run_dir%/}"`) before the sync call —
  confirmed working via a fresh `sbatch scripts/frontier_smoketest.sh` run synced automatically.

---

## Immediate next step

W&B sync is verified working. Launch the full 4-node training job:

```bash
# 1. Pull the wandb_sync.sh trailing-slash fix
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training
git pull

# 2. Start the sync loop in a separate terminal (safe to start before submitting —
#    it polls and idles harmlessly until a run directory appears)
bash scripts/wandb_sync.sh

# 3. Submit the full run
sbatch scripts/frontier_run.sh
```

---

## Future steps

### Must do before / during training

- [x] **Verify W&B** — `wandb_sync.sh` fixed (trailing-slash bug) and confirmed syncing automatically
- [ ] **Submit full 4-node run** — `sbatch scripts/frontier_run.sh`
- [ ] **Monitor for loss spikes or NaNs** — grad norm should stay below 1.0 during warmup;
      spikes after warmup are worth investigating

### Nice to have

- [ ] **Add eval datasets** — the current config has `evaluators: []`. Adding standard
      perplexity benchmarks (e.g. Pile subsets) would give cleaner eval curves and make
      the replication more comparable to the original OLMo-1B paper results.
- [ ] **Save unsharded checkpoints** for long-term storage — add back to config:
      ```yaml
      save_interval_unsharded: 10000
      save_num_unsharded_checkpoints_to_keep: -1
      ```
- [ ] **Scale warmup steps** — current config has `t_warmup: 100` (shortened for smoketest
      iteration speed). Original OLMo-1B used `t_warmup: 2000`. Update before or shortly
      after launch.
- [ ] **Resubmit job periodically** — the `extended` partition allows up to 24-hour jobs.
      Resubmitting picks up automatically from the latest checkpoint.
