# Project Status — OLMo-1B Replication on Frontier

Last updated: 2026-07-03

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
- One smoketest run is pending W&B sync verification (run `offline-run-20260703_174346-e3dda5kk`)

---

## Immediate next step

Verify W&B sync works end-to-end for the latest smoketest run, then launch the full 4-node training job:

```bash
# 1. Sync the latest smoketest run to W&B and verify both losses appear in the UI
wandb sync /lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/olmo1b-frontier/wandb/wandb/offline-run-20260703_174346-e3dda5kk

# 2. If W&B looks correct, submit the full run
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training
git pull
sbatch scripts/frontier_run.sh

# 3. Start the sync loop in a separate terminal
bash scripts/wandb_sync.sh
```

---

## Future steps

### Must do before / during training

- [ ] **Verify W&B** — confirm `train/CrossEntropyLoss` and `train/CrossEntropyLoss_all_tokens`
      both appear in the UI, and that `global_effective_tokens_seen` < `global_train_tokens_seen`
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
