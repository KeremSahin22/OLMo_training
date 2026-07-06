# Project Status — OLMo-1B Replication on Frontier

Last updated: 2026-07-06 (full run crashed AGAIN at step-1100 checkpoint save on 8 nodes — 10s wait_for timeout too tight for cross-node Lustre visibility; bumped to 120s)

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
- **Smoketest isolated from the full run (2026-07-06):** `frontier_smoketest.sh` previously wrote
  checkpoints into the full run's `save_folder` (`checkpoints/olmo1b-frontier-full/`), so the full
  run's `try_load_latest_save` would silently resume from smoketest weights (100 steps at global
  batch 8), and a smoketest launched mid-run could rotate out a real checkpoint
  (`save_num_checkpoints_to_keep: 2`). The smoketest now uses its own
  `run_name`/`save_folder` (`olmo1b-smoketest`) with `try_load_latest_save=false` — safe to run
  any time, no checkpoint cleanup needed before it.

- **Full run crashed at the step-200 checkpoint save (2026-07-06, job 4944229):** the error
  `Checkpoint for step 200 already exists, use --save_overwrite` was NOT a real pre-existing
  checkpoint. Without `OLMO_SHARED_FS=1`, `get_fs_local_rank()` falls back to the node-local
  rank, so all 4 node leaders (global ranks 0/8/16/24) raced the `step200-tmp -> step200`
  rename on shared Lustre; a losing rank got EEXIST (surfaced as `FileExistsError`, wrapped
  into the misleading message) and took the job down. Step 100 saved fine only by timing luck.
  Fixed two ways: `export OLMO_SHARED_FS=1` in all Frontier run scripts (single filesystem
  leader, matching upstream OLMo's LUMI setup), and `_temporary_wd` in `olmo/checkpoint.py`
  now tolerates ENOENT/EEXIST/ENOTEMPTY from a losing rename when the final dir exists.

- **Full run crashed AGAIN at the step-1100 checkpoint save (2026-07-06, first 8-node job):**
  `TimeoutError: Waiting for checkpoint directory timed out`, from `_temporary_wd` in
  `olmo/checkpoint.py`. This is the flip side of the `OLMO_SHARED_FS=1` fix above: with it set,
  only true global rank 0 (on one node) creates `step1100-tmp`; every other rank across all 8
  nodes just polls for it to appear via `wait_for(..., timeout=10.0)`. All 8 ranks on one
  non-leader node timed out — cross-node Lustre metadata visibility took longer than 10s under
  the I/O load of 64 ranks flushing sharded checkpoint state at once (this got measurably worse
  going from 4 to 8 nodes: more concurrent writers hitting the same metadata server). Not a race
  or a real bug, just too tight an SLA. Fixed by bumping both `wait_for` timeouts in
  `_temporary_wd` (`olmo/checkpoint.py`, tmp-dir-visible and final-rename-visible waits) from
  10s to 120s — this only gates on filesystem propagation with a `barrier()` right after, so a
  longer timeout costs nothing in the common case.

---

## Immediate next step

Recover the crashed full run and resubmit:

```bash
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training

# 1. Pull the fix (wait_for timeout bumped 10s -> 120s in _temporary_wd)
git pull

# 2. The step1100 save failed WHILE creating the tmp dir, before any writing happened, so
#    there is no partial step1100 checkpoint to worry about (unlike the earlier step200
#    incident). Just remove the empty/partial step1100-tmp leftover:
ls /lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/olmo1b-frontier-full/
rm -rf /lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/olmo1b-frontier-full/step1100-tmp
#    try_load_latest_save will fall back to step1000 (the last successful save).

# 3. Start the sync loop in a separate terminal (safe to start before submitting)
bash scripts/wandb_sync.sh

# 4. Resubmit — try_load_latest_save resumes from the latest intact checkpoint
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
