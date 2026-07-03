#!/bin/bash
# Run from login node to sync wandb offline runs while a job is active.
# Usage: bash scripts/wandb_sync.sh [interval_seconds]
#   Default interval: 60 seconds

module load miniforge3/23.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining

WANDB_DIR=/lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/olmo1b-frontier/wandb/wandb
INTERVAL=${1:-60}

echo "Syncing $WANDB_DIR every ${INTERVAL}s. Ctrl-C to stop."

while true; do
    echo "[$(date '+%H:%M:%S')] Syncing..."
    for run_dir in "$WANDB_DIR"/offline-run-*/; do
        run_dir="${run_dir%/}"
        if [ -d "$run_dir" ] && [ ! -f "${run_dir}.synced" ]; then
            wandb sync "$run_dir"
        fi
    done
    sleep "$INTERVAL"
done
