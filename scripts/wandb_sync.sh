#!/bin/bash
# Run from login node to sync wandb offline runs while a job is active.
# Usage: bash scripts/wandb_sync.sh [interval_seconds]
#   Default interval: 60 seconds

module load miniforge3/23.11.0-0
conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining

WANDB_DIR=/lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training/wandb
INTERVAL=${1:-60}

echo "Syncing $WANDB_DIR every ${INTERVAL}s. Ctrl-C to stop."

while true; do
    echo "[$(date '+%H:%M:%S')] Syncing..."
    wandb sync --sync-all "$WANDB_DIR"
    sleep "$INTERVAL"
done
