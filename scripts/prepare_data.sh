#!/bin/bash
# Download OLMo-mix shards to local Lustre scratch and update the frontier config.
# Replace <username> with your OLCF username, then submit with:
#   sbatch scripts/prepare_data.sh
#
#SBATCH -A lrn089
#SBATCH -J prepare-data
#SBATCH -o /lustre/orion/lrn089/scratch/<username>/logs/prepare-data-%j.out
#SBATCH -e /lustre/orion/lrn089/scratch/<username>/logs/prepare-data-%j.err
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch

set -euo pipefail

module load miniforge3/23.11.0-0
conda activate /ccs/home/<username>/.conda/envs/olmo_pretraining

REPO=/lustre/orion/lrn089/scratch/<username>/OLMo_training
cd "$REPO"

mkdir -p /lustre/orion/lrn089/scratch/<username>/data/olmo-mix-50b
mkdir -p /lustre/orion/lrn089/scratch/<username>/logs

python scripts/download_shards.py \
    --shard-list configs/data/olmo-mix-50b-shards.yaml \
    --output-dir /lustre/orion/lrn089/scratch/<username>/data/olmo-mix-50b \
    --update-config configs/olmo1b-frontier.yaml

echo "Done. configs/olmo1b-frontier.yaml data.paths now point at local Lustre files."
