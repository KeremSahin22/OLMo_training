#!/bin/bash
#SBATCH -A lrn089
#SBATCH -J olmo1b-smoketest
#SBATCH -o /lustre/orion/lrn089/scratch/kerem.sahin/logs/%x-%j.out
#SBATCH -e /lustre/orion/lrn089/scratch/kerem.sahin/logs/%x-%j.err
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:30:00
#SBATCH -p batch

module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a

conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining

cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MPICH_GPU_SUPPORT_ENABLED=1
export HF_HOME=/lustre/orion/lrn089/scratch/kerem.sahin/.cache/huggingface
# Compute nodes block outbound internet; log wandb offline and sync from login node after.
export WANDB_MODE=offline
# Nodes share Lustre — one filesystem leader for checkpoint dir management (see frontier_run.sh).
export OLMO_SHARED_FS=1

# Get the IPv4 address of the first allocated node for rendezvous (avoid IPv6 issues)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_ADDR" | awk 'NR==1{print $1}')

# The smoketest runs fully isolated from the full run: its own run_name / save_folder /
# wandb run, no auto-resume. This guarantees it always starts from a fresh model and can
# never pollute the full run's checkpoint chain (or be resumed BY the full run).
srun -N $SLURM_NNODES --gpus-per-node=8 \
    python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    scripts/train.py configs/olmo1b-frontier.yaml \
    --run_name=olmo1b-smoketest \
    --save_folder=/lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/olmo1b-smoketest \
    --try_load_latest_save=false \
    --max_duration=100 \
    --global_train_batch_size=8 \
    --save_overwrite=true
