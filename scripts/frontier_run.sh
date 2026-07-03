#!/bin/bash
#SBATCH -A lrn089
#SBATCH -J olmo1b-train
#SBATCH -o /lustre/orion/lrn089/scratch/kerem.sahin/logs/%x-%j.out
#SBATCH -e /lustre/orion/lrn089/scratch/kerem.sahin/logs/%x-%j.err
#SBATCH -N 4                  # <--- change number of nodes here
#SBATCH --gpus-per-node=8     # always 8 on Frontier (fixed per node)
#SBATCH -t 24:00:00
#SBATCH -p extended

# Replace kerem.sahin with your OLCF username before submitting.
# Submit from the repo root: sbatch scripts/frontier_run.sh
# Create log dir first: mkdir -p /lustre/orion/lrn089/scratch/kerem.sahin/logs
#
# When changing -N above, also update global_train_batch_size in olmo1b-frontier.yaml:
#   global_train_batch_size = N_nodes * 8 * device_train_microbatch_size
#   e.g. 4 nodes: 4 * 8 * 4 = 128

module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a

conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining

# Run from Lustre, not home — NFS is too slow for training I/O
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MPICH_GPU_SUPPORT_ENABLED=1
export HF_HOME=/lustre/orion/lrn089/scratch/kerem.sahin/.cache/huggingface

# Get the hostname of the first allocated node for rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)

srun -N $SLURM_NNODES --gpus-per-node=8 \
    python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    scripts/train.py configs/olmo1b-frontier.yaml
