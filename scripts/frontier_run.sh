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
# Compute nodes block outbound internet; log wandb offline and sync from login node after.
export WANDB_MODE=offline

# --- Distributed transport (RCCL over Slingshot) ---
# Quiet by default; submit with `NCCL_DEBUG=INFO sbatch scripts/frontier_run.sh` to diagnose.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
#
# Once validated with scripts/frontier_nccl_test.sh, enable the RCCL/OFI plugin + Slingshot
# env here for full inter-node bandwidth (uncomment and fill in the plugin path/module):
#   module load <rccl-ofi-plugin-module>
#   export LD_LIBRARY_PATH=/path/to/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
#   export NCCL_SOCKET_IFNAME=hsn0
#   export NCCL_NET_GDR_LEVEL=3
#   export NCCL_CROSS_NIC=1
#   export FI_CXI_DEFAULT_CQ_SIZE=131072
#   export FI_MR_CACHE_MONITOR=userfaultfd

# Get the IPv4 address of the first allocated node for rendezvous (avoid IPv6 issues)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_ADDR" | awk 'NR==1{print $1}')

srun -N $SLURM_NNODES --gpus-per-node=8 \
    python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    scripts/train.py configs/olmo1b-frontier.yaml
