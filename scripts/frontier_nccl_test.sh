#!/bin/bash
# Inter-node RCCL bandwidth test — the fast iteration loop for the throughput fix.
#
# Grab a small interactive allocation ONCE (schedules fast; iterate inside it):
#   salloc -A lrn089 -N 2 --gpus-per-node=8 -t 01:00:00 -p batch
# then repeatedly:
#   bash scripts/frontier_nccl_test.sh
# Edit the NETWORK ENV block below between runs to compare transports (seconds each).
#
# Look for two things in the output:
#   1. 'NET/OFI' (good, Slingshot) vs 'NET/Socket' (bad, TCP fallback)
#   2. the all_reduce / all_gather busbw line — tens of GB/s = fixed, single digits = still broken

module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a
conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MPICH_GPU_SUPPORT_ENABLED=1

# Show the transport RCCL selects, limited to init+net so the log stays readable.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# ===================== NETWORK ENV (toggle to test) =====================
# Step A — load the RCCL/libfabric plugin once you locate it (one of these):
#   module load <rccl-ofi-plugin-module>
#   export LD_LIBRARY_PATH=/path/to/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
#
# Step B — Frontier Slingshot/CXI env (safe once the plugin above is active):
# export NCCL_SOCKET_IFNAME=hsn0
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_CROSS_NIC=1
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export FI_MR_CACHE_MONITOR=userfaultfd
# ========================================================================

# IPv4 rendezvous address of the first node in the allocation (avoid IPv6 issues).
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_ADDR" | awk 'NR==1{print $1}')

srun -N "$SLURM_NNODES" --gpus-per-node=8 \
    python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes="$SLURM_NNODES" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:29500" \
    scripts/nccl_bw_test.py --size-mib 1024 --iters 20
