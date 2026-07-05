#!/bin/bash
# 2-node THROUGHPUT smoke test — find the right sharding strategy + microbatch size
# WITHOUT booking the full 4-node run and burning GPU hours.
#
# Why 2 nodes: a single node can't reveal the inter-node comm cost, and 2-node hybrid_shard
# shards 8-way *within* a node exactly like the 4-node run would — so per-GCD memory and
# throughput here faithfully predict the 4-node run.
#
# Grab ONE allocation and iterate inside it (each variant is ~2-3 min):
#   salloc -A lrn089 -N 2 --gpus-per-node=8 -t 01:00:00 -p batch
# then, from the repo root, run variants back-to-back:
#   SHARDING=FULL_SHARD   bash scripts/frontier_smoketest_scaling.sh   # baseline (current config)
#   SHARDING=HYBRID_SHARD bash scripts/frontier_smoketest_scaling.sh   # the proposed fix
#   SHARDING=HYBRID_SHARD MBS=8 bash scripts/frontier_smoketest_scaling.sh  # + bigger microbatch
#
# Compare the 'throughput/device/tokens_per_second' line (steady-state, after ~step 30)
# across runs. Higher = better. Watch peak GPU memory too (stays well under 64 GB).

module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a rccl-net-plugin/1.0
conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MPICH_GPU_SUPPORT_ENABLED=1
export WANDB_MODE=offline

# ---- knobs (override on the command line) ----
SHARDING=${SHARDING:-FULL_SHARD}   # FULL_SHARD | HYBRID_SHARD
MBS=${MBS:-4}                       # device_train_microbatch_size
STEPS=${STEPS:-80}                  # enough to reach steady-state past warmup
GBS=$(( SLURM_NNODES * 8 * MBS ))   # keep grad-accum realistic for the node count

echo ">>> sharding=$SHARDING  microbatch=$MBS  nodes=$SLURM_NNODES  global_batch=$GBS  steps=$STEPS"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_ADDR" | awk 'NR==1{print $1}')

srun -N "$SLURM_NNODES" --gpus-per-node=8 \
    python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes="$SLURM_NNODES" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:29500" \
    scripts/train.py configs/olmo1b-frontier.yaml \
    --run_name=smoke-${SHARDING}-mbs${MBS} \
    --max_duration=${STEPS} \
    --global_train_batch_size=${GBS} \
    --device_train_microbatch_size=${MBS} \
    --fsdp.sharding_strategy=${SHARDING} \
    --save_folder=/lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/smoke \
    --save_overwrite=true \
    --wandb=null
