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
#   SHARDING=HYBRID_SHARD bash scripts/frontier_smoketest_scaling.sh   # sharded within node
#   SHARDING=HYBRID_SHARD MBS=8 bash scripts/frontier_smoketest_scaling.sh  # + bigger microbatch
#   STRATEGY=ddp          bash scripts/frontier_smoketest_scaling.sh   # NO sharding (model fits)
#   STRATEGY=ddp MBS=8    bash scripts/frontier_smoketest_scaling.sh   # DDP + bigger microbatch (~54 GB)
#
# Compare the 'throughput/device/tokens_per_second' line (steady-state, after ~step 30)
# across runs. Higher = better. Watch peak GPU memory too (stays under 64 GB).

module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a rccl-net-plugin/1.0
conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining
cd /lustre/orion/lrn089/scratch/kerem.sahin/OLMo_training

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MPICH_GPU_SUPPORT_ENABLED=1
export WANDB_MODE=offline

# ---- knobs (override on the command line) ----
STRATEGY=${STRATEGY:-fsdp}          # fsdp | ddp   (ddp = no sharding, full model per GCD)
SHARDING=${SHARDING:-FULL_SHARD}    # FULL_SHARD | HYBRID_SHARD  (only used when STRATEGY=fsdp)
MBS=${MBS:-4}                       # device_train_microbatch_size
STEPS=${STEPS:-80}                  # enough to reach steady-state past warmup
GBS=$(( SLURM_NNODES * 8 * MBS ))   # keep grad-accum realistic for the node count

# Strategy-specific flags + a short tag used for the log/checkpoint folder names.
if [ "$STRATEGY" = "ddp" ]; then
    # DDP builds the full model directly on GPU, so it needs init_device=cuda
    # (the yaml uses meta, which is required by FSDP but rejected by DDP).
    STRAT_ARGS=(--distributed_strategy=ddp --ddp.grad_sync_mode=batch --ddp.find_unused_params=false --model.init_device=cuda)
    TAG="ddp-mbs${MBS}"
else
    STRAT_ARGS=(--distributed_strategy=fsdp --fsdp.sharding_strategy=${SHARDING})
    TAG="${SHARDING}-mbs${MBS}"
fi

echo ">>> strategy=$STRATEGY  $( [ "$STRATEGY" = fsdp ] && echo "sharding=$SHARDING" )  microbatch=$MBS  nodes=$SLURM_NNODES  global_batch=$GBS  steps=$STEPS"

# Persist output so throughput numbers survive after the terminal scrolls (wandb is off here).
LOGDIR=/lustre/orion/lrn089/scratch/kerem.sahin/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/smoke-${TAG}-$(date +%H%M%S).log"
echo ">>> logging to $LOG"

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
    --run_name=smoke-${TAG} \
    --max_duration=${STEPS} \
    --global_train_batch_size=${GBS} \
    --device_train_microbatch_size=${MBS} \
    "${STRAT_ARGS[@]}" \
    --try_load_latest_save=false \
    --save_folder=/lustre/orion/lrn089/scratch/kerem.sahin/checkpoints/smoke-${TAG} \
    --save_overwrite=true \
    --wandb=null 2>&1 | tee "$LOG"

echo ""
echo ">>> throughput lines for $TAG (read the steady-state value past ~step 30):"
grep -i "tokens_per_second" "$LOG" | tail -20
