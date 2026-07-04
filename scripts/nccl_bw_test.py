"""
Standalone distributed collective-bandwidth benchmark for diagnosing the
inter-node RCCL/NCCL transport on Frontier (or any multi-node GPU cluster).

It runs the two collectives that dominate FSDP training — all-reduce (gradient
reduction) and all-gather (parameter gathering) — and reports bandwidth. This
isolates pure network performance from the model, so each run takes seconds.

Launch under srun via torch.distributed.run, e.g. inside a 2-node salloc; see
scripts/frontier_nccl_test.sh for the wrapper.

Set NCCL_DEBUG=INFO (and NCCL_DEBUG_SUBSYS=INIT,NET) to see whether RCCL selects
NET/OFI (fast, libfabric/Slingshot) or NET/Socket (slow TCP fallback) — grep the
log for 'NET/'.

Interpreting busbw (bus bandwidth) for inter-node all-reduce:
    - NET/OFI over Slingshot: tens of GB/s          -> healthy
    - NET/Socket TCP fallback: ~1-3 GB/s or worse   -> the multi-minute/step cause
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size-mib", type=int, default=1024, help="tensor size per collective, MiB")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    bytes_per = 2 if dtype == torch.bfloat16 else 4
    numel = args.size_mib * 1024 * 1024 // bytes_per
    size_bytes = numel * bytes_per

    def bench(fn) -> float:
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        return (time.perf_counter() - t0) / args.iters

    # all-reduce (gradient reduction in FSDP/DDP)
    x = torch.ones(numel, dtype=dtype, device=device)
    t_ar = bench(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM))

    # all-gather (FSDP parameter gather): each rank contributes numel//world
    shard = numel // world
    xin = torch.ones(shard, dtype=dtype, device=device)
    xout = torch.empty(shard * world, dtype=dtype, device=device)
    t_ag = bench(lambda: dist.all_gather_into_tensor(xout, xin))

    if rank == 0:
        s = size_bytes
        ar_algbw = s / t_ar / 1e9
        ar_busbw = ar_algbw * 2 * (world - 1) / world
        ag_algbw = s / t_ag / 1e9
        ag_busbw = ag_algbw * (world - 1) / world
        print("=" * 64)
        print(f"world_size={world}  tensor={args.size_mib} MiB  dtype={args.dtype}  iters={args.iters}")
        print(f"all_reduce : {t_ar * 1e3:8.2f} ms/iter   algbw {ar_algbw:6.1f} GB/s   busbw {ar_busbw:6.1f} GB/s")
        print(f"all_gather : {t_ag * 1e3:8.2f} ms/iter   algbw {ag_algbw:6.1f} GB/s   busbw {ag_busbw:6.1f} GB/s")
        print("=" * 64)
        print("Single-digit busbw + NET/Socket in the log => TCP fallback; load the RCCL/OFI plugin.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
