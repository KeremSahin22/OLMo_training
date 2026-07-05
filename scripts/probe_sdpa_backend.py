"""
Probe which torch SDPA backend is available on this (ROCm/MI250X) build.

OLMo calls F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
when flash_attention=false and no attention_mask/alibi (our config). That call is
eligible for a *fused* kernel IF the torch+ROCm build ships one (AOTriton flash /
mem-efficient) for gfx90a. This script forces each backend in turn and reports
whether it runs — telling us definitively if attention is already fast.

Run on a single GCD, e.g. inside a short salloc:
    salloc -A lrn089 -N 1 --gpus-per-node=1 -t 00:10:00 -p batch
    module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a
    conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining
    python scripts/probe_sdpa_backend.py
"""
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as sdpa

# OLMo-1B attention shape: B=microbatch, H=n_heads=16, T=2048, hd=d_model/n_heads=128
B, H, T, HD = 4, 16, 2048, 128
dev = torch.device("cuda", 0)
q = torch.randn(B, H, T, HD, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, H, T, HD, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, H, T, HD, device=dev, dtype=torch.bfloat16)

print(f"torch {torch.__version__}  |  device {torch.cuda.get_device_name(0)}")
print(f"shape B={B} H={H} T={T} head_dim={HD} dtype=bf16 is_causal=True\n")

for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
    try:
        with sdpa_kernel(be):
            for _ in range(3):  # warmup
                o = sdpa(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                o = sdpa(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1e3
        print(f"  {be.name:20s} OK    {ms:7.2f} ms/call")
    except Exception as e:
        print(f"  {be.name:20s} FAIL  {str(e)[:90]}")

# What torch picks on its own (what OLMo actually gets at runtime):
try:
    for _ in range(3):
        o = sdpa(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        o = sdpa(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"\n  {'AUTO (what OLMo gets)':20s} OK    {ms:7.2f} ms/call")
except Exception as e:
    print(f"\n  AUTO FAIL  {str(e)[:90]}")
