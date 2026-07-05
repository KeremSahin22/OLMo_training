"""
Minimal repro: does torch.compile's FUSED flash-attention BACKWARD work on this
ROCm/MI250X build — with ZERO OLMo code involved?

The full-run crash was:
    assert_size_stride(getitem_1, (8, 16, 2048), (32768, 2048, 1))
    AssertionError: wrong number of dimensions
in the *compiled backward*. getitem_1 (B, n_heads, T) is the log-sum-exp that
torch's flash-attention forward saves for its backward — a torch-internal tensor,
not something OLMo constructs. This script isolates exactly that: a single
F.scaled_dot_product_attention under torch.compile, then .backward().

If this crashes with the same assertion, the bug is torch-Inductor vs AOTriton on
ROCm, independent of OLMo. If it runs clean, the problem is something OLMo does
around attention and we should keep digging.

Run on one GCD inside a short salloc:
    salloc -A lrn089 -N 1 --gpus-per-node=1 -t 00:10:00 -p batch
    module load miniforge3/23.11.0-0 rocm/6.2.4 craype-accel-amd-gfx90a
    conda activate /ccs/home/kerem.sahin/.conda/envs/olmo_pretraining
    python scripts/probe_compile_sdpa.py
"""
import torch
import torch.nn.functional as F

# Exact OLMo-1B attention shape at mbs=8: B=8, H=16, T=2048, head_dim=128
B, H, T, HD = 8, 16, 2048, 128
dev = torch.device("cuda", 0)

print(f"torch {torch.__version__}  |  device {torch.cuda.get_device_name(0)}")
try:
    import triton
    print(f"triton {triton.__version__}")
except Exception as e:
    print(f"triton import failed: {e}")
print(f"shape B={B} H={H} T={T} head_dim={HD} dtype=bf16 is_causal=True\n")


def attn(q, k, v):
    # The single call OLMo makes (no mask, causal). Sum to get a scalar to backprop.
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)


def make_inputs():
    q = torch.randn(B, H, T, HD, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, HD, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, HD, device=dev, dtype=torch.bfloat16, requires_grad=True)
    return q, k, v


# ---- 1. eager forward+backward (sanity: fused attention itself is fine) ----
q, k, v = make_inputs()
attn(q, k, v).float().sum().backward()
torch.cuda.synchronize()
print("eager  forward+backward: OK  (grad shape", tuple(q.grad.shape), ")")

# ---- 2. compiled forward+backward (this is what the training run does) ----
cattn = torch.compile(attn, mode="default")
q, k, v = make_inputs()
try:
    out = cattn(q, k, v)
    out.float().sum().backward()          # <-- crash point in the full run
    torch.cuda.synchronize()
    print("compiled forward+backward: OK  (grad shape", tuple(q.grad.shape), ")")
    print("\n=> compile+SDPA-backward works standalone. The full-run crash is NOT")
    print("   this path — dig into what OLMo does around attention.")
except Exception as e:
    print("compiled forward+backward: FAIL")
    print("   ", type(e).__name__, str(e)[:200])
    print("\n=> Reproduced with ZERO OLMo code => torch-Inductor vs AOTriton (ROCm)")
    print("   bug, not ours. Keeping compile off is the right call.")
