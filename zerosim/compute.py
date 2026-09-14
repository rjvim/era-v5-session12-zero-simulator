"""Compute accounting.

The single most important thing to state up front, because it took me a while
to accept it: **ZeRO does not change the FLOPs at all.** Every stage runs the
identical mathematics on the identical data -- that is exactly what the
bit-identical weights in tests/test_equivalence.py prove.

So when the assignment says "show how computation changes", the answer is NOT
that the arithmetic gets cheaper. It is that *achieved throughput* changes,
because communication sits in the critical path and ZeRO-3 puts more of it
there. FLOPs stay flat; tokens/second does not.

That distinction is the whole content of this file.
"""
from __future__ import annotations

from dataclasses import dataclass

from .memory_model import FP16, comm_volume_bytes


def flops_per_token(psi: float, n_layer: int, d_model: int, seq_len: int,
                    include_attention: bool = True) -> float:
    """Training FLOPs per token: 6*Psi (fwd 2 + bwd 4), plus attention.

    The 6*Psi term is the standard estimate. The attention term
    (12 * n_layer * d_model * seq_len) matters once sequences get long -- at
    seq=8192 on a 30B model it is far from negligible, which is why it is here
    rather than dropped.
    """
    f = 6.0 * psi
    if include_attention:
        f += 12.0 * n_layer * d_model * seq_len
    return f


@dataclass(frozen=True)
class GPUProfile:
    name: str
    bf16_tflops: float      # realistic sustained, not peak marketing number
    fp8_tflops: float
    hbm_gb: float
    intra_gbps: float
    inter_gbps: float


# Sustained, not peak. Peak numbers assume perfect utilisation nobody gets.
H100_PROFILE = GPUProfile("H100", bf16_tflops=400.0, fp8_tflops=560.0,
                          hbm_gb=80.0, intra_gbps=450.0, inter_gbps=45.0)
B200_PROFILE = GPUProfile("B200", bf16_tflops=1100.0, fp8_tflops=1550.0,
                          hbm_gb=192.0, intra_gbps=900.0, inter_gbps=100.0)


def step_time(psi: float, stage: int, world: int, profile: GPUProfile,
              tokens_per_gpu: int, n_layer: int, d_model: int, seq_len: int,
              nodes: int = 1, overlap: float = 0.0, fp8: bool = False) -> dict:
    """Time for one optimizer step on one GPU.

    overlap: fraction of communication hidden behind compute (0 = none,
    1 = perfectly hidden). Real DeepSpeed/FSDP land somewhere in between, and
    ZeRO-3 overlaps worse than ZeRO-1/2 because its all-gathers are on the
    critical path of the forward and backward passes themselves, not just the
    optimizer step. That asymmetry is modelled below.
    """
    f = flops_per_token(psi, n_layer, d_model, seq_len) * tokens_per_gpu
    tflops = profile.fp8_tflops if fp8 else profile.bf16_tflops
    compute_s = f / (tflops * 1e12)

    bw = profile.intra_gbps if nodes == 1 else profile.inter_gbps
    ring = (world - 1) / world if world > 1 else 0.0
    comm_bytes = comm_volume_bytes(psi, stage) * ring
    comm_s = comm_bytes / (bw * 1e9)

    # ZeRO-3's weight all-gathers block the forward/backward themselves, so a
    # smaller share of them can hide behind compute.
    eff_overlap = overlap * (0.5 if stage == 3 else 1.0)
    exposed_s = comm_s * (1.0 - eff_overlap)

    total = compute_s + exposed_s
    return {
        "flops_per_gpu": f,
        "compute_seconds": compute_s,
        "comm_bytes_per_gpu": comm_bytes,
        "comm_seconds_raw": comm_s,
        "comm_seconds_exposed": exposed_s,
        "step_seconds": total,
        "tokens_per_second_per_gpu": tokens_per_gpu / total,
        "tokens_per_second_cluster": tokens_per_gpu * world / total,
        # Deliberately NOT called MFU. This is compute_s/total -- the share of
        # the step not lost to exposed communication. It says nothing about
        # kernel efficiency, and since compute_s was itself derived from
        # `tflops`, calling it MFU would be circular: it would report ~1.0 for
        # a perfectly overlapped run no matter how bad the kernels were. Real
        # MFU needs measured wall-clock against a hardware ceiling.
        "compute_fraction_of_step": compute_s / total,
        "comm_fraction_of_step": exposed_s / total,
    }
