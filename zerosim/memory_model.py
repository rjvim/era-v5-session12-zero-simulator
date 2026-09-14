"""Closed-form ZeRO memory arithmetic (Rajbhandari et al., 2020).

Mixed-precision Adam, per parameter:
    fp16 weights        2 bytes   \\
    fp16 gradients      2 bytes    |  the "4*Psi" that DP replicates
    fp32 master copy    4 bytes   \\
    fp32 momentum       4 bytes    |  K = 12, the optimizer states
    fp32 variance       4 bytes   /
                       --------
                       16 bytes per parameter, replicated on every GPU in
                       plain data parallelism.

Sharded across N GPUs:
    DP      : 4*Psi + K*Psi
    ZeRO-1  : 4*Psi + K*Psi/N
    ZeRO-2  : 2*Psi + (2*Psi + K*Psi)/N
    ZeRO-3  : (4*Psi + K*Psi)/N
"""
from __future__ import annotations

from dataclasses import dataclass

GB = 1024 ** 3

# byte widths under mixed precision
FP16 = 2
FP32 = 4
K_ADAM = 12  # fp32 master + momentum + variance


@dataclass(frozen=True)
class MemBreakdown:
    fp16_params: float
    fp16_grads: float
    optimizer: float

    @property
    def total(self) -> float:
        return self.fp16_params + self.fp16_grads + self.optimizer

    def as_dict(self):
        return {
            "fp16_params_bytes": self.fp16_params,
            "fp16_grads_bytes": self.fp16_grads,
            "optimizer_bytes": self.optimizer,
            "total_bytes": self.total,
        }


def model_state_bytes(psi: float, world: int, stage: int, k: int = K_ADAM) -> MemBreakdown:
    """Per-GPU model-state bytes for `psi` parameters over `world` GPUs."""
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"stage must be 0..3, got {stage}")
    p = FP16 * psi
    g = FP16 * psi
    o = k * psi
    if stage == 0:
        return MemBreakdown(p, g, o)
    if stage == 1:
        return MemBreakdown(p, g, o / world)
    if stage == 2:
        return MemBreakdown(p, g / world, o / world)
    return MemBreakdown(p / world, g / world, o / world)


def model_state_gb(psi: float, world: int, stage: int, k: int = K_ADAM) -> float:
    return model_state_bytes(psi, world, stage, k).total / 1e9


def comm_volume_psi(stage: int) -> float:
    """Per-GPU communication per optimizer step in multiples of Psi ELEMENTS.

    Units matter here and are easy to get wrong: the ZeRO paper states volume
    in elements, so under fp16 the byte count is twice this. Use
    comm_volume_bytes() if you want bytes.

    DP / ZeRO-1 / ZeRO-2 all cost one all-reduce equivalent = 2*Psi.
    ZeRO-3 adds a second all-gather of the weights (forward AND backward),
    giving 3*Psi -- the 1.5x that buys you the memory.
    """
    return {0: 2.0, 1: 2.0, 2: 2.0, 3: 3.0}[stage]


def comm_volume_bytes(psi: float, stage: int, dtype_bytes: int = FP16) -> float:
    """Per-GPU BYTES moved per optimizer step (ring-ideal, N -> inf)."""
    return comm_volume_psi(stage) * psi * dtype_bytes


def fits(psi: float, world: int, stage: int, gpu_bytes: float, k: int = K_ADAM) -> bool:
    return model_state_bytes(psi, world, stage, k).total <= gpu_bytes


def max_params(world: int, stage: int, gpu_bytes: float, k: int = K_ADAM) -> float:
    """Largest model (in parameters) whose *model states* fit one GPU."""
    per_param = model_state_bytes(1.0, world, stage, k).total
    return gpu_bytes / per_param
