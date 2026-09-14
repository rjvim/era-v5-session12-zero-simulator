"""The interconnect.

Implements the three collectives the session is built around, and charges for
them. Volume uses the standard ring costs: a ring all-gather or reduce-scatter
moves (N-1)/N * S bytes per GPU, and all-reduce = reduce-scatter + all-gather,
so 2*(N-1)/N * S. Those are the numbers that make ZeRO-3's 1.5x communication
fall out rather than being asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Link:
    """Bandwidth profile. Values are per-GPU unidirectional bus bandwidth."""

    name: str
    intra_node_gbps: float   # GB/s inside one machine (NVLink)
    inter_node_gbps: float   # GB/s between machines (InfiniBand)


# Session 12 quoted ~450 GB/s intra-node and roughly 10x slower between nodes
# (120 GB in ~0.27 s on NVLink vs ~2.4 s on InfiniBand).
H100 = Link("H100/NVLink+IB", intra_node_gbps=450.0, inter_node_gbps=45.0)
B200 = Link("B200/NVLink5+IB", intra_node_gbps=900.0, inter_node_gbps=100.0)


@dataclass
class CommStats:
    bytes_on_wire: int = 0          # summed per-GPU bytes moved
    seconds: float = 0.0
    by_op: dict = field(default_factory=dict)

    def add(self, op: str, per_gpu_bytes: float, seconds: float, world: int):
        self.bytes_on_wire += int(per_gpu_bytes * world)
        self.seconds += seconds
        d = self.by_op.setdefault(op, {"per_gpu_bytes": 0.0, "seconds": 0.0, "calls": 0})
        d["per_gpu_bytes"] += per_gpu_bytes
        d["seconds"] += seconds
        d["calls"] += 1

    def as_dict(self):
        return {
            "bytes_on_wire": self.bytes_on_wire,
            "seconds": self.seconds,
            "by_op": {k: dict(v) for k, v in self.by_op.items()},
        }


class Fabric:
    """Runs collectives over a Cluster and records what they cost."""

    def __init__(self, cluster, link: Link = H100, dtype_bytes: int = 2):
        self.cluster = cluster
        self.link = link
        self.dtype_bytes = dtype_bytes
        self.stats = CommStats()

    # ---- cost model ----
    def _bandwidth(self) -> float:
        """Slowest link in play. Any collective spanning >1 node is IB-bound."""
        if self.cluster.n_nodes > 1:
            return self.link.inter_node_gbps
        return self.link.intra_node_gbps

    def _charge(self, op: str, payload_bytes: float, factor: float):
        """factor: 1 for gather/scatter (ring), 2 for all-reduce."""
        N = len(self.cluster)
        if N <= 1:
            return
        per_gpu = factor * (N - 1) / N * payload_bytes
        secs = per_gpu / (self._bandwidth() * 1e9)
        self.stats.add(op, per_gpu, secs, N)

    # ---- collectives ----
    def all_gather(self, shards: list[np.ndarray]) -> np.ndarray:
        """Each rank contributes its slice; everyone ends with the full buffer."""
        full = np.concatenate(shards)
        self._charge("all_gather", full.size * self.dtype_bytes, 1.0)
        return full

    def reduce_scatter(self, fulls: list[np.ndarray], op: str = "mean") -> list[np.ndarray]:
        """Everyone contributes a full buffer; each rank keeps one reduced slice."""
        N = len(fulls)
        stacked = np.stack(fulls)
        red = stacked.mean(0) if op == "mean" else stacked.sum(0)
        self._charge("reduce_scatter", red.size * self.dtype_bytes, 1.0)
        return list(np.array_split(red, N))

    def all_reduce(self, fulls: list[np.ndarray], op: str = "mean") -> list[np.ndarray]:
        """Everyone contributes and everyone receives the same reduced buffer."""
        stacked = np.stack(fulls)
        red = stacked.mean(0) if op == "mean" else stacked.sum(0)
        self._charge("all_reduce", red.size * self.dtype_bytes, 2.0)
        return [red.copy() for _ in fulls]

    def broadcast(self, buf: np.ndarray, src: int = 0) -> list[np.ndarray]:
        self._charge("broadcast", buf.size * self.dtype_bytes, 1.0)
        return [buf.copy() for _ in range(len(self.cluster))]

    def reset(self):
        self.stats = CommStats()


def shard_sizes(n: int, world: int) -> list[int]:
    """How np.array_split divides n elements across `world` ranks."""
    base, rem = divmod(n, world)
    return [base + (1 if r < rem else 0) for r in range(world)]


def shard_bounds(n: int, world: int) -> list[tuple[int, int]]:
    out, off = [], 0
    for s in shard_sizes(n, world):
        out.append((off, off + s))
        off += s
    return out
