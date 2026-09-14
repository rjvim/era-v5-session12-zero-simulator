"""A virtual GPU: a memory ledger plus whatever buffers we hand it.

The point of the assignment is to *watch memory move*, so every allocation is
named and the peak is tracked. Bytes are modelled (fp16/fp32 as a real run
would store them), not measured from the float64 NumPy arrays we compute with.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Alloc:
    name: str
    bytes: int
    category: str  # params | grads | optimizer | activations | transient


class VirtualGPU:
    """One fake accelerator. Holds tensors and accounts for their bytes."""

    def __init__(self, rank: int, node: int, capacity_bytes: int | None = None):
        self.rank = rank
        self.node = node
        self.capacity_bytes = capacity_bytes
        self.store: dict[str, object] = {}
        self._allocs: dict[str, Alloc] = {}
        self.peak_bytes = 0
        self.peak_breakdown: dict[str, int] = {}
        self.oom = False

    # ---- storage ----
    def put(self, name: str, value, nbytes: int, category: str):
        self.store[name] = value
        self._allocs[name] = Alloc(name, int(nbytes), category)
        self._touch()

    def get(self, name: str):
        return self.store[name]

    def free(self, name: str):
        self.store.pop(name, None)
        self._allocs.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self.store

    # ---- accounting ----
    @property
    def live_bytes(self) -> int:
        return sum(a.bytes for a in self._allocs.values())

    def breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self._allocs.values():
            out[a.category] = out.get(a.category, 0) + a.bytes
        return out

    def _touch(self):
        live = self.live_bytes
        if live > self.peak_bytes:
            self.peak_bytes = live
            self.peak_breakdown = self.breakdown()
        if self.capacity_bytes is not None and live > self.capacity_bytes:
            self.oom = True

    def reset_peak(self):
        self.peak_bytes = self.live_bytes
        self.peak_breakdown = self.breakdown()
        self.oom = False


@dataclass
class Cluster:
    """`world_size` virtual GPUs arranged into nodes of `gpus_per_node`."""

    world_size: int = 32
    gpus_per_node: int = 8
    capacity_bytes: int | None = None
    gpus: list = field(default_factory=list)

    def __post_init__(self):
        if not self.gpus:
            self.gpus = [
                VirtualGPU(r, r // self.gpus_per_node, self.capacity_bytes)
                for r in range(self.world_size)
            ]

    @property
    def n_nodes(self) -> int:
        return (self.world_size + self.gpus_per_node - 1) // self.gpus_per_node

    def __iter__(self):
        return iter(self.gpus)

    def __getitem__(self, r):
        return self.gpus[r]

    def __len__(self):
        return self.world_size

    def reset_peaks(self):
        for g in self.gpus:
            g.reset_peak()

    def peak(self) -> int:
        return max(g.peak_bytes for g in self.gpus)
