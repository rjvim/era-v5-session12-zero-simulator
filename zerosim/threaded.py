"""Run the 32 ranks on real CPU threads.

The assignment allows "CPU threads or Colab GPU". The engine's default path
runs ranks sequentially, which is fine for correctness but makes "32 virtual
GPUs" a bit of a claim about data structures rather than execution.

This module actually dispatches the 32 per-rank forward/backward passes to a
thread pool, with a real barrier at each collective -- so the ranks genuinely
proceed in lockstep the way SPMD ranks do, and the barrier is where you can
*see* that a collective is a synchronisation point and not just a function
call.

Honest caveat, stated because it matters: NumPy releases the GIL inside large
BLAS calls but not around the small Python-level tape operations that dominate
our tiny model, so this gives limited wall-clock speedup. It is included
because it makes the execution model faithful, not because it is fast. The
timing numbers in the report come from the analytical model in compute.py,
which is where the realistic-scale reasoning lives.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .model import grad_flat


class RankBarrier:
    """A real barrier, so collectives are visible as synchronisation points."""

    def __init__(self, n: int):
        self.n = n
        self._b = threading.Barrier(n)
        self.waits = 0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            self.waits += 1
        self._b.wait()


def threaded_fwd_bwd(cfg, params_per_rank, batches, max_workers: int = 8,
                     barrier: RankBarrier | None = None):
    """Run all ranks' forward+backward concurrently on CPU threads.

    Returns (losses, grads) in rank order -- identical to the sequential path,
    which tests/test_threaded.py asserts.
    """
    n = len(batches)
    losses = [None] * n
    grads = [None] * n

    def run(r):
        x, y = batches[r]
        if barrier is not None:
            barrier.wait()          # all ranks enter the step together
        loss, g = grad_flat(cfg, params_per_rank[r], x, y)
        losses[r] = loss
        grads[r] = g
        if barrier is not None:
            barrier.wait()          # all ranks reach the collective together
        return r

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as ex:
        list(ex.map(run, range(n)))
    return losses, grads
