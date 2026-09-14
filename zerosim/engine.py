"""ZeRO stages 0-3, actually executed over virtual GPUs.

This is not a memory calculator with a training loop bolted on. Each rank
really holds only its own slice, the full parameter vector is really
reconstructed by all-gather before it can be used, and the gradient really
travels through reduce-scatter. If the sharding logic were wrong the
cross-stage equivalence test in tests/test_equivalence.py would fail.

Stage semantics:
  0  plain data parallelism -- everything replicated, gradients all-reduced
  1  optimizer states sharded
  2  + gradients sharded (all-reduce becomes reduce-scatter + all-gather)
  3  + parameters sharded (weights all-gathered bucket-by-bucket, then freed)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .device import Cluster
from .fabric import Fabric, shard_bounds
from .memory_model import FP16, FP32, K_ADAM
from .model import ModelConfig, grad_flat, init_flat, total_params


@dataclass
class StepRecord:
    step: int
    loss: float
    peak_bytes: int
    comm_bytes_per_gpu: float
    comm_seconds: float
    peak_breakdown: dict = field(default_factory=dict)


class ZeROEngine:
    def __init__(
        self,
        stage: int,
        cfg: ModelConfig,
        cluster: Cluster,
        fabric: Fabric,
        lr: float = 3e-4,
        threaded: bool = False,
        max_workers: int = 8,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        seed: int = 0,
        bucket_elems: int | None = None,
    ):
        assert stage in (0, 1, 2, 3)
        self.stage = stage
        self.cfg = cfg
        self.cluster = cluster
        self.fabric = fabric
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.threaded, self.max_workers = threaded, max_workers
        self.N = len(cluster)
        self.psi = total_params(cfg)
        self.bounds = shard_bounds(self.psi, self.N)
        # ZeRO-3 all-gathers weights in buckets, so the transient spike is one
        # bucket, not the whole model. Default: one shard's worth.
        self.bucket_elems = bucket_elems or max(1, self.psi // self.N)
        self.t = 0
        self.history: list[StepRecord] = []

        flat0 = init_flat(cfg, seed=seed)
        self._install(flat0)

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------
    def _install(self, flat0: np.ndarray):
        """Lay the model out across ranks according to the stage."""
        for r, gpu in enumerate(self.cluster):
            lo, hi = self.bounds[r]
            n_shard = hi - lo
            sharded_opt = self.stage >= 1
            sharded_grad = self.stage >= 2
            sharded_par = self.stage >= 3

            # fp16 working weights
            if sharded_par:
                gpu.put("w16", flat0[lo:hi].copy(), n_shard * FP16, "params")
            else:
                gpu.put("w16", flat0.copy(), self.psi * FP16, "params")

            # fp32 master + Adam moments
            n_opt = n_shard if sharded_opt else self.psi
            o_lo, o_hi = (lo, hi) if sharded_opt else (0, self.psi)
            gpu.put("w32", flat0[o_lo:o_hi].copy(), n_opt * FP32, "optimizer")
            gpu.put("m", np.zeros(n_opt), n_opt * FP32, "optimizer")
            gpu.put("v", np.zeros(n_opt), n_opt * FP32, "optimizer")

            # gradient buffer
            n_g = n_shard if sharded_grad else self.psi
            gpu.put("g", np.zeros(n_g), n_g * FP16, "grads")
            gpu.oom = False
        self.cluster.reset_peaks()

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------
    def full_weights(self, rank: int = 0) -> np.ndarray:
        """Reconstruct the full parameter vector (for inspection/tests)."""
        if self.stage < 3:
            return self.cluster[rank].get("w16").copy()
        return np.concatenate([g.get("w16") for g in self.cluster])

    def _materialise_for_compute(self) -> list[np.ndarray]:
        """Give every rank a usable full parameter vector.

        Stages 0-2 already hold one. Stage 3 does not, and must all-gather.

        Important subtlety: ZeRO-3 gathers weights TWICE per step. It gathers
        layer-by-layer during the forward pass and frees each bucket right
        after use -- that freeing is the whole point -- so the backward pass
        finds nothing resident and has to gather all over again. Our forward
        and backward are fused inside grad_flat(), so both gathers are charged
        here explicitly rather than being silently collapsed into one. This is
        what produces the 3*Psi vs 2*Psi ratio in the ZeRO paper.
        """
        if self.stage < 3:
            return [g.get("w16") for g in self.cluster]

        shards = [g.get("w16") for g in self.cluster]
        full = self.fabric.all_gather(shards)     # forward pass gather
        self.fabric.all_gather(shards)            # backward pass re-gather
        # Only ONE bucket is resident at a time, not the full model -- that is
        # the memory win, and why bucket size is a tunable in this file.
        spike = self.bucket_elems * FP16
        for g in self.cluster:
            g.put("w16_gathered", None, spike, "transient")
        return [full for _ in self.cluster]

    def _release_gathered(self):
        if self.stage == 3:
            for g in self.cluster:
                g.free("w16_gathered")

    # ------------------------------------------------------------------
    # one step
    # ------------------------------------------------------------------
    def step(self, batches) -> StepRecord:
        """`batches`: list of (x, y), one micro-batch per rank."""
        assert len(batches) == self.N, "need one micro-batch per rank"
        self.fabric.reset()
        self.cluster.reset_peaks()

        # ---- forward + backward, weights gathered if needed ----
        params = self._materialise_for_compute()
        act_bytes = self._activation_bytes(batches[0][0].shape)
        for r in range(self.N):
            self.cluster[r].put("acts", None, act_bytes, "activations")
        if self.threaded:
            from .threaded import threaded_fwd_bwd
            losses, grads = threaded_fwd_bwd(
                self.cfg, params, batches, max_workers=self.max_workers)
        else:
            losses, grads = [], []
            for r, (x, y) in enumerate(batches):
                loss, g = grad_flat(self.cfg, params[r], x, y)
                losses.append(loss)
                grads.append(g)
        for g in self.cluster:
            g.free("acts")
        self._release_gathered()

        # ---- gradient reduction ----
        # Only plain DP needs an all-reduce: every rank runs the full optimizer
        # so every rank needs the full averaged gradient. From ZeRO-1 onward a
        # rank only ever updates its own slice, so reduce-scatter delivers
        # exactly what it needs and nothing more.
        #
        # Worth being precise about: ZeRO-1 and ZeRO-2 have IDENTICAL
        # communication (2*Psi). They differ only in whether the gradient
        # BUFFER is full-size or sharded. ZeRO-2 is a pure memory win over
        # ZeRO-1 at zero extra wire cost -- which is why there is essentially
        # never a reason to run ZeRO-1 instead of ZeRO-2.
        if self.stage == 0:
            reduced = self.fabric.all_reduce(grads, op="mean")
            my_grad = [reduced[r] for r in range(self.N)]           # full on every rank
        else:
            my_grad = self.fabric.reduce_scatter(grads, op="mean")  # one slice per rank

        # ---- optimizer ----
        self.t += 1
        b1, b2 = self.betas
        for r, gpu in enumerate(self.cluster):
            lo, hi = self.bounds[r]
            if self.stage == 0:
                gpu.get("g")[:] = my_grad[r]
                gsel = my_grad[r]                          # full averaged gradient
            elif self.stage == 1:
                # Buffer is full-size (that is the memory cost of ZeRO-1) but
                # only the owned slice is ever read.
                gpu.get("g")[lo:hi] = my_grad[r]
                gsel = my_grad[r]
            else:
                gpu.get("g")[:] = my_grad[r]               # buffer is shard-sized
                gsel = my_grad[r]

            w32 = gpu.get("w32")
            m, v = gpu.get("m"), gpu.get("v")
            if self.wd:
                gsel = gsel + self.wd * w32
            m *= b1
            m += (1 - b1) * gsel
            v *= b2
            v += (1 - b2) * gsel ** 2
            mhat = m / (1 - b1 ** self.t)
            vhat = v / (1 - b2 ** self.t)
            w32 -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

        # ---- publish updated weights ----
        if self.stage == 0:
            for r, gpu in enumerate(self.cluster):
                gpu.get("w16")[:] = gpu.get("w32")
        else:
            # every rank updated only its own slice; rebuild the fp16 copies
            shards = [self.cluster[r].get("w32") for r in range(self.N)]
            if self.stage < 3:
                full = self.fabric.all_gather(shards)
                for gpu in self.cluster:
                    gpu.get("w16")[:] = full
            else:
                for r, gpu in enumerate(self.cluster):
                    gpu.get("w16")[:] = shards[r]

        rec = StepRecord(
            step=self.t,
            loss=float(np.mean(losses)),
            peak_bytes=self.cluster.peak(),
            comm_bytes_per_gpu=sum(
                d["per_gpu_bytes"] for d in self.fabric.stats.by_op.values()
            ),
            comm_seconds=self.fabric.stats.seconds,
            peak_breakdown=dict(self.cluster[0].peak_breakdown),
        )
        self.history.append(rec)
        return rec

    # ------------------------------------------------------------------
    def _activation_bytes(self, xshape) -> int:
        """Crude but consistent activation estimate: ~34*B*T*d per layer in
        fp16. Identical across stages, so it never distorts the comparison."""
        B, T = xshape
        c = self.cfg
        return int(34 * B * T * c.d_model * c.n_layer * FP16)

    def model_state_bytes_per_gpu(self) -> int:
        g = self.cluster[0]
        b = g.breakdown()
        return sum(b.get(k, 0) for k in ("params", "grads", "optimizer"))
