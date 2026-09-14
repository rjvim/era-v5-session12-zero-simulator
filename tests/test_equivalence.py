"""If ZeRO is implemented correctly, the stage is invisible to the math.

Stages 0/1/2/3 differ only in *where bytes live*. Given identical init and
identical micro-batches they must walk the identical optimization trajectory.
This is the test that catches a wrong shard boundary, a reduce-scatter that
sums where it should average, or an all-gather that reassembles out of order.
"""
import numpy as np
import pytest

from zerosim.device import Cluster
from zerosim.fabric import Fabric, H100
from zerosim.engine import ZeROEngine
from zerosim.model import ModelConfig

WORLD = 8   # small world keeps the test fast; run_experiments.py uses 32


def make(stage, world=WORLD, steps=0, seed=0, cfg=None):
    cfg = cfg or ModelConfig(vocab_size=11, n_layer=2, n_head=2, d_model=16, block_size=6)
    cl = Cluster(world_size=world, gpus_per_node=8)
    fb = Fabric(cl, H100)
    return ZeROEngine(stage, cfg, cl, fb, lr=1e-2, seed=seed), cfg


def batches(cfg, world, seed=7, B=2):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(world):
        x = rng.integers(0, cfg.vocab_size, size=(B, cfg.block_size))
        y = rng.integers(0, cfg.vocab_size, size=(B, cfg.block_size))
        out.append((x, y))
    return out


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_stage_matches_data_parallel(stage):
    """ZeRO-1/2/3 must reproduce plain data parallelism exactly."""
    ref, cfg = make(0)
    tst, _ = make(stage)
    bs = [batches(cfg, WORLD, seed=100 + s) for s in range(5)]

    for b in bs:
        r0 = ref.step(b)
        r1 = tst.step(b)
        assert abs(r0.loss - r1.loss) < 1e-12, f"loss drift at step {r0.step}"

    w_ref = ref.full_weights()
    w_tst = tst.full_weights()
    assert np.allclose(w_ref, w_tst, atol=1e-12, rtol=0), (
        f"stage {stage} weights diverged: max |d| = {np.abs(w_ref - w_tst).max():.3e}"
    )


def test_loss_actually_decreases():
    """A trajectory that doesn't learn would make the equivalence test vacuous."""
    eng, cfg = make(3)
    b = batches(cfg, WORLD, seed=42)
    first = eng.step(b).loss
    for _ in range(40):
        last = eng.step(b).loss
    assert last < first * 0.8, f"loss went {first:.4f} -> {last:.4f}"


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_shard_ownership(stage):
    """Every parameter is owned by exactly one rank (stages 1-3)."""
    eng, _ = make(stage)
    if stage == 0:
        for g in eng.cluster:
            assert g.get("w32").size == eng.psi
        return
    covered = np.zeros(eng.psi, dtype=int)
    for (lo, hi) in eng.bounds:
        covered[lo:hi] += 1
    assert (covered == 1).all(), "shards overlap or leave gaps"
    for r, g in enumerate(eng.cluster):
        lo, hi = eng.bounds[r]
        assert g.get("w32").size == hi - lo


def test_memory_is_monotonically_nonincreasing_in_stage():
    """Higher stage must never cost more model-state memory."""
    prev = None
    for stage in (0, 1, 2, 3):
        eng, cfg = make(stage)
        eng.step(batches(cfg, WORLD))
        cur = eng.model_state_bytes_per_gpu()
        if prev is not None:
            assert cur <= prev, f"stage {stage} used more than stage {stage-1}"
        prev = cur


def test_zero3_communicates_more():
    """The 1.5x: ZeRO-3 buys memory with wire time."""
    vols = {}
    for stage in (0, 1, 2, 3):
        eng, cfg = make(stage)
        rec = eng.step(batches(cfg, WORLD))
        vols[stage] = rec.comm_bytes_per_gpu
    assert vols[3] > vols[2], "ZeRO-3 should move more than ZeRO-2"
    ratio = vols[3] / vols[0]
    assert 1.4 < ratio < 1.6, f"expected ~1.5x all-reduce volume, got {ratio:.3f}"


def test_world_size_32_equivalence():
    """The assignment's actual configuration."""
    cfg = ModelConfig(vocab_size=11, n_layer=1, n_head=2, d_model=16, block_size=6)
    ref, _ = make(0, world=32, cfg=cfg)
    tst, _ = make(3, world=32, cfg=cfg)
    b = batches(cfg, 32, seed=9)
    for _ in range(3):
        r0, r1 = ref.step(b), tst.step(b)
        assert abs(r0.loss - r1.loss) < 1e-12
    assert np.allclose(ref.full_weights(), tst.full_weights(), atol=1e-12, rtol=0)
