"""Threaded execution must give byte-identical results to sequential."""
import numpy as np

from zerosim.data import batches, build_stream, build_vocab
from zerosim.device import Cluster
from zerosim.engine import ZeROEngine
from zerosim.fabric import H100, Fabric
from zerosim.model import ModelConfig

_, ITOS = build_vocab()


def _run(threaded, stage=3, world=32, steps=3):
    cfg = ModelConfig(vocab_size=len(ITOS), n_layer=1, n_head=2, d_model=16,
                      block_size=6)
    stream = build_stream(repeats=200, seed=0)
    bs = batches(stream, world, steps, B=2, T=cfg.block_size, seed=5)
    cl = Cluster(world_size=world, gpus_per_node=8)
    e = ZeROEngine(stage, cfg, cl, Fabric(cl, H100), lr=1e-2, seed=0,
                   threaded=threaded, max_workers=8)
    losses = [e.step(b).loss for b in bs]
    return losses, e.full_weights()


def test_threaded_matches_sequential():
    ls, ws = _run(False)
    lt, wt = _run(True)
    assert np.allclose(ls, lt, atol=1e-12, rtol=0)
    assert np.abs(ws - wt).max() == 0.0, "threading changed the result"


def test_barrier_counts_synchronisations():
    from zerosim.threaded import RankBarrier, threaded_fwd_bwd
    from zerosim.model import ModelConfig, init_flat
    cfg = ModelConfig(vocab_size=len(ITOS), n_layer=1, n_head=2, d_model=16,
                      block_size=6)
    flat = init_flat(cfg, seed=0)
    stream = build_stream(repeats=50, seed=0)
    bs = batches(stream, 8, 1, B=1, T=cfg.block_size, seed=1)[0]
    bar = RankBarrier(8)
    threaded_fwd_bwd(cfg, [flat] * 8, bs, max_workers=8, barrier=bar)
    assert bar.waits == 16, "8 ranks x 2 barriers per step"
