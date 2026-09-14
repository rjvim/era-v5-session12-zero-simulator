"""Finite-difference verification.

Nothing in the ZeRO engine is trustworthy if the gradients underneath it are
wrong, so every op and the assembled model are checked against central
differences before anything else runs.
"""
import numpy as np
import pytest

from zerosim import autodiff as ad
from zerosim.autodiff import Tensor
from zerosim.model import ModelConfig, grad_flat, init_flat, total_params


def numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def check(fn, shapes, seed=0, tol=1e-6):
    """fn takes Tensors, returns a scalar Tensor."""
    rng = np.random.default_rng(seed)
    arrs = [rng.normal(size=s) for s in shapes]
    ts = [Tensor(a.copy(), requires_grad=True) for a in arrs]
    out = fn(*ts)
    out.backward()
    for j, t in enumerate(ts):
        def f(v, j=j):
            tmp = [Tensor(a.copy()) for a in arrs]
            tmp[j] = Tensor(v.copy())
            return float(fn(*tmp).data)
        ng = numeric_grad(f, arrs[j].copy())
        assert np.allclose(t.grad, ng, atol=tol, rtol=1e-4), (
            f"arg {j}: max abs err {np.abs(t.grad - ng).max():.3e}"
        )


def _sum(t):
    out = Tensor(t.data.sum(), _parents=(t,))

    def bw():
        t._accum(np.ones_like(t.data) * out.grad)
    out._backward = bw
    return out


def test_add():
    check(lambda a, b: _sum(ad.add(a, b)), [(3, 4), (4,)])


def test_mul():
    check(lambda a, b: _sum(ad.mul(a, b)), [(3, 4), (3, 1)])


def test_matmul():
    check(lambda a, b: _sum(ad.matmul(a, b)), [(2, 3, 4), (4, 5)])


def test_gelu():
    check(lambda a: _sum(ad.gelu(a)), [(3, 5)])


def test_layernorm():
    check(lambda a, g, b: _sum(ad.layernorm(a, g, b)), [(2, 3, 6), (6,), (6,)], tol=1e-5)


def test_transpose_reshape():
    check(lambda a: _sum(ad.reshape(ad.transpose(a, (1, 0, 2)), (12, 2))), [(3, 4, 2)])


def test_masked_softmax():
    mask = np.tril(np.ones((4, 4), dtype=bool))
    check(lambda a: _sum(ad.masked_softmax(a, mask)), [(2, 4, 4)], tol=1e-5)


def test_scale():
    check(lambda a: _sum(ad.scale(a, 0.37)), [(3, 3)])


def test_cross_entropy():
    tgt = np.array([[1, 0, 2], [2, 2, 1]])
    check(lambda a: ad.cross_entropy(a, tgt), [(2, 3, 4)], tol=1e-6)


def test_embedding():
    idx = np.array([[0, 2, 2], [1, 0, 3]])
    check(lambda t: _sum(ad.embedding(t, idx)), [(4, 5)])


def test_full_model_gradient():
    """The end-to-end check that matters: dL/dflat vs central differences."""
    cfg = ModelConfig(vocab_size=11, n_layer=2, n_head=2, d_model=8, block_size=5)
    flat = init_flat(cfg, seed=3)
    rng = np.random.default_rng(1)
    x = rng.integers(0, cfg.vocab_size, size=(2, cfg.block_size))
    y = rng.integers(0, cfg.vocab_size, size=(2, cfg.block_size))

    _, g = grad_flat(cfg, flat, x, y)

    # Spot-check 60 coordinates spread across the whole vector (full check is
    # O(n) forward passes and n is ~3k here).
    n = total_params(cfg)
    idxs = np.linspace(0, n - 1, 60).astype(int)
    eps = 1e-6
    for i in idxs:
        p = flat.copy(); p[i] += eps
        lp, _ = _loss_only(cfg, p, x, y)
        m = flat.copy(); m[i] -= eps
        lm, _ = _loss_only(cfg, m, x, y)
        ng = (lp - lm) / (2 * eps)
        assert abs(g[i] - ng) < 1e-5 + 1e-4 * abs(ng), (
            f"param {i}: analytic {g[i]:.8f} vs numeric {ng:.8f}"
        )


def _loss_only(cfg, flat_np, x, y):
    from zerosim.model import forward_loss
    t = Tensor(flat_np)
    return float(forward_loss(cfg, t, x, y).data), None
