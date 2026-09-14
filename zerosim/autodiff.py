"""Minimal reverse-mode autodiff over NumPy.

Deliberately tiny. Every op here is finite-difference tested in
tests/test_autodiff.py so nothing in the ZeRO engine sits on an unverified
gradient.
"""
from __future__ import annotations

import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_parents", "requires_grad")

    def __init__(self, data, requires_grad=False, _parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"

    # ---- graph ----
    def backward(self):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()

    def _accum(self, g):
        if not self.requires_grad and not self._parents:
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g


def _out(data, parents, requires_grad):
    t = Tensor(data, requires_grad=requires_grad, _parents=parents)
    return t


def _needs(*ts):
    return any(t.requires_grad or t._parents for t in ts)


def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape` after numpy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


# --------------------------------------------------------------------------
# ops
# --------------------------------------------------------------------------
def add(a: Tensor, b: Tensor) -> Tensor:
    out = _out(a.data + b.data, (a, b), _needs(a, b))

    def bw():
        if out.grad is None:
            return
        a._accum(_unbroadcast(out.grad, a.data.shape))
        b._accum(_unbroadcast(out.grad, b.data.shape))

    out._backward = bw
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = _out(a.data * b.data, (a, b), _needs(a, b))

    def bw():
        if out.grad is None:
            return
        a._accum(_unbroadcast(out.grad * b.data, a.data.shape))
        b._accum(_unbroadcast(out.grad * a.data, b.data.shape))

    out._backward = bw
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = _out(a.data @ b.data, (a, b), _needs(a, b))

    def bw():
        if out.grad is None:
            return
        g = out.grad
        ga = g @ np.swapaxes(b.data, -1, -2)
        gb = np.swapaxes(a.data, -1, -2) @ g
        a._accum(_unbroadcast(ga, a.data.shape))
        b._accum(_unbroadcast(gb, b.data.shape))

    out._backward = bw
    return out


def reshape(a: Tensor, shape) -> Tensor:
    out = _out(a.data.reshape(shape), (a,), _needs(a))

    def bw():
        if out.grad is None:
            return
        a._accum(out.grad.reshape(a.data.shape))

    out._backward = bw
    return out


def transpose(a: Tensor, axes) -> Tensor:
    out = _out(np.transpose(a.data, axes), (a,), _needs(a))
    inv = np.argsort(axes)

    def bw():
        if out.grad is None:
            return
        a._accum(np.transpose(out.grad, inv))

    out._backward = bw
    return out


def gelu(a: Tensor) -> Tensor:
    """Tanh-approximation GELU (the GPT-2 variant)."""
    c = np.sqrt(2.0 / np.pi)
    x = a.data
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    out = _out(0.5 * x * (1.0 + t), (a,), _needs(a))

    def bw():
        if out.grad is None:
            return
        dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
        d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dinner
        a._accum(out.grad * d)

    out._backward = bw
    return out


def layernorm(a: Tensor, gamma: Tensor, beta: Tensor, eps=1e-5) -> Tensor:
    x = a.data
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv
    out = _out(xhat * gamma.data + beta.data, (a, gamma, beta), _needs(a, gamma, beta))

    def bw():
        if out.grad is None:
            return
        g = out.grad
        D = x.shape[-1]
        gamma._accum(_unbroadcast((g * xhat).reshape(-1, D).sum(0), gamma.data.shape))
        beta._accum(_unbroadcast(g.reshape(-1, D).sum(0), beta.data.shape))
        dxhat = g * gamma.data
        dx = inv / D * (
            D * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )
        a._accum(dx)

    out._backward = bw
    return out


def embedding(table: Tensor, idx: np.ndarray) -> Tensor:
    idx = np.asarray(idx)
    out = _out(table.data[idx], (table,), _needs(table))

    def bw():
        if out.grad is None:
            return
        g = np.zeros_like(table.data)
        np.add.at(g, idx.reshape(-1), out.grad.reshape(-1, table.data.shape[-1]))
        table._accum(g)

    out._backward = bw
    return out


def masked_softmax(a: Tensor, mask: np.ndarray) -> Tensor:
    """Softmax over the last axis with an additive causal mask (bool: True=keep)."""
    x = np.where(mask, a.data, -1e30)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=-1, keepdims=True)
    out = _out(p, (a,), _needs(a))

    def bw():
        if out.grad is None:
            return
        g = out.grad
        dx = p * (g - (g * p).sum(axis=-1, keepdims=True))
        a._accum(np.where(mask, dx, 0.0))

    out._backward = bw
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean token-level cross entropy. logits (..., V), targets (...)."""
    V = logits.data.shape[-1]
    flat = logits.data.reshape(-1, V)
    tgt = np.asarray(targets).reshape(-1)
    m = flat.max(axis=-1, keepdims=True)
    z = flat - m
    lse = np.log(np.exp(z).sum(axis=-1)) + m[:, 0]
    loss = (lse - flat[np.arange(flat.shape[0]), tgt]).mean()
    out = _out(loss, (logits,), _needs(logits))

    def bw():
        if out.grad is None:
            return
        p = np.exp(z - (lse - m[:, 0])[:, None])
        p[np.arange(p.shape[0]), tgt] -= 1.0
        p /= p.shape[0]
        logits._accum((p * out.grad).reshape(logits.data.shape))

    out._backward = bw
    return out


def scale(a: Tensor, s: float) -> Tensor:
    out = _out(a.data * s, (a,), _needs(a))

    def bw():
        if out.grad is None:
            return
        a._accum(out.grad * s)

    out._backward = bw
    return out
