"""A tiny causal transformer LM whose parameters live in ONE flat 1-D vector.

The flat-vector representation is the whole point: ZeRO shards *bytes*, not
"layers", so the engine only ever needs to slice a contiguous buffer. Views
into the flat vector are handed to the forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import autodiff as ad
from .autodiff import Tensor


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 32
    block_size: int = 8

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: tuple
    offset: int

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))


def build_param_specs(cfg: ModelConfig) -> list[ParamSpec]:
    specs, off = [], 0

    def add(name, shape):
        nonlocal off
        specs.append(ParamSpec(name, tuple(shape), off))
        off += int(np.prod(shape))

    add("wte", (cfg.vocab_size, cfg.d_model))
    add("wpe", (cfg.block_size, cfg.d_model))
    for i in range(cfg.n_layer):
        add(f"h{i}.ln1.g", (cfg.d_model,))
        add(f"h{i}.ln1.b", (cfg.d_model,))
        add(f"h{i}.attn.qkv", (cfg.d_model, 3 * cfg.d_model))
        add(f"h{i}.attn.proj", (cfg.d_model, cfg.d_model))
        add(f"h{i}.ln2.g", (cfg.d_model,))
        add(f"h{i}.ln2.b", (cfg.d_model,))
        add(f"h{i}.mlp.fc", (cfg.d_model, cfg.d_ff))
        add(f"h{i}.mlp.fcb", (cfg.d_ff,))
        add(f"h{i}.mlp.proj", (cfg.d_ff, cfg.d_model))
        add(f"h{i}.mlp.projb", (cfg.d_model,))
    add("lnf.g", (cfg.d_model,))
    add("lnf.b", (cfg.d_model,))
    add("head", (cfg.d_model, cfg.vocab_size))
    return specs


def total_params(cfg: ModelConfig) -> int:
    specs = build_param_specs(cfg)
    return specs[-1].offset + specs[-1].numel


def init_flat(cfg: ModelConfig, seed: int = 0) -> np.ndarray:
    """Deterministic init. Same seed -> bit-identical vector, which is what
    makes the cross-stage equivalence test meaningful."""
    rng = np.random.default_rng(seed)
    flat = np.zeros(total_params(cfg), dtype=np.float64)
    for s in build_param_specs(cfg):
        sl = slice(s.offset, s.offset + s.numel)
        if s.name.endswith(".g"):
            flat[sl] = 1.0
        elif s.name.endswith(".b") or s.name.endswith("b"):
            flat[sl] = 0.0
        else:
            flat[sl] = rng.normal(0.0, 0.02, size=s.numel)
    return flat


def views(cfg: ModelConfig, flat: Tensor) -> dict:
    """Slice the flat Tensor into named parameter Tensors (autodiff-connected)."""
    out = {}
    for s in build_param_specs(cfg):
        piece = ad.reshape(_slice(flat, s.offset, s.numel), s.shape)
        out[s.name] = piece
    return out


def _slice(t: Tensor, off: int, n: int) -> Tensor:
    data = t.data[off:off + n]
    out = Tensor(data, requires_grad=False, _parents=(t,))

    def bw():
        if out.grad is None:
            return
        g = np.zeros_like(t.data)
        g[off:off + n] = out.grad
        t._accum(g)

    out._backward = bw
    return out


def forward_loss(cfg: ModelConfig, flat: Tensor, x: np.ndarray, y: np.ndarray) -> Tensor:
    """x, y: (B, T) int arrays. Returns scalar mean cross-entropy Tensor."""
    p = views(cfg, flat)
    B, T = x.shape
    H, D = cfg.n_head, cfg.d_model
    hd = D // H

    h = ad.add(ad.embedding(p["wte"], x), ad.embedding(p["wpe"], np.arange(T)))
    causal = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]

    for i in range(cfg.n_layer):
        # ---- attention ----
        a = ad.layernorm(h, p[f"h{i}.ln1.g"], p[f"h{i}.ln1.b"])
        qkv = ad.matmul(a, p[f"h{i}.attn.qkv"])              # (B,T,3D)
        qkv = ad.reshape(qkv, (B, T, 3, H, hd))
        qkv = ad.transpose(qkv, (2, 0, 3, 1, 4))             # (3,B,H,T,hd)
        q = _idx0(qkv, 0)
        k = _idx0(qkv, 1)
        v = _idx0(qkv, 2)
        att = ad.scale(ad.matmul(q, ad.transpose(k, (0, 1, 3, 2))), 1.0 / np.sqrt(hd))
        att = ad.masked_softmax(att, causal)
        o = ad.matmul(att, v)                                # (B,H,T,hd)
        o = ad.reshape(ad.transpose(o, (0, 2, 1, 3)), (B, T, D))
        h = ad.add(h, ad.matmul(o, p[f"h{i}.attn.proj"]))

        # ---- mlp ----
        m = ad.layernorm(h, p[f"h{i}.ln2.g"], p[f"h{i}.ln2.b"])
        m = ad.add(ad.matmul(m, p[f"h{i}.mlp.fc"]), p[f"h{i}.mlp.fcb"])
        m = ad.gelu(m)
        m = ad.add(ad.matmul(m, p[f"h{i}.mlp.proj"]), p[f"h{i}.mlp.projb"])
        h = ad.add(h, m)

    h = ad.layernorm(h, p["lnf.g"], p["lnf.b"])
    logits = ad.matmul(h, p["head"])
    return ad.cross_entropy(logits, y)


def _idx0(t: Tensor, i: int) -> Tensor:
    """Index the leading axis, keeping the autodiff link."""
    out = Tensor(t.data[i], requires_grad=False, _parents=(t,))

    def bw():
        if out.grad is None:
            return
        g = np.zeros_like(t.data)
        g[i] = out.grad
        t._accum(g)

    out._backward = bw
    return out


def grad_flat(cfg: ModelConfig, flat_np: np.ndarray, x, y) -> tuple[float, np.ndarray]:
    """One forward+backward. Returns (loss, dL/dflat)."""
    flat = Tensor(flat_np, requires_grad=True)
    loss = forward_loss(cfg, flat, x, y)
    loss.backward()
    return float(loss.data), flat.grad.copy()
