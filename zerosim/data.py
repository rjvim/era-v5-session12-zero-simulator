"""A tiny syllable-level corpus.

Chosen so the model has real structure to learn: the loss curve has to
actually move, otherwise "all four stages produced the same loss" is a
statement about a model that learned nothing and proves much less.

Syllable-level rather than character-level because these are chants -- the
syllable is the unit that actually repeats.
"""
from __future__ import annotations

import numpy as np

MANTRAS = [
    ["om", "na", "mah", "shi", "va", "ya"],
    ["om", "na", "mo", "bha", "ga", "va", "te"],
    ["om", "ga", "na", "pa", "ta", "ye", "na", "mah"],
    ["om", "shan", "ti", "shan", "ti", "shan", "ti"],
    ["om", "bhur", "bhu", "vah", "sva", "ha"],
    ["ha", "re", "krish", "na", "ha", "re", "ra", "ma"],
]


def build_vocab() -> tuple[dict, list]:
    syls = sorted({s for m in MANTRAS for s in m})
    itos = ["<pad>"] + syls
    stoi = {s: i for i, s in enumerate(itos)}
    return stoi, itos


def build_stream(repeats: int = 400, seed: int = 0) -> np.ndarray:
    """Concatenate mantras in a shuffled but deterministic order."""
    stoi, _ = build_vocab()
    rng = np.random.default_rng(seed)
    toks = []
    for _ in range(repeats):
        m = MANTRAS[rng.integers(0, len(MANTRAS))]
        toks.extend(stoi[s] for s in m)
    return np.array(toks, dtype=np.int64)


def batches(stream: np.ndarray, world: int, n_steps: int, B: int, T: int,
            seed: int = 0):
    """Deterministic (x, y) micro-batches, one per rank per step.

    Determinism matters: the cross-stage equivalence claim only means
    something if every stage sees byte-identical data.
    """
    rng = np.random.default_rng(seed)
    steps = []
    for _ in range(n_steps):
        per_rank = []
        for _ in range(world):
            ix = rng.integers(0, len(stream) - T - 1, size=B)
            x = np.stack([stream[i:i + T] for i in ix])
            y = np.stack([stream[i + 1:i + 1 + T] for i in ix])
            per_rank.append((x, y))
        steps.append(per_rank)
    return steps
