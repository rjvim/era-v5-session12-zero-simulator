"""Figures.

Four plots, one per claim I want to make visually rather than in a table:
memory by stage, the memory ladder vs world size, the bucket trade-off, and
the memory/throughput frontier.

Deliberately plain: no seaborn, no styling beyond readability. The point is to
make the shape of the trade-off visible, not to decorate.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .compute import H100_PROFILE, step_time
from .memory_model import model_state_bytes, model_state_gb

STAGE_LABEL = ["DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
COLORS = {"params": "#4C72B0", "grads": "#DD8452", "optimizer": "#55A868"}


def fig_memory_breakdown(psi: float = 27.9e9, world: int = 8, ax=None):
    """Stacked bar: where the bytes actually go, per stage.

    The story this tells: the optimizer states (green) dominate DP completely,
    which is why they are the first thing ZeRO shards.
    """
    ax = ax or plt.subplots(figsize=(7, 4.2))[1]
    x = np.arange(4)
    parts = {k: [] for k in COLORS}
    for s in range(4):
        b = model_state_bytes(psi, world, s)
        parts["params"].append(b.fp16_params / 1e9)
        parts["grads"].append(b.fp16_grads / 1e9)
        parts["optimizer"].append(b.optimizer / 1e9)

    bottom = np.zeros(4)
    for k in ("params", "grads", "optimizer"):
        v = np.array(parts[k])
        ax.bar(x, v, bottom=bottom, label=k, color=COLORS[k])
        bottom += v

    for i, t in enumerate(bottom):
        ax.text(i, t + max(bottom) * 0.02, f"{t:.0f} GB", ha="center", fontsize=9)

    ax.axhline(80, ls="--", c="crimson", lw=1.2)
    ax.text(3.42, 84, "80 GB H100", color="crimson", fontsize=8, ha="right")
    ax.set_xticks(x, STAGE_LABEL)
    ax.set_ylabel("per-GPU model states (GB)")
    ax.set_title(f"Where the memory goes  (Psi={psi/1e9:.1f}B, N={world})")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(bottom) * 1.15)
    return ax


def fig_memory_ladder(psi: float = 27.9e9, ax=None):
    """Per-GPU memory vs world size, log-log.

    The shape is the whole point: DP is FLAT. Adding GPUs to plain data
    parallelism buys you exactly nothing in memory. Only ZeRO-3 has slope -1.
    """
    ax = ax or plt.subplots(figsize=(7, 4.2))[1]
    worlds = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    for s in range(4):
        y = [model_state_gb(psi, int(n), s) for n in worlds]
        ax.plot(worlds, y, marker="o", ms=4, label=STAGE_LABEL[s])
    ax.axhline(80, ls="--", c="crimson", lw=1.2, label="80 GB H100")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of GPUs (N)")
    ax.set_ylabel("per-GPU model states (GB)")
    ax.set_title(f"Memory ladder  (Psi={psi/1e9:.1f}B)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9)
    return ax


def fig_bucket_tradeoff(sweep: dict, ax=None):
    """Peak memory vs bucket size, with communication overlaid.

    Expect: peak falls, communication is a flat line. That flatness is the
    finding -- smaller buckets are close to free.
    """
    ax = ax or plt.subplots(figsize=(7, 4.2))[1]
    keys = sorted(sweep.keys(), key=lambda k: int(k))
    labels = [f"Psi/{k}" for k in keys]
    peak = [sweep[k]["peak_bytes_per_gpu"] / 1024 for k in keys]
    comm = [sweep[k]["comm_bytes_per_gpu_per_step"] / 1024 for k in keys]
    x = np.arange(len(keys))

    ax.bar(x, peak, color="#4C72B0", label="peak memory / GPU")
    ax.plot(x, comm, marker="s", color="#C44E52", lw=2, label="communication / step")
    ax.set_xticks(x, labels)
    ax.set_xlabel("bucket size")
    ax.set_ylabel("KiB")
    ax.set_title("Smaller buckets: peak memory falls, communication does not move")
    ax.legend(frameon=False, fontsize=9)
    return ax


def fig_activation_floor(toy: dict, ax=None):
    """Model states shrink 32x; activations do not move at all.

    The grey block is what ZeRO cannot touch. By ZeRO-3 it is most of the bar,
    which is the difference between the 32x headline and the 5.92x measured.
    """
    ax = ax or plt.subplots(figsize=(7.5, 4.4))[1]
    x = np.arange(4)
    ms = np.array([toy[str(s)]["model_state_bytes"] / 1024 for s in range(4)])
    act = np.array([toy[str(s)]["activation_bytes"] / 1024 for s in range(4)])
    ax.bar(x, ms, label="model states (ZeRO shards these)", color="#4C72B0")
    ax.bar(x, act, bottom=ms, label="activations (ZeRO does NOT)", color="#999999")
    for i in range(4):
        ax.text(i, ms[i] + act[i] + 8, f"{toy[str(i)]['peak_reduction']:.2f}x",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, STAGE_LABEL)
    ax.set_ylabel("peak per-GPU memory (KiB)")
    ax.set_title("What ZeRO shards, and what it doesn't\n"
                 "labels = measured PEAK reduction (model-state reduction is 32x)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, (ms + act).max() * 1.16)
    return ax


def fig_throughput_frontier(ax=None):
    """Memory vs throughput for models that fit under every stage.

    Reads as: every step up the ZeRO ladder moves you left (less memory) and
    slightly down (less throughput). OOM points are marked hollow.
    """
    ax = ax or plt.subplots(figsize=(7, 4.6))[1]
    markers = {1: "o", 3: "s", 7: "^"}
    for psi_b, mk in markers.items():
        psi = psi_b * 1e9
        xs, ys, fits = [], [], []
        for s in range(4):
            r = step_time(psi, s, 8, H100_PROFILE, 4096, 32, 4096, 4096,
                          nodes=1, overlap=0.8)
            xs.append(model_state_gb(psi, 8, s))
            ys.append(r["tokens_per_second_per_gpu"])
            fits.append(model_state_bytes(psi, 8, s).total <= 80e9)
        ax.plot(xs, ys, ls="-", lw=1, c="gray", alpha=0.5, zorder=1)
        for x, y, f, lbl in zip(xs, ys, fits, STAGE_LABEL):
            ax.scatter(x, y, marker=mk, s=70, zorder=3,
                       facecolor=("#4C72B0" if f else "none"),
                       edgecolor=("#4C72B0" if f else "crimson"),
                       linewidths=1.6)
        ax.annotate(f"{psi_b}B", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    ax.axvline(80, ls="--", c="crimson", lw=1.2)
    ax.text(78, ax.get_ylim()[0], " 80 GB", color="crimson", fontsize=8,
            rotation=90, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("per-GPU model states (GB)  [lower = better]")
    ax.set_ylabel("tokens / s / GPU  [higher = better]")
    ax.set_title("Memory vs throughput (N=8, H100, 80% overlap)\n"
                 "hollow red = does not fit in 80 GB")
    ax.grid(alpha=0.3)
    return ax
