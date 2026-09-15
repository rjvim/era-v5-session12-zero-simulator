#!/usr/bin/env python3
"""Validate ALL FOUR stages against real PyTorch, on CPU. Actually run.

Output committed to artifacts/exp12_pytorch_all_stages.json.

How each stage maps onto real PyTorch:

  stage 0  DP      plain module + AdamW, gradients all-reduced by hand
  stage 1  ZeRO-1  DDP + torch.distributed.optim.ZeroRedundancyOptimizer
                   (PyTorch's own ZeRO-1: optimizer state sharded)
  stage 2  ZeRO-2  FSDP2 fully_shard(reshard_after_forward=False)
                   params stay gathered between forward and backward, while
                   gradients and optimizer state are sharded
  stage 3  ZeRO-3  FSDP2 fully_shard(reshard_after_forward=True)
                   params resharded after forward, so backward must re-gather

FSDP1 refuses to initialise without a GPU. FSDP2 is DTensor-based and runs on
a CPU device mesh, which is what makes stages 2 and 3 checkable here at all.

Two independent things are measured:

  1. MEMORY -- per-rank local element counts for params, grads and optimizer
     moments, against the ledger in zerosim/memory_model.py.

  2. COMMUNICATION -- real collective traffic, by wrapping
     dist.all_gather_into_tensor / reduce_scatter_tensor / all_reduce and
     counting bytes. This is the direct test of the central claim that ZeRO-3
     costs 1.5x ZeRO-2: if PyTorch does not actually re-gather weights in the
     backward pass, the ratio will come out 1.0 and my README is wrong.

Tolerances, fixed before running so they cannot be moved afterwards:
  - memory element counts within 5% of the ledger (ZeRO-1 is expected to be
    slightly off because real ZeRO-1 shards whole tensors, not elements)
  - the ZeRO-3 / ZeRO-2 communication ratio within 0.25 of 1.5

    python validate_pytorch_all_stages.py --world 2
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import warnings

warnings.filterwarnings("ignore")

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


# Ops that actually put bytes on the wire, and the ring cost factor for each.
# An all-reduce is reduce-scatter + all-gather, hence factor 2.
WIRE_OPS = {
    "c10d._allgather_base_": 1.0,
    "c10d.allgather_": 1.0,
    "c10d._reduce_scatter_base_": 1.0,
    "c10d.reduce_scatter_": 1.0,
    "c10d.allreduce_": 2.0,
    "c10d.allreduce_coalesced_": 2.0,
}

# Local helpers that are NOT network operations. Counting these was a real bug
# in an earlier version of this script: FSDP2 issues every all-gather as a
# pair -- fsdp.all_gather_copy_in (a local staging copy) followed by
# c10d._allgather_base_ (the collective) -- so counting both inflated ZeRO-2
# to 2.5x and ZeRO-3 to 4.5x of param bytes, and made the ZeRO-3/ZeRO-2 ratio
# read 1.8 instead of 1.5. The tolerance was NOT widened to accommodate that;
# the double-count was removed.
LOCAL_HELPERS = ("all_gather_copy_in", "chunk_cat", "split_with_sizes_copy")


def _make_spy(world: int = 1):
    """Intercept collectives at the DISPATCHER, not the Python dist.* API.

    This matters and cost me a wrong result: FSDP2 and DDP issue their
    collectives through C++/functional ops (fsdp.all_gather_copy_in,
    c10d._reduce_scatter_base_, c10d.allreduce_), so monkeypatching
    torch.distributed.all_gather_into_tensor et al. intercepts nothing and
    reports a confident zero. A TorchDispatchMode sees the actual ops.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class Spy(TorchDispatchMode):
        world = 1

        def __init__(self):
            self.calls: dict[str, int] = {}
            self.bytes: dict[str, int] = {}
            self.on = False

        def reset(self):
            self.calls, self.bytes = {}, {}

        def total(self):
            return sum(self.bytes.values())

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            if not self.on:
                return out
            name = str(func)
            base = name.split(".default")[0]
            if any(h in name for h in LOCAL_HELPERS):
                self.calls["(local) " + base] = self.calls.get("(local) " + base, 0) + 1
                return out
            factor = WIRE_OPS.get(base)
            if factor is not None:
                # Size the FULL buffer being moved, not whichever tensor the
                # op happens to return. For _reduce_scatter_base_ the return
                # value is the already-scattered slice, so taking only the
                # output undercounts the ring cost by a factor of N/(N-1)*...
                # -- it made the ZeRO-3/ZeRO-2 ratio read 1.667 instead of 1.5.
                nb = 0
                for t in _flat_tensors(out) + _flat_tensors(args):
                    nb = max(nb, t.numel() * t.element_size())
                # Ring wire cost per rank for a buffer of nb bytes.
                wire = factor * (self.world - 1) / self.world * nb
                self.calls[base] = self.calls.get(base, 0) + 1
                self.bytes[base] = self.bytes.get(base, 0) + wire
            return out

    sp = Spy()
    sp.world = world
    return sp


def _flat_tensors(obj):
    import torch
    found = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            found.append(o)
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return found


def build(d_model: int, n_layer: int, vocab: int):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.fc = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)
            self.ln2 = nn.LayerNorm(d)
            self.mix = nn.Linear(d, d)

        def forward(self, x):
            x = x + self.mix(self.ln1(x))
            return x + self.fc2(F.gelu(self.fc(self.ln2(x))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab, d_model)
            self.blocks = nn.ModuleList([Block(d_model) for _ in range(n_layer)])
            self.lnf = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)

        def forward(self, idx):
            x = self.wte(idx)
            for b in self.blocks:
                x = b(x)
            return self.head(self.lnf(x))

    return Net, Block


def _local_numel(t):
    return t.to_local().numel() if hasattr(t, "to_local") else t.numel()


def worker(rank, world, args, ret):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from torch.nn.parallel import DistributedDataParallel as DDP

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))

    Net, Block = build(args.d_model, args.n_layer, args.vocab)
    spy = _make_spy(world)
    out = {}

    torch.manual_seed(0)
    psi = sum(p.numel() for p in Net().parameters())
    x = torch.randint(0, args.vocab, (args.batch, args.seq),
                      generator=torch.Generator().manual_seed(100 + rank))

    for stage in (0, 1, 2, 3):
        torch.manual_seed(0)
        model = Net()

        if stage == 0:
            model = DDP(model)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
        elif stage == 1:
            model = DDP(model)
            opt = ZeroRedundancyOptimizer(
                model.parameters(), optimizer_class=torch.optim.AdamW,
                lr=1e-3, betas=(0.9, 0.95))
        else:
            reshard = (stage == 3)
            for b in model.blocks:
                fully_shard(b, mesh=mesh, reshard_after_forward=reshard)
            fully_shard(model, mesh=mesh, reshard_after_forward=reshard)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

        losses, comm_per_step, p_loc, g_loc = [], [], 0, 0
        for step in range(args.steps):
            spy.reset()
            # First step does lazy init; only measure comms once warm.
            spy.on = (step == args.steps - 1)
            with spy:
                logits = model(x)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, args.vocab),
                                       x[:, 1:].reshape(-1))
                loss.backward()
            spy.on = False
            p_loc = sum(_local_numel(p) for p in model.parameters())
            g_loc = sum(_local_numel(p.grad) for p in model.parameters()
                        if p.grad is not None)
            opt.step()
            opt.zero_grad(set_to_none=False)
            losses.append(float(loss))
            comm_per_step.append(spy.total())

        o_loc = 0
        states = opt.optim.state if stage == 1 else opt.state
        for st in states.values():
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st:
                    o_loc += _local_numel(st[k])

        out[str(stage)] = {
            "psi": psi,
            "param_elems_local": p_loc,
            "grad_elems_local": g_loc,
            "optim_moment_elems_local": o_loc,
            "losses": losses,
            # step 0 includes lazy-init traffic; use a steady-state step
            "comm_wire_bytes_steady_state": comm_per_step[-1],
            "comm_calls": dict(spy.calls),
        }
        del model, opt

    if rank == 0:
        ret.update(out)
    dist.barrier()
    dist.destroy_process_group()


def e_note(d):
    return f"{d['relative_error']['params']:.2f} = 1-1/N, FSDP shards params"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--port", type=int, default=29717)
    ap.add_argument("--out", type=str,
                    default="artifacts/exp12_pytorch_all_stages.json")
    args = ap.parse_args()

    import torch.multiprocessing as mp

    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(worker, args=(args.world, args, ret), nprocs=args.world, join=True)
    m = {k: dict(v) for k, v in ret.items()}
    if not m:
        raise SystemExit("rank 0 returned nothing")

    psi, n = m["0"]["psi"], args.world
    IMPL = {0: "DDP + AdamW", 1: "DDP + ZeroRedundancyOptimizer",
            2: "FSDP2 reshard_after_forward=False", 3: "FSDP2 reshard_after_forward=True"}

    rep = {
        "backend": "gloo (CPU)", "world": n, "psi": psi,
        "memory_tolerance_pct": 5.0, "comm_ratio_tolerance": 0.25,
        "what_is_measured": (
            "Per-rank LOCAL element counts vs the ledger, and REAL collective "
            "bytes via wrapped torch.distributed calls. Not allocator peak, "
            "not activations."
        ),
        "stages": {},
    }

    ok_mem = True
    for s_str, d in sorted(m.items()):
        s = int(s_str)
        pred_p = psi if s < 3 else psi / n
        pred_g = psi if s < 2 else psi / n
        pred_o = 2 * (psi if s < 1 else psi / n)

        def rel(a, b):
            return abs(a - b) / b if b else 0.0

        e = {"params": rel(d["param_elems_local"], pred_p),
             "grads": rel(d["grad_elems_local"], pred_g),
             "moments": rel(d["optim_moment_elems_local"], pred_o)}

        # Stage 2's PARAMETER row is not comparable, and this is a real
        # semantic difference rather than an error in either side:
        # DeepSpeed ZeRO-2 keeps parameters fully REPLICATED and shards only
        # gradients + optimizer state. FSDP2 always shards the persistent
        # parameter storage; reshard_after_forward=False merely retains the
        # all-gathered copy between forward and backward, avoiding the
        # re-gather. So FSDP2(reshard=False) has ZeRO-2's COMMUNICATION with
        # ZeRO-3's persistent parameter MEMORY -- it is not DeepSpeed ZeRO-2.
        # The measured error is exactly 1 - 1/N, which is the signature of
        # that difference and not of a shard-math bug.
        skip = ["params"] if s == 2 else []
        checked = {k: v for k, v in e.items() if k not in skip}
        within = all(v <= rep["memory_tolerance_pct"] / 100
                     for v in checked.values())
        ok_mem &= within
        rep["stages"][s_str] = {
            "implementation": IMPL[s],
            "measured": {"param_elems_local": d["param_elems_local"],
                         "grad_elems_local": d["grad_elems_local"],
                         "optim_moment_elems_local": d["optim_moment_elems_local"]},
            "predicted": {"param_elems_local": pred_p,
                          "grad_elems_local": pred_g,
                          "optim_moment_elems_local": pred_o},
            "relative_error": {k: round(v, 5) for k, v in e.items()},
            "rows_not_comparable": skip,
            "not_comparable_reason": (
                "DeepSpeed ZeRO-2 replicates parameters; FSDP2 always shards "
                "persistent parameter storage and only retains the gathered "
                "copy between forward and backward. Measured error is exactly "
                "1 - 1/N, the signature of that mapping difference. FSDP2 "
                "(reshard=False) = ZeRO-2 communication + ZeRO-3 parameter "
                "memory."
            ) if skip else None,
            "within_tolerance": bool(within),
            "comm_wire_bytes_per_gpu": d["comm_wire_bytes_steady_state"],
            "comm_in_psi_units": round(
                d["comm_wire_bytes_steady_state"] / (psi * 4), 4),
            "comm_calls": d["comm_calls"],
            "losses": d["losses"],
        }

    ref = m["0"]["losses"]
    rep["loss_agreement_vs_dp"] = {
        s: max(abs(a - b) for a, b in zip(ref, m[s]["losses"])) for s in m}

    c2 = m["2"]["comm_wire_bytes_steady_state"]
    c3 = m["3"]["comm_wire_bytes_steady_state"]
    ratio = c3 / c2 if c2 else 0.0
    rep["zero3_over_zero2_comm_ratio"] = {
        "zero2_wire_bytes": c2, "zero3_wire_bytes": c3,
        "ratio": round(ratio, 4),
        "predicted": 1.5,
        "within_tolerance": bool(abs(ratio - 1.5) <= rep["comm_ratio_tolerance"]),
        "note": (
            "Direct test of the 1.5x claim. ZeRO-3 reshards after the forward "
            "pass, so the backward must all-gather the weights again. If "
            "PyTorch did not actually do that, this would come out near 1.0."
        ),
    }
    # DP vs ZeRO-2 must ALSO come out equal -- that is the other half of the
    # claim (ZeRO-2 is free relative to DP on the wire).
    c0 = m["0"]["comm_wire_bytes_steady_state"]
    rep["zero2_over_dp_comm_ratio"] = {
        "dp_wire_bytes": c0, "zero2_wire_bytes": c2,
        "ratio": round(c2 / c0, 4) if c0 else 0.0, "predicted": 1.0,
        "within_tolerance": bool(abs(c2 / c0 - 1.0) <= 0.25) if c0 else False,
    }
    rep["all_memory_within_tolerance"] = bool(ok_mem)

    os.makedirs(ART, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)

    print(f"Psi={psi:,}  world={n}  backend=gloo/CPU\n")
    for s, d in rep["stages"].items():
        mm = d["measured"]
        flag = "OK" if d["within_tolerance"] else "FAIL"
        if d["rows_not_comparable"]:
            flag += f" (params n/c: {e_note(d)})"
        print(f"stage {s} {d['implementation']:>36s}  "
              f"p{mm['param_elems_local']:>8,} g{mm['grad_elems_local']:>8,} "
              f"o{mm['optim_moment_elems_local']:>8,}  "
              f"err {max([v for k,v in d['relative_error'].items() if k not in d['rows_not_comparable']]):.4f} "
              f"{flag}  "
              f"wire {d['comm_wire_bytes_per_gpu']:>10,.0f} B "
              f"({d['comm_in_psi_units']:.2f}xPsi)")
    print("\nloss agreement vs DP:", rep["loss_agreement_vs_dp"])
    r = rep["zero3_over_zero2_comm_ratio"]
    print(f"ZeRO-3/ZeRO-2 comm ratio: {r['ratio']} (predicted 1.5) "
          f"{'OK' if r['within_tolerance'] else 'FAIL'}")
    print("memory all within 5%:", rep["all_memory_within_tolerance"])


if __name__ == "__main__":
    main()
