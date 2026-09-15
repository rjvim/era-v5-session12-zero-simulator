#!/usr/bin/env python3
"""Measure REAL memory, rather than modelling it. Actually run.

Output committed to artifacts/exp13_measured_memory.json.

README limitation 1 said memory here is *modelled* -- the ledger counts
fp16/fp32 bytes as a real run would store them, but nothing was measured.
Limitation 6 said activations were *estimated* at ~34*B*T*d per layer. Both of
those are the honest description of a simulator, but both are checkable
without a GPU, and "no CUDA" is not a reason to leave them unchecked.

There is no torch.cpu.max_memory_allocated(), so this builds one: a
TorchDispatchMode registers every tensor storage it sees, weakref.finalize
decrements when the storage is freed, and the running total gives a true peak
of live tensor bytes. That is the same quantity CUDA's allocator reports,
measured on CPU.

Two measurements:

  1. PEAK live tensor bytes per rank, for all four stages, against the peak my
     simulator predicts. Absolute bytes will not match (dtype differs -- the
     simulator models fp16/fp32 mixed precision, this runs fp32) so the
     comparison is on RATIOS between stages, which is what the simulator
     actually claims.

  2. ACTIVATION bytes, isolated properly:
         peak(forward with grad) - peak(forward under no_grad)
     The difference is the tensors retained for the backward pass, i.e. the
     activations. Compared against the 34*B*T*d rule of thumb the engine uses.

Tolerances, fixed before running:
  - stage-to-stage peak ratios within 25% of predicted (looser than the
    element-count checks in exp12 because an allocator peak includes
    transients and fragmentation the ledger does not model -- that gap is the
    point of the measurement, not a defect in it)
  - the activation rule of thumb is reported as a ratio with no pass/fail; I
    want the number, not a verdict I chose the threshold for.

    python validate_measured_memory.py --world 2
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
import weakref

warnings.filterwarnings("ignore")

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def make_tracker():
    """A CPU analogue of torch.cuda.max_memory_allocated()."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class Tracker(TorchDispatchMode):
        def __init__(self):
            self.live = 0
            self.peak = 0
            self._seen: dict[int, int] = {}
            self.on = False

        def reset(self, baseline: int = 0):
            self.live = baseline
            self.peak = baseline
            self._seen.clear()

        def register(self, t):
            """Count a tensor's storage if we have not already counted it.

            Wrapped defensively: DTensor wrappers, meta/fake tensors and
            already-invalidated storages all raise on .data_ptr(), and a
            tracker that crashes the run is worse than one that misses a
            tensor.
            """
            try:
                if t.device.type != "cpu" or t.is_meta:
                    return
                st = t.untyped_storage()
                ptr = st.data_ptr()
                nb = st.nbytes()
            except Exception:
                return
            if ptr == 0 or ptr in self._seen:
                return
            self._seen[ptr] = nb
            self.live += nb
            if self.live > self.peak:
                self.peak = self.live

            def on_free(ptr=ptr, nb=nb, box=self):
                if box._seen.pop(ptr, None) is not None:
                    box.live -= nb

            try:
                weakref.finalize(st, on_free)
            except TypeError:
                pass

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if self.on:
                for t in _tensors(out):
                    self.register(t)
            return out

    return Tracker()


def _tensors(obj):
    import torch
    found, stack = [], [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            found.append(o)
            if hasattr(o, "to_local"):
                try:
                    found.append(o.to_local())
                except Exception:
                    pass
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return found


def build(d_model, n_layer, vocab):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        """Real attention on purpose: the 34*B*T*d rule of thumb is derived
        for attention blocks, so a probe model without attention would make
        the comparison meaningless (an earlier version of this script used a
        plain linear 'mix' and measured 0.31x, which said nothing about the
        rule and everything about the architecture mismatch)."""

        def __init__(self, d, n_head=4):
            super().__init__()
            self.h = n_head
            self.ln1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.ln2 = nn.LayerNorm(d)
            self.fc = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            B, T, D = x.shape
            hd = D // self.h
            q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
            q = q.view(B, T, self.h, hd).transpose(1, 2)
            k = k.view(B, T, self.h, hd).transpose(1, 2)
            v = v.view(B, T, self.h, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, T, D)
            x = x + self.proj(a)
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


def _loc(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _nbytes(t):
    t = _loc(t)
    return t.numel() * t.element_size()


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
    tr = make_tracker()
    out = {}
    x = torch.randint(0, args.vocab, (args.batch, args.seq))

    for stage in (0, 1, 2, 3):
        torch.manual_seed(0)
        model = Net()
        psi = sum(p.numel() for p in model.parameters())

        if stage == 0:
            model = DDP(model)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        elif stage == 1:
            model = DDP(model)
            opt = ZeroRedundancyOptimizer(model.parameters(),
                                          optimizer_class=torch.optim.AdamW,
                                          lr=1e-3)
        else:
            for b in model.blocks:
                fully_shard(b, mesh=mesh, reshard_after_forward=(stage == 3))
            fully_shard(model, mesh=mesh, reshard_after_forward=(stage == 3))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # warm up: lazy init, optimizer state creation
        for _ in range(2):
            F.cross_entropy(model(x)[:, :-1].reshape(-1, args.vocab),
                            x[:, 1:].reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=False)

        # resident model state, measured directly
        p_b = sum(_nbytes(p) for p in model.parameters())
        g_b = sum(_nbytes(p.grad) for p in model.parameters()
                  if p.grad is not None)
        states = opt.optim.state if stage == 1 else opt.state
        o_b = sum(_nbytes(v) for st in states.values() for k, v in st.items()
                  if k in ("exp_avg", "exp_avg_sq"))
        resident = p_b + g_b + o_b

        # peak during a full step
        tr.reset(baseline=resident)
        tr.on = True
        with tr:
            loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, args.vocab),
                                   x[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
        tr.on = False
        peak_step = tr.peak
        opt.zero_grad(set_to_none=False)

        # forward WITH grad: activations are retained
        tr.reset(baseline=0)
        tr.on = True
        with tr:
            y = model(x)
        tr.on = False
        peak_fwd_grad = tr.peak
        del y

        # forward under no_grad: nothing retained for backward
        tr.reset(baseline=0)
        tr.on = True
        with tr, torch.no_grad():
            y = model(x)
        tr.on = False
        peak_fwd_nograd = tr.peak
        del y

        out[str(stage)] = {
            "psi": psi,
            "resident_param_bytes": p_b,
            "resident_grad_bytes": g_b,
            "resident_optim_bytes": o_b,
            "resident_total_bytes": resident,
            "measured_peak_step_bytes": peak_step,
            "peak_forward_with_grad": peak_fwd_grad,
            "peak_forward_no_grad": peak_fwd_nograd,
            "measured_activation_bytes": max(0, peak_fwd_grad - peak_fwd_nograd),
        }
        del model, opt

    if rank == 0:
        ret.update(out)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--port", type=int, default=29951)
    ap.add_argument("--out", type=str,
                    default="artifacts/exp13_measured_memory.json")
    args = ap.parse_args()

    import torch.multiprocessing as mp

    from zerosim.memory_model import model_state_bytes

    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(worker, args=(args.world, args, ret), nprocs=args.world, join=True)
    m = {k: dict(v) for k, v in ret.items()}
    if not m:
        raise SystemExit("rank 0 returned nothing")

    psi, n = m["0"]["psi"], args.world
    IMPL = {0: "DDP + AdamW", 1: "DDP + ZeroRedundancyOptimizer",
            2: "FSDP2 reshard=False", 3: "FSDP2 reshard=True"}

    rep = {
        "backend": "gloo (CPU), fp32",
        "world": n, "psi": psi,
        "peak_ratio_tolerance_pct": 25.0,
        "how_peak_is_measured": (
            "TorchDispatchMode registers every tensor storage; "
            "weakref.finalize decrements on free; peak of the running total. "
            "The CPU analogue of torch.cuda.max_memory_allocated()."
        ),
        "how_activations_are_isolated": (
            "peak(forward with grad) - peak(forward under no_grad). The "
            "difference is what is retained for the backward pass."
        ),
        "caveat": (
            "This run is fp32 throughout; the simulator models mixed "
            "precision (fp16 params/grads, fp32 optimizer). So ABSOLUTE bytes "
            "are not comparable and only stage-to-stage RATIOS are."
        ),
        "stages": {},
    }

    base_meas = m["0"]["measured_peak_step_bytes"]
    base_pred = model_state_bytes(psi, n, 0).total
    ok = True
    for s_str, d in sorted(m.items()):
        s = int(s_str)
        pred_state = model_state_bytes(psi, n, s).total
        meas_ratio = base_meas / d["measured_peak_step_bytes"]
        pred_ratio = base_pred / pred_state
        rel = abs(meas_ratio - pred_ratio) / pred_ratio
        within = rel <= rep["peak_ratio_tolerance_pct"] / 100
        ok &= within
        rep["stages"][s_str] = {
            "implementation": IMPL[s],
            "measured_resident_bytes": d["resident_total_bytes"],
            "measured_peak_step_bytes": d["measured_peak_step_bytes"],
            "measured_activation_bytes": d["measured_activation_bytes"],
            "measured_peak_reduction_vs_dp": round(meas_ratio, 3),
            "simulator_model_state_reduction_vs_dp": round(pred_ratio, 3),
            "relative_error_on_reduction": round(rel, 4),
            "within_tolerance": bool(within),
        }

    # Activations: measured vs the 34*B*T*d*L rule the engine uses.
    est = 34 * args.batch * args.seq * args.d_model * args.n_layer * 4  # fp32
    # Stages 0 and 1 are the clean read for ACTIVATIONS: both are DDP, so
    # parameters are replicated and nothing parameter-shaped is gathered or
    # freed around the forward pass. Any difference between them would be a
    # real activation difference.
    ddp_act = {s: m[s]["measured_activation_bytes"] for s in ("0", "1")}
    rep["activation_rule_of_thumb"] = {
        "formula": "34 * B * T * d * n_layer * bytes",
        "estimate_bytes": est,
        "measured_bytes_ddp": ddp_act["0"],
        "measured_over_estimate": round(ddp_act["0"] / est, 3) if est else 0.0,
        "identical_across_ddp_stages": ddp_act["0"] == ddp_act["1"],
        "per_stage_retained_bytes": {
            s: m[s]["measured_activation_bytes"] for s in sorted(m)},
        "why_fsdp_stages_differ": (
            "The FSDP numbers are NOT evidence that activations differ by "
            "stage. My measurement is peak(forward with grad) - peak(forward "
            "under no_grad), which captures everything retained for the "
            "backward pass -- and under reshard_after_forward=False the "
            "GATHERED PARAMETERS are deliberately retained too, while "
            "reshard=True frees them. So ZeRO-2 retains more than DP and "
            "ZeRO-3 retains less, which is the transient side of the same "
            "memory/communication trade. Activations proper are unsharded by "
            "ZeRO; stages 0 and 1 are the clean comparison for that."
        ),
        "note": (
            "No pass/fail on the rule of thumb, on purpose -- I want the "
            "number, not a verdict against a threshold I chose myself."
        ),
    }
    rep["all_peak_ratios_within_tolerance"] = bool(ok)

    os.makedirs(ART, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)

    print(f"Psi={psi:,}  world={n}  fp32 on CPU\n")
    for s, d in rep["stages"].items():
        print(f"stage {s} {d['implementation']:>32s}  "
              f"resident {d['measured_resident_bytes']:>10,}  "
              f"MEASURED peak {d['measured_peak_step_bytes']:>10,}  "
              f"peak red {d['measured_peak_reduction_vs_dp']:>6.2f}x "
              f"(ledger {d['simulator_model_state_reduction_vs_dp']:>6.2f}x, "
              f"err {d['relative_error_on_reduction']:.3f}) "
              f"{'OK' if d['within_tolerance'] else 'FAIL'}")
    a = rep["activation_rule_of_thumb"]
    print(f"\nactivations (DDP, clean read) {a['measured_bytes_ddp']:,} B vs "
          f"34*B*T*d*L estimate {a['estimate_bytes']:,} B "
          f"-> {a['measured_over_estimate']}x")
    print("identical across the two DDP stages:",
          a["identical_across_ddp_stages"])
    print("retained bytes per stage:", a["per_stage_retained_bytes"])
    print("all peak ratios within 25%:", rep["all_peak_ratios_within_tolerance"])


if __name__ == "__main__":
    main()
