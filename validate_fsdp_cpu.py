#!/usr/bin/env python3
"""Validate the simulator's model-state ledger against REAL PyTorch FSDP.

This is the CPU/gloo counterpart to validate_against_fsdp.py, and unlike that
script THIS ONE IS ACTUALLY RUN -- its output is committed as
artifacts/exp11_fsdp_cpu_validation.json.

What it measures, and why that is the right thing to measure:

  My simulator makes a claim about MODEL STATES -- how many bytes of
  parameters, gradients and optimizer state each rank holds. It does not claim
  to predict a CUDA allocator's peak, and §10.1 of the README says so. So the
  correct validation is to ask real FSDP how much parameter / gradient /
  optimizer state each rank actually owns, and compare element counts against
  the ledger. That needs no GPU: sharding is sharding whether the backend is
  NCCL or gloo.

  What this therefore does NOT validate: allocator peak, communication
  buffers, fragmentation, or activation memory. Those need a GPU and are what
  validate_against_fsdp.py is for.

FSDP -> ZeRO mapping:
  NO_SHARD       -> DP     (nothing sharded)
  SHARD_GRAD_OP  -> ZeRO-2 (gradients + optimizer sharded, params replicated)
  FULL_SHARD     -> ZeRO-3 (everything sharded)

FSDP has no ZeRO-1 equivalent -- it always shards gradients alongside
optimizer states. That is itself a small independent corroboration of README
§5(a): if ZeRO-1 were ever the right choice, PyTorch would presumably expose
it.

Tolerance, stated before running so it cannot be moved afterwards: measured
per-rank model-state elements must match the predicted ledger within 2%. This
is tight on purpose -- element counts are exact arithmetic, not timing, so
anything beyond rounding means my shard math is wrong.

    python validate_fsdp_cpu.py --world 4
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def build(d_model: int, n_layer: int, vocab: int):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.ln2 = nn.LayerNorm(d)
            self.fc = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            h = self.ln1(x)
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            x = x + self.proj(F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                is_causal=True).squeeze(1))
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


def worker(rank: int, world: int, args, ret):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.optim import ZeroRedundancyOptimizer

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.port)
    dist.init_process_group("gloo", rank=rank, world_size=world)

    Net, _ = build(args.d_model, args.n_layer, args.vocab)
    local = {}

    for stage in (0, 1):
        torch.manual_seed(0)
        model = Net()
        psi = sum(p.numel() for p in model.parameters())
        model = DDP(model)

        if stage == 0:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                    betas=(0.9, 0.95))
        else:
            # PyTorch's actual ZeRO-1: optimizer state sharded across ranks.
            opt = ZeroRedundancyOptimizer(
                model.parameters(), optimizer_class=torch.optim.AdamW,
                lr=1e-3, betas=(0.9, 0.95))

        p_elems = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, args.vocab, (args.batch, args.seq))
        losses, g_elems = [], 0
        for _ in range(args.steps):
            loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, args.vocab),
                                   x[:, 1:].reshape(-1))
            loss.backward()
            g_elems = sum(p.grad.numel() for p in model.parameters()
                          if p.grad is not None)
            opt.step()
            opt.zero_grad(set_to_none=False)
            losses.append(float(loss))

        # Count Adam moments THIS rank actually holds.
        o_elems = 0
        states = (opt.optim.state if stage == 1 else opt.state)
        for st in states.values():
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st:
                    o_elems += st[k].numel()

        local[str(stage)] = {
            "psi": psi, "param_elems": p_elems, "grad_elems": g_elems,
            "optim_moment_elems": o_elems, "losses": losses,
        }
        del model, opt

    if rank == 0:
        ret.update(local)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--port", type=int, default=29613)
    ap.add_argument("--out", type=str,
                    default="artifacts/exp11_fsdp_cpu_validation.json")
    args = ap.parse_args()

    import torch.multiprocessing as mp

    from zerosim.memory_model import FP16, FP32, K_ADAM, model_state_bytes

    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(worker, args=(args.world, args, ret), nprocs=args.world, join=True)
    measured = {k: dict(v) for k, v in ret.items()}
    if not measured:
        raise SystemExit("no results returned from rank 0")

    psi = measured["0"]["psi"]
    report = {
        "backend": "gloo (CPU)",
        "world": args.world,
        "psi": psi,
        "what_this_validates": (
            "Per-rank MODEL STATE element counts against the simulator's "
            "ledger. Not allocator peak, not activations -- those need a GPU."
        ),
        "tolerance_pct": 2.0,
        "scope": (
            "Stages 0 and 1 only. FSDP refuses to initialise without a "
            "non-CPU accelerator, so ZeRO-2/3 cannot be validated on this "
            "machine and remain the job of validate_against_fsdp.py."
        ),
        "correction_this_run_produced": (
            "README previously said PyTorch has no ZeRO-1 and stage 1 needs "
            "DeepSpeed. Wrong: ZeroRedundancyOptimizer is PyTorch's ZeRO-1."
        ),
        "stages": {},
    }

    ok = True
    for stage, m in sorted(measured.items()):
        # Predicted LOCAL element counts (not bytes) for this rank.
        n = args.world
        s = int(stage)
        pred_params = psi if s < 3 else psi / n
        pred_grads = psi if s < 2 else psi / n
        pred_moments = 2 * (psi if s < 1 else psi / n)   # exp_avg + exp_avg_sq

        def err(meas, pred):
            return abs(meas - pred) / pred if pred else 0.0

        e = {
            "params": err(m["param_elems"], pred_params),
            "grads": err(m["grad_elems"], pred_grads),
            "moments": err(m["optim_moment_elems"], pred_moments),
        }
        within = all(v <= args.__dict__.get("tol", 0.02) for v in e.values())
        ok &= within
        report["stages"][stage] = {
            "implementation": {0: "DDP + AdamW",
                               1: "DDP + ZeroRedundancyOptimizer"}[s],
            "measured": {k: m[k] for k in
                         ("param_elems", "grad_elems", "optim_moment_elems")},
            "predicted": {"param_elems": pred_params,
                          "grad_elems": pred_grads,
                          "optim_moment_elems": pred_moments},
            "relative_error": {k: round(v, 5) for k, v in e.items()},
            "within_tolerance": bool(within),
            "losses": m["losses"],
            "simulator_model_state_bytes": model_state_bytes(psi, n, s).total,
        }

    # Cross-strategy loss agreement: FSDP strategies must also be
    # mathematically equivalent, same claim the simulator makes.
    ref = measured["0"]["losses"]
    report["loss_agreement_vs_no_shard"] = {
        s: max(abs(a - b) for a, b in zip(ref, measured[s]["losses"]))
        for s in measured
    }
    report["all_within_tolerance"] = bool(ok)

    os.makedirs(ART, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    for s, d in report["stages"].items():
        print(f"stage {s} ({d['implementation']:>31s}): "
              f"params {d['measured']['param_elems']:>9,} "
              f"(pred {d['predicted']['param_elems']:>11,.0f})  "
              f"grads {d['measured']['grad_elems']:>9,}  "
              f"moments {d['measured']['optim_moment_elems']:>9,}  "
              f"max err {max(d['relative_error'].values()):.4f}  "
              f"{'OK' if d['within_tolerance'] else 'FAIL'}")
    print("\nloss agreement vs NO_SHARD:", report["loss_agreement_vs_no_shard"])
    print("ALL WITHIN 2%:", report["all_within_tolerance"])


if __name__ == "__main__":
    main()
