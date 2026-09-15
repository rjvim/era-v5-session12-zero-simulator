#!/usr/bin/env python3
"""Validate this simulator's memory model against a REAL ZeRO implementation.

Why this file exists separately: the assignment says to make sure the
simulation matches what ZeRO actually does. Sections 3 and 6 of the README
check against the ZeRO *paper's* formulas and against the numbers from the
lecture whiteboard. Those are both checks against arithmetic, not against a
running implementation -- and I want to be explicit that those are different
things.

This script closes that gap by running PyTorch FSDP (which implements the same
sharding as ZeRO-1/2/3 via ShardingStrategy) on a model matched to the
simulator's, and comparing measured per-GPU memory against the predicted
ledger.

IT IS NOT RUN IN THIS REPO. It needs torch + CUDA, which the machine I built
this on did not have. Any output committed here would be fabricated, so there
is none. Run it on Colab (T4 is enough) or any CUDA box:

    python validate_against_fsdp.py --world 2 --stage 3

Mapping (FSDP2):
    DP      =  DistributedDataParallel
    ZeRO-1  =  ZeroRedundancyOptimizer -- already verified on CPU, see 9b
    ZeRO-2  =  fully_shard(reshard_after_forward=False)
    ZeRO-3  =  fully_shard(reshard_after_forward=True)

NEEDS AT LEAST 2 GPUs. NCCL wants one process per device, so a single-GPU
runtime (free Colab) cannot measure sharding at all -- world_size=1 shards
nothing. Kaggle's free 2x T4 runtime works.

What "match" should mean here, stated before running so it can't be moved
afterwards:
  - The RATIOS between stages should match the simulator within ~15%.
  - The ABSOLUTE bytes will NOT match and are not expected to. FSDP allocates
    a caching allocator, communication buffers, and activation memory that the
    simulator models crudely or not at all. Predicting absolute
    torch.cuda.max_memory_allocated() is not what the simulator claims to do.
  - If ratios disagree by more than ~15%, the simulator is wrong and the
    README needs correcting, not the tolerance.
"""
from __future__ import annotations

import argparse
import json
import os


def build_model(d_model=1024, n_layer=8, vocab=4096):
    """A model big enough that optimizer states dominate allocator noise."""
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.ln2 = nn.LayerNorm(d)
            self.fc = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            import torch.nn.functional as F
            h = self.ln1(x)
            q, k, v = self.attn(h).chunk(3, dim=-1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(a)
            h = self.ln2(x)
            return x + self.fc2(F.gelu(self.fc(h)))

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

    return Net(), Block


def run(rank, world, stage, args, out):
    """Uses FSDP2 (fully_shard), not FSDP1.

    Deliberate: FSDP2 is the API actually exercised in
    validate_pytorch_all_stages.py on CPU, so the sharding semantics here are
    the ones already verified. FSDP1 was never run anywhere in this repo and
    would be an untested code path in a script I cannot test myself.

    stage 2 -> reshard_after_forward=False, stage 3 -> True. Note the caveat
    from README 9b: FSDP2 always shards persistent parameter storage, so its
    stage-2 parameter memory is ZeRO-3-like, not DeepSpeed ZeRO-2.
    """
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    model, Block = build_model(args.d_model, args.n_layer, args.vocab)
    psi = sum(p.numel() for p in model.parameters())
    model = model.cuda(rank)

    if stage != 0:
        mesh = init_device_mesh("cuda", (world,))
        reshard = (stage == 3)
        for b in model.blocks:
            fully_shard(b, mesh=mesh, reshard_after_forward=reshard)
        fully_shard(model, mesh=mesh, reshard_after_forward=reshard)
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    torch.cuda.reset_peak_memory_stats(rank)
    x = torch.randint(0, args.vocab, (args.batch, args.seq), device=rank)
    for _ in range(args.steps):
        loss = torch.nn.functional.cross_entropy(
            model(x)[:, :-1].reshape(-1, args.vocab), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=False)   # keep grads resident, as ZeRO does

    peak = torch.cuda.max_memory_allocated(rank)
    if rank == 0:
        out.update({"psi": psi, "stage": stage, "world": world,
                    "measured_peak_bytes": int(peak), "loss": float(loss)})
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--stages", type=str, default="0,2,3")
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", type=str, default="artifacts/exp10_fsdp_validation.json")
    args = ap.parse_args()

    import torch.multiprocessing as mp

    from zerosim.memory_model import model_state_bytes

    results = {}
    for stage in [int(s) for s in args.stages.split(",")]:
        mgr = mp.Manager()
        out = mgr.dict()
        mp.spawn(run, args=(args.world, stage, args, out),
                 nprocs=args.world, join=True)
        results[str(stage)] = dict(out)

    psi = results[list(results)[0]]["psi"]
    report = {"psi": psi, "world": args.world, "measured": results,
              "predicted": {}, "ratio_comparison": {}}
    for s, r in results.items():
        report["predicted"][s] = model_state_bytes(psi, args.world, int(s)).total

    base_m = results["0"]["measured_peak_bytes"]
    base_p = report["predicted"]["0"]
    for s in results:
        mr = base_m / results[s]["measured_peak_bytes"]
        pr = base_p / report["predicted"][s]
        report["ratio_comparison"][s] = {
            "measured_reduction_vs_dp": round(mr, 3),
            "predicted_reduction_vs_dp": round(pr, 3),
            "relative_error": round(abs(mr - pr) / pr, 3),
            "within_15pct": bool(abs(mr - pr) / pr < 0.15),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["ratio_comparison"], indent=2))
    bad = [s for s, v in report["ratio_comparison"].items() if not v["within_15pct"]]
    if bad:
        print(f"\nSTAGES OUTSIDE TOLERANCE: {bad}")
        print("The simulator is wrong. Fix README section 3, do not widen this.")
    else:
        print("\nAll stage ratios within 15% of the simulator's prediction.")


if __name__ == "__main__":
    main()
