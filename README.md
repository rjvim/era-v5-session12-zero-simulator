# ERA V5 — Session 12: ZeRO on 32 Virtual GPUs

**Assignment:** build 32 virtual GPUs, run a demo model on them, simulate
ZeRO-1/2/3, show how memory and computation change.

```
pip install -r requirements.txt   # numpy, matplotlib, pytest, nbformat
python run_experiments.py       # regenerates every artifact
python -m pytest tests/ -q      # 64 tests
jupyter notebook notebook.ipynb # narrated walkthrough, figures + outputs committed
```

No PyTorch — pure NumPy with a hand-written reverse-mode autodiff tape. That
was deliberate. If I had imported `deepspeed` and printed
`torch.cuda.max_memory_allocated()`, I would have demonstrated that DeepSpeed
works, not that I understand why. Every collective, every shard boundary and
every gradient here is code I can defend line by line, and every autodiff op is
finite-difference tested in `tests/test_autodiff.py`.

---

## How I decided to build it

My first instinct was a memory calculator: implement the formulas from the ZeRO
paper, print a table, done. I rejected that, because a table of formulas proves
nothing about whether I understand what the formulas describe — I could produce
exactly the right numbers with a completely wrong mental model of what
reduce-scatter does.

So I set one rule: **each virtual GPU may only touch bytes it actually owns.**
Under ZeRO-3 a rank holds 1/32nd of the parameters and genuinely cannot run a
forward pass until an all-gather hands it the rest. That constraint is what
makes the simulation capable of being *wrong*, and therefore worth running.

It gave me one very strong test, which became the backbone of everything:

> If sharding is implemented correctly, **the stage is invisible to the
> mathematics.** All four stages must walk identical optimization trajectories.

A wrong shard boundary, a reduce-scatter that sums where it should average, an
all-gather that reassembles out of order — all break this instantly. Measured
after 30 steps: **max |w − w_DP| = 0.0 for every stage.** Exactly zero, not
"within tolerance".

---

## 1. Why there is a problem at all

Under mixed-precision Adam, each parameter costs:

| Item | Bytes |
| --- | ---: |
| fp16 weights | 2 |
| fp16 gradients | 2 |
| fp32 master copy | 4 |
| fp32 momentum | 4 |
| fp32 variance | 4 |
| **total** | **16** |

The thing that genuinely surprised me: **the model is the small part.** The
weights you think of as "the model" are 2 of those 16 bytes. The optimizer
states are 12 — three quarters of the bill is bookkeeping.

That one fact explains the entire design of ZeRO and the order of its stages.
ZeRO-1 shards optimizer states *first* because that is where the money is.
Sharding the weights (ZeRO-3) comes last because it is the smallest win and the
most expensive to do.

| Stage | Sharded | Per-GPU model states |
| --- | --- | --- |
| DP | nothing | `16Ψ` |
| ZeRO-1 | optimizer states | `4Ψ + 12Ψ/N` |
| ZeRO-2 | + gradients | `2Ψ + 14Ψ/N` |
| ZeRO-3 | + parameters | `16Ψ/N` |

---

## 2. The collectives, in my own words

These three are the whole mechanism, and I want to be precise because I got the
third one wrong at first.

- **all-gather** — every rank contributes its slice, every rank ends with the
  whole buffer.
- **reduce-scatter** — every rank contributes a *full* buffer; they get
  combined, and each rank keeps only *one slice* of the result.
- **all-reduce** — every rank contributes and every rank receives the full
  combined result.

The bit that made it click: **all-reduce is not a primitive.** It is
reduce-scatter followed by all-gather. That is why it costs exactly twice as
much, and it's why the whole ZeRO ladder is possible — once a rank only updates
its own slice, it doesn't need the all-gather half of the gradient reduction,
so it simply doesn't do it. The saving isn't a clever trick; it's declining to
do work you never needed.

Ring costs live in `zerosim/fabric.py`: all-gather and reduce-scatter move
`(N−1)/N · S` bytes per GPU, all-reduce moves `2(N−1)/N · S`. I put those in
the fabric rather than hard-coding the famous 2Ψ / 3Ψ results, so the results
have to *emerge*. They do, which is the point.

---

## 3. Memory results — 32 virtual GPUs

A 27,072-parameter transformer (2 layers, 2 heads, d=32), 30 steps, on a
syllable-level mantra corpus, across 4 nodes of 8.

| Stage | Model states/GPU | Reduction | **Peak/GPU** | **Peak red.** | Comm/GPU/step | In Ψ | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DP | 433,152 B | 1.00× | 502,784 B | 1.00× | 104,904 B | 1.94Ψ | 3.243 → 0.510 |
| ZeRO-1 | 118,440 B | 3.66× | 188,072 B | 2.67× | 104,904 B | 1.94Ψ | 3.243 → 0.510 |
| ZeRO-2 | 65,988 B | 6.56× | 135,620 B | 3.71× | 104,904 B | 1.94Ψ | 3.243 → 0.510 |
| ZeRO-3 | 13,536 B | **32.00×** | 84,860 B | **5.92×** | 157,356 B | 2.91Ψ | 3.243 → 0.510 |

**ZeRO-3's reduction is exactly 32.00× — the world size.** Not approximately.
`16Ψ → 16Ψ/N`, nothing left replicated.

**The communication ratio is exactly 1.5×.** 157,356 / 104,904 = 1.5. The
`1.94Ψ` and `2.91Ψ` are ring-ideal `2Ψ` and `3Ψ` scaled by `(N−1)/N = 31/32`.

**The loss curve matters.** I used a structured corpus rather than random
tokens specifically because "all four stages produced the same loss" is nearly
worthless if the model learned nothing. 3.243 → 0.510 means the trajectory is
non-trivial, so agreeing on it is informative.

### 3b. The 32× is model states. Real peak falls 5.92×.

I nearly published only the 32× figure, and I think that would have been
cherry-picking my own result. The two reduction columns above disagree, and
the gap is the most important thing in this section.

**ZeRO does not shard activations. At all.** They are byte-identical on every
stage — 69,632 B here — so they set a hard floor under peak memory. As you
shard model states away, activations become a larger and larger share of
what's left:

| Stage | Activation share of peak |
| --- | ---: |
| DP | 14% |
| ZeRO-1 | 37% |
| ZeRO-2 | 51% |
| ZeRO-3 | **82%** |

By ZeRO-3, four fifths of the memory is something ZeRO doesn't touch. That is
why 32× on model states becomes 5.92× on peak.

And at realistic scale it is far worse. Ψ = 27.9B, 64 layers, d = 7168,
seq = 8192, batch 1 per GPU, N=8, **no activation checkpointing**:

| Stage | Model states | + Activations | Total | Fits 80 GB? |
| --- | ---: | ---: | ---: | --- |
| DP | 446.4 GB | 255.6 GB | 702.0 GB | ✗ |
| ZeRO-1 | 153.4 GB | 255.6 GB | 409.0 GB | ✗ |
| ZeRO-2 | 104.6 GB | 255.6 GB | 360.2 GB | ✗ |
| ZeRO-3 | 55.8 GB | 255.6 GB | **311.4 GB** | **✗** |

**Nothing fits.** Activations alone are 255.6 GB — more than three 80 GB cards
— and sharding model states 32× is almost beside the point next to that.

This changes a claim I make later in §4 and I want to flag it here rather than
let the two sections quietly contradict each other: when I say "ZeRO-3 is the
only stage that fits at 27.9B", that is a statement about **model states
only**. Counting activations, no stage fits without activation
checkpointing. That is precisely why checkpointing is mandatory at scale
rather than an optimisation, and why the ZeRO paper has a separate ZeRO-R for
sharding activations. My simulator implements neither, so the table above is
an **upper bound** on activation cost, not a prediction of a real run.

---

## 4. Compute — what changes, and what doesn't

This is the part I initially had wrong in my own head, so I want to state it
carefully.

**ZeRO does not change FLOPs. At all.** `exp7` asserts it directly:
1.741×10¹⁵ FLOPs per GPU per step, *identical* across all four stages. It has
to be — that's the same claim as the bit-identical weights. Same arithmetic,
same data.

So "show how computation changes" cannot mean the arithmetic gets cheaper. It
means **achieved throughput** changes, because communication sits in the
critical path and ZeRO-3 puts more of it there.

H100, Ψ = 27.9B, 80% of communication overlapped with compute:

| Setup | Stage | Fits in 80GB? | tok/s/GPU | Step lost to comms |
| --- | --- | --- | ---: | ---: |
| 8 GPU, 1 node | DP / Z1 / Z2 | ✗ / ✗ / ✗ | 1,864 | 1.0% |
| 8 GPU, 1 node | ZeRO-3 | ✓ | 1,802 | 4.29% |
| 64 GPU, 8 nodes | DP / Z1 | ✗ | 1,692 | 10.1% |
| 64 GPU, 8 nodes | ZeRO-2 | ✓ | 1,692 | 10.1% |
| 64 GPU, 8 nodes | ZeRO-3 | ✓ | **1,251** | **33.55%** |

Two things I would not have predicted before running it:

**ZeRO-3's cost depends almost entirely on whether you crossed a node
boundary.** Inside one node it costs 3.3% throughput — essentially free. Across
8 nodes it costs 26%. Same stage, same model. The 1.5× multiplier is constant;
what changes is the bandwidth it multiplies against.

**At 27.9B on 80GB H100s with 8 GPUs, ZeRO-3 is the only stage whose model
states fit.** DP needs 446 GB, ZeRO-1 153, ZeRO-2 105, ZeRO-3 55.8. So the
throughput comparison against DP is hypothetical — you cannot run DP. That
reframed the question for me: often the choice isn't "which is faster", it's
"which ones are even available".

Important caveat, carried forward from §3b: the "fits" column here counts
**model states only**. Add activations and no stage fits on an 80 GB card
without activation checkpointing. Fitting is a joint property of ZeRO stage
*and* activation strategy, and this simulator only models the first.

### Where the trade actually bites

A model small enough that every stage fits, so they're comparable on speed
(N=8, H100, 80% overlap):

| Model | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- | --- |
| 1B | 16.0 GB · 31,760 tok/s | 5.5 GB · 31,760 | 3.8 GB · 31,760 | 2.0 GB · 30,474 |
| 3B | 48.0 GB · 16,065 | 16.5 GB · 16,065 | 11.2 GB · 16,065 | 6.0 GB · 15,098 |
| 7B | **112 GB · doesn't fit** | 38.5 GB · 8,080 | 26.2 GB · 8,080 | 14.0 GB · 7,515 |

Read down a column: higher stage always costs throughput. Read across the 7B
row: higher stage is what makes it run at all. **Take the lowest stage that
fits** — there is never a reason to shard more than you have to.

---

## 5. Three things I got wrong

Including these because the assignment asks whether *I* understood this, and
honestly I understood it by getting it wrong first.

**(a) I had ZeRO-1 doing a full all-reduce.** My first engine used all-reduce
for stages 0 and 1, reduce-scatter for 2 and 3, which made ZeRO-1 look like it
cost 3Ψ. Wrong: a ZeRO-1 rank only updates its own slice, so reduce-scatter
delivers exactly what it needs. ZeRO-1 and ZeRO-2 have **identical**
communication — both 104,904 B here.

The consequence is worth stating plainly: **ZeRO-2 is a strict improvement over
ZeRO-1.** Better memory, identical wire cost, identical math. There is no
configuration where ZeRO-1 is the right answer. I did not know that before
building this, and I don't think it's obvious from the lecture table, where
they look like two adjacent rungs with a trade-off between them.

My communication test (`test_zero3_communicates_more`) caught it by failing.

**(b) My ZeRO-3 gathered weights only once.** I had it all-gather before the
step and hold the full vector across forward *and* backward. That's wrong in a
way that mattered: ZeRO-3 frees each bucket immediately after the forward uses
it — **that freeing is the entire memory saving** — so the backward finds
nothing resident and has to gather all over again. Two gathers plus one
reduce-scatter is where `3Ψ` comes from. My version was quietly claiming
ZeRO-3's memory savings while paying ZeRO-2's communication. Same test caught
it: the ratio came out 1.0 instead of 1.5.

**(c) I guessed a number instead of deriving it.** Explaining ZeRO-2 at N=8
before building anything, I said "~95 GB" by eyeballing between the stated 153
and 55. The arithmetic says **104.62 GB** (`3.75Ψ`). Interpolating linearly on
a curve of the form `a + b/N` is just wrong. Corrected everywhere here, and I'm
leaving the error recorded rather than quietly fixing it.

---

## 6. Checking against real ZeRO

The session whiteboard gave six numbers but never stated the model size. I
solved the two clearly-stated N=8 values (`16Ψ = 447`, `2Ψ = 55`) to recover
**Ψ ≈ 27.9B parameters**, then checked whether that single constant predicts
everything else.

| N | Stage | Computed | Board | Error |
| ---: | --- | ---: | ---: | ---: |
| 8 | DP | 446.4 GB | 447 | 0.6 |
| 8 | ZeRO-1 | 153.45 GB | 153 | 0.45 |
| 8 | ZeRO-2 | 104.62 GB | — | — |
| 8 | ZeRO-3 | 55.80 GB | 55 | 0.8 |
| 64 | ZeRO-1 | 116.83 GB | 117 | 0.17 |
| 64 | ZeRO-2 | 61.90 GB | — | — |
| 64 | ZeRO-3 | 6.97 GB | 7 | 0.03 |

Worst error 0.8 GB on a 447 GB figure. **The N=64 values were never used to fit
Ψ**, so they're genuinely out-of-sample.

Then an independent check from a completely different direction. The board
separately said DP moves "around 120 GB per GPU per step", carried by NVLink in
"about a quarter second" and InfiniBand in "about 2.4 seconds". A DP all-reduce
at Ψ = 27.9B in fp16 is `2Ψ × 2 bytes` = **111.6 GB**; ÷450 GB/s = **0.248 s**;
÷45 GB/s = **2.48 s**. All three within 10%, from a Ψ recovered purely off the
memory table. That the communication arithmetic and the memory arithmetic agree
on the same hidden constant is the strongest evidence I have that I'm modelling
the same thing the lecture was.

**One correction to the transcript.** ZeRO-2 at N=64 is transcribed as 72. No
ZeRO-2 form gives 72 for the Ψ that fits the other five values; `2Ψ + 14Ψ/64` =
61.90 GB. Most likely a 62 → 72 transcription slip — flagging rather than
silently substituting my own number.

---

## 7. Bucket size

ZeRO-3 gathers weights bucket-by-bucket and frees each after use, so the
resident transient is one bucket, not the whole model. Sweeping at N=32:

| Bucket | Peak/GPU | Comm/step |
| --- | ---: | ---: |
| Ψ (whole model) | 137,312 B | 157,356 B |
| Ψ/2 | 110,240 B | 157,356 B |
| Ψ/8 | 89,936 B | 157,356 B |
| Ψ/32 | 84,860 B | 157,356 B |

**Communication is completely flat. Peak memory falls 1.62× for free.** The
same bytes cross the wire either way; only the resident transient shrinks. This
is what Rohan was drawing when he said to go as small as possible.

The floor is *latency* — past some point you're issuing many tiny transfers and
per-call overhead dominates. My cost model is pure bandwidth and doesn't
capture that, so this sweep overstates the benefit at the small end. That's a
limitation of my model, not a finding.

---

## 8. Pros and cons

**ZeRO-1** — 3.66× memory, identical comms to DP. But dominated by ZeRO-2 in
every respect. I wouldn't use it.

**ZeRO-2** — 6.56×, still zero extra communication over DP. **The default.**

**ZeRO-3** — reduction equal to world size, and the only stage where adding
GPUs increases trainable model size. Costs 1.5× communication, and critically
that cost now lands *inside* forward and backward rather than only at the
optimizer step, so it overlaps worse. Cheap within a node (3.3%), expensive
across nodes (26%).

Largest trainable model on 80GB H100s at N=32:

| DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| ---: | ---: | ---: | ---: |
| 5.00B | 18.29B | 32.82B | **160.0B** |

**Decision rule: take the lowest stage that fits.** Memory is the binding
constraint, not compute. You're not making the GPU think faster — you're making
a bigger model fit inside it. And read that DP column carefully: it stays 5.00B
**no matter how many GPUs you add** (verified as a test). Adding GPUs to plain
DP buys throughput and exactly zero model size. That's the whole motivation for
ZeRO in one line.

### What I'd argue for the V5 run

- **Stage:** ZeRO-2 by default. Move to ZeRO-3 only when the target genuinely
  doesn't fit, because at 8 nodes ZeRO-3 costs ~26% throughput.
- **Topology:** fill nodes before adding them. The node boundary is a 10×
  bandwidth cliff, not a slope — 8→16 GPUs costs more than 16→64.
- **Bucket size:** small. Volume unchanged, peak memory drops, bounded below
  only by per-call latency.
- **FP8:** orthogonal to all of this. It shrinks the 2Ψ fp16 terms but not the
  12Ψ fp32 optimizer states, so the sharding argument is completely unchanged.
  Worth being explicit, because it's easy to assume FP8 reduces the need for
  ZeRO. It doesn't.

---

## 9. Repository layout

```
zerosim/autodiff.py      reverse-mode autodiff tape (finite-difference tested)
zerosim/model.py         transformer over ONE flat parameter vector
zerosim/device.py        VirtualGPU: named memory ledger + peak tracking
zerosim/fabric.py        all-gather / reduce-scatter / all-reduce + ring costs
zerosim/memory_model.py  closed-form ZeRO memory arithmetic
zerosim/compute.py       FLOPs and achieved-throughput model
zerosim/threaded.py      real CPU-thread execution of the 32 ranks
zerosim/plots.py         the four figures
zerosim/engine.py        stages 0-3, actually executed
zerosim/data.py          syllable-level mantra corpus
run_experiments.py       regenerates artifacts/ (10 experiments)
validate_fsdp_cpu.py     RUN: validates ZeRO-1 vs PyTorch ZeroRedundancyOptimizer
tools/build_notebook.py  regenerates notebook.ipynb
requirements.txt         core deps; -validation.txt is torch, for 9b only
validate_against_fsdp.py UNRUN: needs a GPU, for ZeRO-2/3 (see 9b)
tests/                   64 tests
notebook.ipynb           narrated walkthrough, 4 figures, outputs committed
```

Every number quoted in this README is re-asserted against its generating
artifact in `tests/test_invariants.py`, so prose and code can't drift. The
figure count and the presence of committed notebook outputs are asserted too.

`notebook.ipynb` is generated from `tools/build_notebook.py` rather than
hand-edited. I briefly considered deleting that script so the notebook
wouldn't look machine-produced, and decided that was the wrong instinct:
a reproducible notebook is better engineering than a hand-edited one, and
hiding the generator to manage appearances is exactly the kind of thing this
README shouldn't be doing. So it's there, and this is why.

The 32 ranks can be executed on real CPU threads
(`ZeROEngine(..., threaded=True)`), with a barrier at each collective so the
synchronisation points are visible. `tests/test_threaded.py` asserts it gives
bit-identical results to the sequential path. It is *not* faster — NumPy holds
the GIL through the small tape ops that dominate a model this size — and I've
kept it because it makes the execution model faithful, not because it wins
wall-clock.

---

## 9b. Validating against a real ZeRO implementation

§3 checks my simulator against the ZeRO **paper's arithmetic** and §6 against
the lecture whiteboard. Both are checks against formulas. I wanted at least
one check against a *running* implementation.

### What I could run: ZeRO-1, verified (`validate_fsdp_cpu.py`)

`torch.distributed.optim.ZeroRedundancyOptimizer` is PyTorch's own ZeRO-1 —
it shards optimizer state across ranks — and it runs on the gloo/CPU backend.
So this one is **actually executed, with output committed** to
`artifacts/exp11_fsdp_cpu_validation.json`. Ψ = 858,880 over 4 ranks:

| Stage | Implementation | Params | Grads | Adam moments | Predicted | Rel. err |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | DDP + AdamW | 858,880 | 858,880 | 1,717,760 | 2Ψ | **0.0000** |
| 1 | DDP + ZeroRedundancyOptimizer | 858,880 | 858,880 | 429,568 | 429,440 | **0.0003** |

Two results worth separating.

**The ledger is right.** Real ZeRO-1 holds full parameters, full gradients, and
a quarter of the Adam moments on 4 ranks — exactly what §1's table predicts.
Params and grads match to zero error.

**The math is stage-invariant, confirmed externally.** Loss agreement between
DDP and DDP+ZeRO-1 is **exactly 0.0** across every step. My central claim —
"the stage is invisible to the mathematics" — now rests on PyTorch's
implementation as well as my own, which is a much better place for it to rest.

And the 0.03% discrepancy is informative rather than noise: **real ZeRO-1
shards at parameter granularity, not element granularity.** It assigns whole
tensors to ranks greedily, so its shards are slightly uneven (429,568 vs a
perfectly-even 429,440). My simulator splits the flat vector at element
boundaries, so its shards are exactly equal. That is a genuine modelling
difference, it is small, and it is in §10.11 rather than rounded away.

### What I could not run: ZeRO-2 and ZeRO-3

FSDP is the ZeRO-2/ZeRO-3 equivalent (`SHARD_GRAD_OP` / `FULL_SHARD`), and it
**refuses to initialise without a GPU** — "FSDP needs a non-CPU accelerator
device". There is no GPU on the machine I built this on, so stages 2 and 3
could not be checked against a real implementation.

`validate_against_fsdp.py` is the harness for that, ready to run on Colab:

```
pip install -r requirements-validation.txt
python validate_against_fsdp.py --world 2 --stages 0,2,3
```

**It has not been run and there is no committed output for it.** Fabricating
numbers would be worse than an admitted gap. Its tolerance is written into the
script *before* running so it cannot be moved afterwards: stage **ratios**
within 15%; **absolute** bytes explicitly not claimed, since FSDP's caching
allocator, communication buffers and real activation memory are things this
simulator models crudely or not at all. If the ratios disagree by more than
15%, **my simulator is wrong and §3 needs correcting** — not the tolerance.

### A claim of mine this exercise corrected

An earlier draft of this README said PyTorch has no ZeRO-1 equivalent and that
stage 1 needs DeepSpeed, and I used that as supporting evidence for §5(a)
("ZeRO-2 dominates ZeRO-1"). **That was wrong** — `ZeroRedundancyOptimizer` is
exactly PyTorch's ZeRO-1. §5(a)'s actual argument stands on its own (identical
communication, measured), but it lost a piece of corroboration it was never
entitled to, and going looking for something runnable is what caught it.

---

## 10. Limitations

Stated plainly, because several change how the numbers above should be read.

1. **Memory is modelled, not measured.** Compute runs in float64 NumPy; the
   ledger accounts fp16/fp32 as a real run would store them. **Ratios are
   exact; absolute byte counts describe a hypothetical mixed-precision run.**
   And the model is validated against the paper's formulas and the lecture
   numbers, **not against a running ZeRO implementation** — see §9b for the
   harness that would close this, which I could not execute.
2. **`compute_fraction_of_step` is not MFU.** It is `compute_s / step_s`, and
   `compute_s` is itself derived from the assumed TFLOP/s — circular as an
   efficiency metric, and it would read ~1.0 for a perfectly overlapped run
   with terrible kernels. It measures only how much of the step is lost to
   exposed communication. Real MFU needs measured wall-clock against a hardware
   ceiling.
3. **No latency term.** Pure bandwidth, so the bucket sweep overstates the
   benefit of very small buckets.
4. **Overlap is a parameter, not a simulation.** I assume a fraction and halve
   it for ZeRO-3. Real DeepSpeed/FSDP overlap depends on scheduling I don't
   model.
5. **The 6Ψ FLOPs rule is an estimate**, and on a 27k-parameter model a poor one
   — `exp9` puts my hand-counted matmuls at 0.94× the estimate, because the
   output head and attention are a large share of a tiny model. At 27.9B the
   rule is good to a few percent. Recorded rather than quoted as exact.
6. **Activations are estimated** (`~34·B·T·d` per layer), not derived from the
   real tape, and **no activation checkpointing is modelled**. Because ZeRO
   does not shard activations, they are identical across stages and never
   distort the *comparison* — but they dominate absolute peak (82% of ZeRO-3's
   peak here), so the realistic table in §3b is an upper bound on memory, not
   a prediction. A real run at that scale would use checkpointing and/or
   ZeRO-R, neither of which is implemented.
7. **Ring collectives assumed.** Tree/hierarchical algorithms have different
   constants.
8. **The model is tiny** (27,072 parameters) — deliberate, so all 32 ranks run
   honestly on a laptop and the equivalence test can be exact. Realistic scale
   lives in §4 and §6.
9. **Ψ ≈ 27.9B is recovered, not stated.** Consistent with six board numbers
   plus three independent communication figures, but never said out loud.
10. **No tensor or pipeline parallelism.** ZeRO is data parallelism with
    sharding. The layer-splitting drawn in the session is a different axis and
    isn't simulated here.
11. **I shard at element granularity; real ZeRO-1 shards at parameter
    granularity.** My flat vector splits at exact element boundaries, giving
    perfectly even shards. `ZeroRedundancyOptimizer` assigns whole tensors to
    ranks greedily, so its shards are slightly uneven — measured at 0.03% off
    even, in §9b. Negligible here, but it would grow for a model with a few
    very large tensors (a big embedding table, say) and few ranks.
