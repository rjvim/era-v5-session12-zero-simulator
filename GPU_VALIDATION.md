# Running the GPU validation (§9b's remaining gap)

Everything else in this repo is already measured. What is left needs a real
CUDA allocator: absolute peak memory, fragmentation, caching-allocator
behaviour, and device-side communication buffers. README §10.1 never claimed
to predict those; this closes it anyway.

## You need 2 GPUs, not 1

NCCL wants one process per device. On a single-GPU runtime the only thing you
can run is `world_size=1`, which shards nothing and measures nothing.

| Option | GPUs | Cost | Works? |
| --- | --- | --- | --- |
| **Kaggle Notebooks** (T4 x2) | 2 | free | **yes — use this** |
| Colab free (T4) | 1 | free | no |
| Colab Pro (A100/L4) | 1 | paid | no |
| vast.ai / RunPod / Lambda | 2+ | ~$1/hr | yes |

On Kaggle: *Notebook → Settings → Accelerator → **GPU T4 x2***. Confirm with
`!nvidia-smi` showing two devices before going further.

## Steps

**1.** New Kaggle notebook, accelerator set to T4 x2, internet enabled.

**2.** One cell:

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
!pip install -q -r requirements.txt
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

You should see two rows. If you see one, the accelerator setting did not take.

**3.** Run it:

```python
!python validate_against_fsdp.py --world 2 --stages 0,2,3
```

Roughly a minute. It prints per-stage measured/predicted reduction ratios and
writes `artifacts/exp10_fsdp_validation.json`.

**4.** Commit the result:

```python
!cat artifacts/exp10_fsdp_validation.json
```

Copy that into `artifacts/exp10_fsdp_validation.json` in the repo and push. Then
update README §9b to say it was run, with the numbers.

## How to read the outcome

The tolerance is fixed in the script, before running: **stage ratios within
15%**. Absolute bytes are explicitly *not* claimed — FSDP's caching allocator,
communication buffers and activation memory are modelled crudely or not at all
here.

- **Ratios inside 15%** → the ledger holds against a real allocator. Say so in
  §9b and move the remaining caveat in §10.1 to "verified".
- **Ratios outside 15%** → **the simulator is wrong and §3 needs correcting.**
  Do not widen the tolerance. That happened twice already in §9b (a local
  staging copy counted as network traffic; reduce-scatter sized from its
  scattered output) and both times the fix was to the instrument, not the
  threshold.

## If it errors

- `no accelerator detected` → accelerator setting did not apply; restart the
  session.
- NCCL init hangs → set `MASTER_PORT` to something else, e.g.
  `--port 29650`.
- `duplicate GPU detected` → you are on one GPU. See the table above.
- FSDP2 import error → older torch; `pip install -U torch`. The script uses
  `fully_shard`, which is the API verified on CPU in
  `validate_pytorch_all_stages.py`, not the older `FullyShardedDataParallel`.

## What you get out of it

Only the allocator-level claims. The sharding arithmetic and communication
volumes are already settled on CPU in §9b (all four stages, ZeRO-3/ZeRO-2
ratio exactly 1.5), and peak-ratio behaviour plus real activation bytes in
§9c. So this is the last narrow slice — worth doing if you have ten spare
minutes, not worth blocking a submission on.
