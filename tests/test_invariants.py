"""Every number that appears in README.md is re-asserted here against the
artifact that generated it.

Standard pattern from Session 6 onward: prose and code cannot drift, because
if someone edits a figure in the README without re-running the experiments,
this file fails.
"""
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
README = os.path.join(ROOT, "README.md")


def art(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def readme():
    with open(README) as f:
        return f.read()


# ---------------------------------------------------------------- experiment 1
def test_model_and_world():
    a = art("exp1_simulated_run.json")
    assert a["config"]["params"] == 27072
    assert a["config"]["world_size"] == 32
    assert a["config"]["gpus_per_node"] == 8


def test_model_state_bytes_per_stage():
    a = art("exp1_simulated_run.json")["stages"]
    assert a["0"]["model_state_bytes_per_gpu"] == 433152
    assert a["1"]["model_state_bytes_per_gpu"] == 118440
    assert a["2"]["model_state_bytes_per_gpu"] == 65988
    assert a["3"]["model_state_bytes_per_gpu"] == 13536


def test_memory_reduction_factors():
    a = art("exp1_simulated_run.json")["memory_reduction_vs_dp"]
    assert a["0"] == pytest.approx(1.00, abs=0.01)
    assert a["1"] == pytest.approx(3.66, abs=0.01)
    assert a["2"] == pytest.approx(6.56, abs=0.01)
    # ZeRO-3 reduction must be exactly the world size: 16*Psi -> 16*Psi/N.
    assert a["3"] == pytest.approx(32.0, abs=0.01)


def test_communication_volume_and_ratio():
    a = art("exp1_simulated_run.json")["stages"]
    for s in ("0", "1", "2"):
        assert a[s]["comm_bytes_per_gpu_per_step"] == pytest.approx(104904, abs=1)
        assert a[s]["comm_volume_in_psi"] == pytest.approx(1.94, abs=0.01)
    assert a["3"]["comm_bytes_per_gpu_per_step"] == pytest.approx(157356, abs=1)
    assert a["3"]["comm_volume_in_psi"] == pytest.approx(2.91, abs=0.01)
    ratio = a["3"]["comm_bytes_per_gpu_per_step"] / a["0"]["comm_bytes_per_gpu_per_step"]
    assert ratio == pytest.approx(1.5, abs=0.001), "ZeRO-3 must be exactly 1.5x"


def test_zero1_and_zero2_communicate_identically():
    """ZeRO-2 is a free memory win over ZeRO-1 -- same wire cost."""
    a = art("exp1_simulated_run.json")["stages"]
    assert (a["1"]["comm_bytes_per_gpu_per_step"]
            == a["2"]["comm_bytes_per_gpu_per_step"])
    assert a["2"]["model_state_bytes_per_gpu"] < a["1"]["model_state_bytes_per_gpu"]


def test_all_stages_bit_identical():
    a = art("exp1_simulated_run.json")
    for s, d in a["equivalence_max_abs_weight_delta_vs_dp"].items():
        assert d == 0.0, f"stage {s} drifted from DP by {d}"


def test_model_actually_learned():
    a = art("exp1_simulated_run.json")["stages"]
    assert a["0"]["loss_first"] == pytest.approx(3.243, abs=0.001)
    for s in "0123":
        assert a[s]["loss_last"] == pytest.approx(0.510, abs=0.001)
        assert a[s]["loss_last"] < 0.2 * a[s]["loss_first"]


# ---------------------------------------------------------------- experiment 2
def test_board_reproduction_within_1gb():
    a = art("exp2_board_reproduction.json")
    assert a["recovered_psi"] == 27.9e9
    assert a["k_adam"] == 12
    errs = []
    for N, row in a["deltas_vs_transcript"].items():
        for s, d in row.items():
            errs.append(d["abs_err"])
    assert max(errs) <= 0.8, f"worst board mismatch {max(errs)} GB"
    assert len(errs) == 6, "should be checking 6 stated board values"


def test_board_table_values():
    t = art("exp2_board_reproduction.json")["table"]
    assert t["8"]["0"] == pytest.approx(446.4, abs=0.01)
    assert t["8"]["1"] == pytest.approx(153.45, abs=0.01)
    assert t["8"]["2"] == pytest.approx(104.62, abs=0.01)
    assert t["8"]["3"] == pytest.approx(55.80, abs=0.01)
    assert t["64"]["1"] == pytest.approx(116.83, abs=0.01)
    assert t["64"]["2"] == pytest.approx(61.90, abs=0.01)
    assert t["64"]["3"] == pytest.approx(6.97, abs=0.01)


def test_transcript_crosscheck():
    c = art("exp6_scaling_economics.json")["transcript_crosscheck"]
    assert c["dp_allreduce_gb_per_gpu"] == pytest.approx(111.6, abs=0.1)
    assert c["nvlink_seconds"] == pytest.approx(0.248, abs=0.001)
    assert c["infiniband_seconds"] == pytest.approx(2.48, abs=0.01)
    # within 10% of the quoted board figures, having never been fit to them
    assert abs(c["dp_allreduce_gb_per_gpu"] - 120) / 120 < 0.10
    assert abs(c["nvlink_seconds"] - 0.27) / 0.27 < 0.10
    assert abs(c["infiniband_seconds"] - 2.4) / 2.4 < 0.10


# ---------------------------------------------------------------- experiment 5
def test_bucket_sweep():
    s = art("exp5_bucket_sweep.json")["sweep"]
    assert s["1"]["peak_bytes_per_gpu"] == 137312
    assert s["32"]["peak_bytes_per_gpu"] == 84860
    ratio = s["1"]["peak_bytes_per_gpu"] / s["32"]["peak_bytes_per_gpu"]
    assert ratio == pytest.approx(1.62, abs=0.01)
    vols = {k: v["comm_bytes_per_gpu_per_step"] for k, v in s.items()}
    assert len(set(vols.values())) == 1, "bucket size must not change wire volume"


# ---------------------------------------------------------------- experiment 4
def test_largest_model_h100_32gpu():
    a = art("exp4_largest_model.json")["H100_80GB"]["32"]
    assert a["0"] == pytest.approx(5.00, abs=0.01)
    assert a["1"] == pytest.approx(18.29, abs=0.01)
    assert a["2"] == pytest.approx(32.82, abs=0.01)
    assert a["3"] == pytest.approx(160.0, abs=0.01)


def test_dp_capacity_independent_of_world_size():
    """Adding GPUs to plain DP buys zero extra model size. The whole point."""
    a = art("exp4_largest_model.json")["H100_80GB"]
    assert len({a[n]["0"] for n in a}) == 1


# ---------------------------------------------------------------- prose lock
NUMBERS_IN_README = [
    "27,072", "433,152", "118,440", "65,988", "13,536",
    "3.66", "6.56", "32.00", "104,904", "157,356",
    "3.243", "0.510", "446.4", "153.45", "104.62", "55.80",
    "116.83", "61.90", "6.97", "111.6", "0.248", "2.48",
    "137,312", "84,860", "160.0",
    "1.741", "1,864", "1,802", "1,692", "1,251", "8,080", "7,515",
    "31,760", "16,065", "0.94", "33.55", "4.29",
    "502,784", "188,072", "135,620", "84,860", "5.92", "69,632",
    "255.6", "311.4", "702.0", "409.0", "360.2", "82%",
    # exp12 (supersedes exp11's ZeRO-1-only run as the quoted result)
    "660,736", "1,321,472", "330,496", "330,368", "165,184",
    "3,964,416", "5,946,624", "2.25", "1.667", "1.8", "4.8e-07",
    # exp13 measured memory
    "31,183,572", "17,310,164", "17,180,116", "3,884,032", "4,835,840",
    "3,746,816", "0.436", "111.4", "167.2", "557.8", "216.0", "264.8",
]


def test_readme_quotes_the_artifact_numbers(readme):
    missing = [n for n in NUMBERS_IN_README if n not in readme]
    assert not missing, f"README no longer quotes: {missing}"


def test_readme_has_limitations_section(readme):
    assert re.search(r"^##+ .*Limitations", readme, re.M | re.I), \
        "every submission carries an explicit limitations section"


# ------------------------------------------------------- compute / throughput
def test_flops_identical_across_stages():
    """The central compute claim: ZeRO changes bytes, not arithmetic."""
    a = art("exp7_throughput.json")
    assert a["flops_identical_across_stages"] is True
    assert a["flops_per_gpu_per_step"] == pytest.approx(1.741e15, rel=0.002)


def test_only_zero3_fits_27b_on_8_h100():
    f = art("exp7_throughput.json")["feasibility_H100_N8"]
    assert f["0"]["fits_80gb"] is False
    assert f["1"]["fits_80gb"] is False
    assert f["2"]["fits_80gb"] is False
    assert f["3"]["fits_80gb"] is True


def test_throughput_node_boundary_effect():
    s = art("exp7_throughput.json")["scenarios"]
    one = s["H100_N8_overlap80_stage3"]["tokens_per_second_per_gpu"]
    many = s["H100_N64_overlap80_stage3"]["tokens_per_second_per_gpu"]
    assert one == pytest.approx(1802, abs=1)
    assert many == pytest.approx(1251, abs=1)
    base8 = s["H100_N8_overlap80_stage2"]["tokens_per_second_per_gpu"]
    base64 = s["H100_N64_overlap80_stage2"]["tokens_per_second_per_gpu"]
    assert base8 == pytest.approx(1864, abs=1)
    assert base64 == pytest.approx(1692, abs=1)
    # ZeRO-3 costs ~3% inside a node, ~26% across 8 nodes
    assert (1 - one / base8) == pytest.approx(0.033, abs=0.005)
    assert (1 - many / base64) == pytest.approx(0.26, abs=0.01)


def test_stages_012_have_identical_throughput():
    s = art("exp7_throughput.json")["scenarios"]
    v = {st: s[f"H100_N64_overlap80_stage{st}"]["tokens_per_second_per_gpu"]
         for st in ("0", "1", "2")}
    assert len(set(v.values())) == 1


def test_frontier_7b_row():
    r = art("exp8_throughput_frontier.json")["rows"]["7"]
    assert r["0"]["fits_80gb"] is False
    assert r["0"]["model_state_gb"] == pytest.approx(112.0, abs=0.1)
    assert r["1"]["tokens_per_second_per_gpu"] == pytest.approx(8080, abs=2)
    assert r["2"]["tokens_per_second_per_gpu"] == pytest.approx(8080, abs=2)
    assert r["3"]["tokens_per_second_per_gpu"] == pytest.approx(7515, abs=2)
    assert r["2"]["tokens_per_second_per_gpu"] > r["3"]["tokens_per_second_per_gpu"]


def test_flops_estimate_discrepancy_recorded():
    a = art("exp9_measured_flops.json")
    assert a["ratio_measured_over_estimate"] == pytest.approx(0.94, abs=0.01)


def test_readme_comm_fractions_match_artifact():
    s = art("exp7_throughput.json")["scenarios"]
    assert s["H100_N64_overlap80_stage3"]["comm_fraction_of_step"] == pytest.approx(0.3355, abs=1e-4)
    assert s["H100_N8_overlap80_stage3"]["comm_fraction_of_step"] == pytest.approx(0.0429, abs=1e-4)


def test_no_mfu_overclaim():
    """compute_fraction_of_step must not be labelled MFU anywhere."""
    a = art("exp7_throughput.json")
    assert "not_mfu_warning" in a
    for v in a["scenarios"].values():
        assert "mfu" not in v


# ------------------------------------------------------------------- figures
def test_notebook_has_figures():
    """The assignment says SHOW how memory and computation change."""
    import nbformat
    nb = nbformat.read(os.path.join(ROOT, "notebook.ipynb"), as_version=4)
    figs = sum(1 for c in nb.cells for o in c.get("outputs", [])
               if o.get("output_type") == "display_data"
               and "image/png" in o.get("data", {}))
    assert figs >= 4, f"expected at least 4 rendered figures, found {figs}"


def test_notebook_outputs_are_committed():
    import nbformat
    nb = nbformat.read(os.path.join(ROOT, "notebook.ipynb"), as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    empty = [i for i, c in enumerate(code) if not c.get("outputs")]
    assert len(empty) <= 1, f"code cells with no committed output: {empty}"


def test_plots_module_importable_and_runs():
    import matplotlib
    matplotlib.use("Agg")
    from zerosim import plots
    for fn in (plots.fig_memory_breakdown, plots.fig_memory_ladder,
               plots.fig_throughput_frontier):
        ax = fn()
        assert ax is not None
    assert plots.fig_activation_floor(art("exp10_activation_floor.json")["toy"]) is not None


def test_fsdp_validation_harness_exists_and_states_tolerance():
    p = os.path.join(ROOT, "validate_against_fsdp.py")
    assert os.path.exists(p)
    src = open(p).read()
    assert "15%" in src, "tolerance must be stated in the script"
    assert "IT IS NOT RUN IN THIS REPO" in src, "must be explicit it was not run"
    # Uses FSDP2 fully_shard -- the API actually verified on CPU in 9b --
    # rather than FSDP1, which is not exercised anywhere in this repo.
    assert "fully_shard" in src and "reshard_after_forward" in src
    assert "FullyShardedDataParallel" not in src, \
        "should not use the untested FSDP1 path"
    assert "2 GPUs" in src, "must state the 2-GPU requirement"


def test_artifacts_are_reproducible():
    """Re-running must reproduce every artifact bit-for-bit except timings.

    Determinism is what makes the committed artifacts checkable by someone
    else, so it is worth asserting rather than assuming. wall_seconds is the
    only field allowed to vary.
    """
    import copy
    import subprocess
    import sys

    before = {n: art(n) for n in sorted(os.listdir(ART)) if n.endswith(".json")}
    subprocess.run([sys.executable, "run_experiments.py"], cwd=ROOT,
                   check=True, capture_output=True)
    after = {n: art(n) for n in before}

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "wall_seconds"}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o

    for n in before:
        assert strip(before[n]) == strip(after[n]), f"{n} is not reproducible"


def test_requirements_covers_every_third_party_import():
    """A fresh clone must be able to follow the README without failing.

    This exists because it did not: matplotlib and nbformat were both in use
    and absent from requirements.txt, so `pip install -r requirements.txt`
    followed by `pytest` would have crashed on someone else's machine.
    """
    import pathlib
    import re
    import sys

    stdlib = set(sys.stdlib_module_names)
    declared = set()
    for req in ("requirements.txt", "requirements-validation.txt"):
        for line in open(os.path.join(ROOT, req)):
            line = line.split("#")[0].strip()
            if line:
                declared.add(re.split(r"[<>=!\[]", line)[0].strip().lower())

    # Parse with ast, not a regex. A regex over source lines matches prose
    # inside docstrings -- "from README 9b: ..." in validate_against_fsdp.py
    # was read as `import readme` and failed this test spuriously.
    import ast

    used = set()
    for f in pathlib.Path(ROOT).rglob("*.py"):
        if ".git" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    used.add(n.name.split(".")[0].lower())
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    used.add(node.module.split(".")[0].lower())

    third_party = {m for m in used
                   if m not in stdlib and m != "zerosim" and not m.startswith("_")}
    missing = third_party - declared
    assert not missing, f"used but not in requirements.txt: {sorted(missing)}"


# ------------------------------------------------------- activation floor
def test_peak_reduction_is_reported_not_just_model_states():
    """Guards against cherry-picking the 32x model-state figure.

    The README must quote the measured PEAK reduction (5.92x) alongside the
    model-state reduction (32x), because ZeRO does not shard activations and
    the two numbers genuinely differ.
    """
    a = art("exp10_activation_floor.json")["toy"]
    assert a["3"]["model_state_reduction"] == pytest.approx(32.0, abs=0.01)
    assert a["3"]["peak_reduction"] == pytest.approx(5.92, abs=0.01)
    assert a["3"]["activation_share_of_peak"] == pytest.approx(0.82, abs=0.01)


def test_activations_identical_across_stages():
    """ZeRO shards model states, never activations."""
    a = art("exp10_activation_floor.json")["toy"]
    acts = {v["activation_bytes"] for v in a.values()}
    assert len(acts) == 1, f"activations differ across stages: {acts}"
    assert acts.pop() == 69632


def test_nothing_fits_at_realistic_scale_with_activations():
    """The correction to my own section 4 claim: counting activations, no
    stage fits an 80 GB card without checkpointing."""
    r = art("exp10_activation_floor.json")["realistic"]
    for s in "0123":
        assert r[s]["total_gb"] > 80, f"stage {s} unexpectedly fits"
    assert r["3"]["total_gb"] == pytest.approx(311.4, abs=0.1)
    assert r["3"]["activation_gb_no_checkpointing"] == pytest.approx(255.6, abs=0.1)


def test_readme_does_not_contradict_itself_on_fitting():
    """Section 4 says 'only ZeRO-3 fits'. That must be scoped to model states
    and cross-referenced, or the two sections contradict."""
    with open(README) as f:
        s = f.read()
    assert "only stage whose model\nstates fit" in s or "model states only" in s
    assert "3b" in s, "the activation-floor section must be cross-referenced"


# --------------------------------------------- real-implementation validation
def test_fsdp_cpu_validation_was_actually_run():
    """exp11 must exist with real measurements (unlike the GPU harness)."""
    a = art("exp11_fsdp_cpu_validation.json")
    assert a["all_within_tolerance"] is True
    assert a["psi"] == 858880 and a["world"] == 4
    assert a["tolerance_pct"] == 2.0


def test_zero1_ledger_matches_pytorch():
    a = art("exp11_fsdp_cpu_validation.json")["stages"]
    psi = 858880
    # DP: full params, full grads, 2*Psi Adam moments, all exact
    assert a["0"]["measured"]["param_elems"] == psi
    assert a["0"]["measured"]["grad_elems"] == psi
    assert a["0"]["measured"]["optim_moment_elems"] == 2 * psi
    assert max(a["0"]["relative_error"].values()) == 0.0
    # ZeRO-1: params and grads still full, moments sharded 4 ways
    assert a["1"]["measured"]["param_elems"] == psi
    assert a["1"]["measured"]["grad_elems"] == psi
    assert a["1"]["measured"]["optim_moment_elems"] == 429568
    assert a["1"]["relative_error"]["moments"] == pytest.approx(0.0003, abs=1e-4)


def test_pytorch_confirms_stage_invariance():
    """Loss must be bit-identical between DDP and DDP+ZeRO-1."""
    a = art("exp11_fsdp_cpu_validation.json")["loss_agreement_vs_no_shard"]
    for s, d in a.items():
        assert d == 0.0, f"stage {s} diverged from DDP by {d}"


def test_readme_does_not_claim_pytorch_lacks_zero1():
    """A corrected claim must stay corrected."""
    with open(README) as f:
        s = f.read()
    assert "ZeroRedundancyOptimizer" in s
    # If the false claim appears at all, it must appear only as a correction.
    if "no ZeRO-1 equivalent" in s:
        assert "**Wrong**" in s or "That was wrong" in s


def test_gpu_harness_still_honestly_marked_unrun():
    assert not os.path.exists(os.path.join(ART, "exp10_fsdp_validation.json")), \
        "GPU harness output must not appear unless actually run"
    src = open(os.path.join(ROOT, "validate_against_fsdp.py")).read()
    assert "IT IS NOT RUN IN THIS REPO" in src


def test_notebook_and_readme_cover_the_same_sections():
    """Two of my notebook patches silently no-op'd (a string anchor that did
    not match), shipping a notebook missing a section the README had. This
    asserts the two stay in step."""
    import nbformat
    nb = nbformat.read(os.path.join(ROOT, "notebook.ipynb"), as_version=4)
    nb_text = "\n".join(
        c.source if isinstance(c.source, str) else "".join(c.source)
        for c in nb.cells if c.cell_type == "markdown")
    with open(README) as f:
        rd = f.read()
    required = [
        ("activation floor", "activations", "activations"),
        ("real-implementation validation", "ZeroRedundancyOptimizer",
         "ZeroRedundancyOptimizer"),
        ("the three mistakes", "got wrong", "got wrong"),
        ("bucket trade-off", "bucket", "bucket"),
        ("limitations", "Limitations", "Limitations"),
    ]
    missing = [name for name, nbk, rdk in required
               if nbk.lower() not in nb_text.lower() or rdk.lower() not in rd.lower()]
    assert not missing, f"topic present in one document but not the other: {missing}"


def test_notebook_builder_anchors_still_resolve():
    """The build script must not contain a replace whose anchor is absent --
    that is how the silent no-ops happened."""
    src = open(os.path.join(ROOT, "tools", "build_notebook.py")).read()
    for marker in ("## 8b", "### 3b", "## 9 "):
        assert marker in src, f"builder lost section {marker!r}"


# ------------------------------------- all-stage validation against PyTorch
def test_all_four_stages_validated_against_pytorch():
    a = art("exp12_pytorch_all_stages.json")
    assert a["world"] == 4 and a["psi"] == 660736
    assert a["all_memory_within_tolerance"] is True
    assert set(a["stages"]) == {"0", "1", "2", "3"}


def test_measured_comm_matches_ring_formulas():
    """The headline external check: 2Psi vs 3Psi, measured on real FSDP2."""
    a = art("exp12_pytorch_all_stages.json")
    n = a["world"]
    ring = (n - 1) / n
    st = a["stages"]
    for s in ("0", "1", "2"):
        assert st[s]["comm_in_psi_units"] == pytest.approx(2 * ring, abs=0.02), \
            f"stage {s} should move 2*Psi*(N-1)/N"
    assert st["3"]["comm_in_psi_units"] == pytest.approx(3 * ring, abs=0.02)


def test_measured_zero3_ratio_is_exactly_1_5():
    r = art("exp12_pytorch_all_stages.json")["zero3_over_zero2_comm_ratio"]
    assert r["ratio"] == pytest.approx(1.5, abs=0.01)
    assert r["within_tolerance"] is True
    # tolerance must not have been loosened to make this pass
    assert art("exp12_pytorch_all_stages.json")["comm_ratio_tolerance"] == 0.25


def test_measured_zero2_equals_dp_on_the_wire():
    r = art("exp12_pytorch_all_stages.json")["zero2_over_dp_comm_ratio"]
    assert r["ratio"] == pytest.approx(1.0, abs=0.01)


def test_zero3_doubles_the_allgather_count():
    """Direct evidence FSDP2 re-gathers weights in the backward pass."""
    st = art("exp12_pytorch_all_stages.json")["stages"]
    def ag(s):
        return sum(v for k, v in st[s]["comm_calls"].items()
                   if "allgather" in k.lower() and not k.startswith("(local)"))
    assert ag("3") == 2 * ag("2"), f"ZeRO-2 {ag('2')} vs ZeRO-3 {ag('3')}"


def test_stage2_param_row_marked_not_comparable():
    """FSDP2 is not DeepSpeed ZeRO-2 on parameter memory; that must be flagged
    rather than passed off or silently tolerated."""
    st = art("exp12_pytorch_all_stages.json")["stages"]
    assert st["2"]["rows_not_comparable"] == ["params"]
    assert st["2"]["not_comparable_reason"]
    n = art("exp12_pytorch_all_stages.json")["world"]
    # the discrepancy must be exactly 1 - 1/N, the signature of the difference
    assert st["2"]["relative_error"]["params"] == pytest.approx(1 - 1 / n, abs=0.01)
    for s in ("0", "1", "3"):
        assert st[s]["rows_not_comparable"] == []


# ------------------------------------------- measured (not modelled) memory
def test_measured_memory_was_actually_run():
    a = art("exp13_measured_memory.json")
    assert a["psi"] == 858880 and a["world"] == 2
    assert a["all_peak_ratios_within_tolerance"] is True
    assert a["peak_ratio_tolerance_pct"] == 25.0


def test_measured_peak_ratios_track_the_ledger():
    st = art("exp13_measured_memory.json")["stages"]
    assert st["0"]["measured_peak_step_bytes"] == 31183572
    assert st["2"]["measured_peak_reduction_vs_dp"] == pytest.approx(1.80, abs=0.01)
    assert st["3"]["measured_peak_reduction_vs_dp"] == pytest.approx(1.81, abs=0.01)
    for s in st.values():
        assert s["relative_error_on_reduction"] <= 0.25


def test_measured_peak_reduction_is_below_model_state_reduction():
    """The activation floor, showing up in a real allocator rather than in my
    own accounting."""
    st = art("exp13_measured_memory.json")["stages"]
    for s in ("1", "3"):
        assert (st[s]["measured_peak_reduction_vs_dp"]
                < st[s]["simulator_model_state_reduction_vs_dp"])


def test_activations_measured_identical_across_ddp_stages():
    """Section 3b's claim that ZeRO does not shard activations, measured.
    Stages 0/1 are the clean read: both DDP, no gathered params retained."""
    a = art("exp13_measured_memory.json")["activation_rule_of_thumb"]
    assert a["identical_across_ddp_stages"] is True
    assert a["measured_bytes_ddp"] == 3884032


def test_fsdp_retained_bytes_reflect_resharding():
    """ZeRO-2 retains gathered params across fwd/bwd; ZeRO-3 frees them."""
    r = art("exp13_measured_memory.json")["activation_rule_of_thumb"]["per_stage_retained_bytes"]
    assert r["2"] > r["0"], "reshard=False should retain MORE than DP"
    assert r["3"] < r["0"], "reshard=True should retain LESS than DP"


def test_activation_rule_of_thumb_bias_is_recorded():
    """The 34*B*T*d rule overestimates; that must be stated, not buried."""
    a = art("exp13_measured_memory.json")["activation_rule_of_thumb"]
    assert a["measured_over_estimate"] == pytest.approx(0.436, abs=0.01)
    with open(README) as f:
        s = f.read()
    assert "2.3" in s and "overestimate" in s.lower()


def test_readme_3b_carries_the_corrected_activation_figures():
    """The corrected numbers must appear, and the conclusion must still hold."""
    with open(README) as f:
        s = f.read()
    for n in ("111.4", "167.2", "557.8"):
        assert n in s, f"corrected figure {n} missing from README"


def test_gpu_validation_instructions_exist_and_are_honest():
    p = os.path.join(ROOT, "GPU_VALIDATION.md")
    assert os.path.exists(p)
    s = open(p).read()
    assert "2 GPUs" in s or "2 GPUs, not 1" in s
    assert "Do not widen the tolerance" in s
    assert "15%" in s
