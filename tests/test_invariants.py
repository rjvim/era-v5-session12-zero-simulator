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
    assert "FULL_SHARD" in src and "SHARD_GRAD_OP" in src


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

    used = set()
    pat = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z_0-9]*)")
    for f in pathlib.Path(ROOT).rglob("*.py"):
        if ".git" in f.parts:
            continue
        for line in f.read_text().splitlines():
            m = pat.match(line)
            if m:
                used.add(m.group(1).lower())

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
