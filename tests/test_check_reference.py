"""RERUN-03 (D-07): scripts/check_reference.py proven on fabricated results dicts, no real run needed.

The reference values typed in test_reference_data_is_pinned are the handoff section 4 comparison
targets (verified against the rendered PDF, D-06). They are not reported results.
"""
import copy
import hashlib
import json
import subprocess
import sys

import pytest

import check_reference as C
from conftest import ROOT

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _model(rmse, nasa, capped=None, sd=0.05):
    m = {"rmse": {"mean": rmse, "std": sd}, "nasa": {"mean": nasa, "std": 1.0}}
    if capped is not None:
        m["rmse_capped_truth"] = {"mean": capped, "std": sd}
    return m


def ok_results():
    """A dict built from the module's own reference data: every check passes, no WARN."""
    res = {}
    for s in C.SUBSETS:
        counts = C.COUNTS[s]
        models = {}
        for name in (C.XGB, C.RF):
            ref = C.REF_TUNED[s][name]
            models[name] = _model(ref["rmse"], ref["nasa"], ref["rmse_capped_truth"])
        if s == "FD001":
            models[C.XGB_DEFAULT] = _model(C.REF_CANARY["rmse"], C.REF_CANARY["nasa"])
            for name in (C.XGB_RAW, C.RF_RAW):
                models[name] = _model(C.REF_ABLATION[name]["rmse"], C.REF_ABLATION[name]["nasa"])
        res[s] = {
            "train_engines": counts["train_engines"],
            "train_rows": counts["train_rows"],
            "test_engines": counts["test_engines"],
            "kept_sensors": [f"s{i}" for i in range(counts["kept_sensors"])],
            "n_features": counts["n_features"],
            "xgb_cv_rmse": C.REF_CV_RMSE[s]["xgb"],
            "rf_cv_rmse": C.REF_CV_RMSE[s]["rf"],
            "xgb_n_estimators": C.REF_FD001_XGB_N_ESTIMATORS,
            "models": models,
            "top_gain_features": {"cycle": 9.0, "s4_mean": 8.0, "s15_mean": 7.0},
        }
    return copy.deepcopy(res)


def run(tmp_path, capsys, results, *args):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    code = C.main([str(path), *args])
    return code, capsys.readouterr().out.splitlines()


def set_metric(res, subset, model, metric, value, stat="mean"):
    res[subset]["models"][model][metric][stat] = value


def fail_lines(lines):
    return [ln for ln in lines if ln.startswith("FAIL ")]


# ---------------------------------------------------------------------------------------------
# inside every band
# ---------------------------------------------------------------------------------------------
def test_in_band_results_pass_all_forty_checks(tmp_path, capsys):
    code, lines = run(tmp_path, capsys, ok_results())
    assert code == 0
    assert lines[-1] == "RESULT: PASS (40 checks, 0 warnings)"
    assert not fail_lines(lines)
    assert not [ln for ln in lines if ln.startswith("WARN ")]


# ---------------------------------------------------------------------------------------------
# one value outside each FAIL category
# ---------------------------------------------------------------------------------------------
def _drop_sensor(res):
    res["FD001"]["kept_sensors"] = res["FD001"]["kept_sensors"][:13]


FAIL_CASES = [
    # id, subset, mutator, check name, value, reference text, band text
    ("canary_rmse", "FD001",
     lambda r: set_metric(r, "FD001", C.XGB_DEFAULT, "rmse", 20.09),
     f"{C.XGB_DEFAULT} rmse", "20.0900", "20.08", "exact (2 dp)"),
    ("canary_nasa", "FD001",
     lambda r: set_metric(r, "FD001", C.XGB_DEFAULT, "nasa", 790),
     f"{C.XGB_DEFAULT} nasa", "790", "786", "exact (0 dp)"),
    ("count_train_engines", "FD002",
     lambda r: r["FD002"].__setitem__("train_engines", 261),
     "train_engines", "261", "260", "exact"),
    ("count_train_rows", "FD003",
     lambda r: r["FD003"].__setitem__("train_rows", 24721),
     "train_rows", "24721", "24720", "exact"),
    ("count_test_engines", "FD004",
     lambda r: r["FD004"].__setitem__("test_engines", 247),
     "test_engines", "247", "248", "exact"),
    ("count_kept_sensors", "FD001", _drop_sensor, "kept_sensors", "13", "14", "exact"),
    ("count_n_features", "FD002",
     lambda r: r["FD002"].__setitem__("n_features", 70),
     "n_features", "70", "69", "exact"),
    ("tuned_rmse_high", "FD003",
     lambda r: set_metric(r, "FD003", C.XGB, "rmse", 17.36 + 1.01),
     f"{C.XGB} rmse", f"{17.36 + 1.01:.4f}", "17.36", "+-1.0 cycle"),
    ("tuned_rmse_low", "FD004",
     lambda r: set_metric(r, "FD004", C.RF, "rmse", 28.49 - 1.01),
     f"{C.RF} rmse", f"{28.49 - 1.01:.4f}", "28.49", "+-1.0 cycle"),
    ("ablation_rmse_xgb_raw", "FD001",
     lambda r: set_metric(r, "FD001", C.XGB_RAW, "rmse", 17.78 + 1.5),
     f"{C.XGB_RAW} rmse", f"{17.78 + 1.5:.4f}", "17.78", "+-1.0 cycle"),
    ("ablation_rmse_rf_raw", "FD001",
     lambda r: set_metric(r, "FD001", C.RF_RAW, "rmse", 17.88 - 1.5),
     f"{C.RF_RAW} rmse", f"{17.88 - 1.5:.4f}", "17.88", "+-1.0 cycle"),
    ("tuned_nasa_high", "FD002",
     lambda r: set_metric(r, "FD002", C.XGB, "nasa", 8618 * 1.26),
     f"{C.XGB} nasa", f"{8618 * 1.26:.4f}", "8618", "+-25%"),
    ("tuned_nasa_low", "FD001",
     lambda r: set_metric(r, "FD001", C.RF, "nasa", 590 * 0.74),
     f"{C.RF} nasa", f"{590 * 0.74:.4f}", "590", "+-25%"),
]


@pytest.mark.parametrize("subset, mutate, check, value, reference, band",
                         [c[1:] for c in FAIL_CASES], ids=[c[0] for c in FAIL_CASES])
def test_value_outside_band_fails_with_named_message(tmp_path, capsys, subset, mutate, check, value, reference, band):
    res = ok_results()
    mutate(res)
    code, lines = run(tmp_path, capsys, res)
    assert code == 1
    bad = fail_lines(lines)
    assert len(bad) == 1, bad
    line = bad[0]
    assert line.startswith(f"FAIL {subset} ")
    assert f" {check}: " in line
    assert f"value {value} " in line
    assert f"reference {reference} " in line
    assert line.endswith(f"band {band}")
    assert lines[-1] == "RESULT: FAIL (1 of 40 checks)"


# ---------------------------------------------------------------------------------------------
# band edges are inclusive
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("subset, model, metric, factor", [
    ("FD003", C.XGB, "rmse", "+1"),
    ("FD003", C.XGB, "rmse", "-1"),
    ("FD002", C.XGB, "nasa", "x1.25"),
    ("FD001", C.RF, "nasa", "x0.75"),
], ids=["rmse_upper_edge", "rmse_lower_edge", "nasa_upper_edge", "nasa_lower_edge"])
def test_value_exactly_on_band_edge_passes(tmp_path, capsys, subset, model, metric, factor):
    res = ok_results()
    ref = C.REF_TUNED[subset][model][metric]
    value = {"+1": ref + 1.0, "-1": ref - 1.0, "x1.25": ref * 1.25, "x0.75": ref * 0.75}[factor]
    set_metric(res, subset, model, metric, value)
    code, lines = run(tmp_path, capsys, res)
    assert code == 0
    assert not fail_lines(lines)
    assert lines[-1].startswith("RESULT: PASS (40 checks, ")


# ---------------------------------------------------------------------------------------------
# missing is a FAIL, never a skip
# ---------------------------------------------------------------------------------------------
def test_missing_subset_is_a_fail(tmp_path, capsys):
    res = ok_results()
    del res["FD003"]
    code, lines = run(tmp_path, capsys, res)
    assert code == 1
    assert "FAIL FD003 subset: value missing reference present band exact" in lines
    assert lines[-1].startswith("RESULT: FAIL (1 of ")


def test_missing_model_is_a_fail(tmp_path, capsys):
    res = ok_results()
    del res["FD002"]["models"][C.RF]
    code, lines = run(tmp_path, capsys, res)
    assert code == 1
    bad = fail_lines(lines)
    assert bad and all(" value missing " in ln and ln.startswith("FAIL FD002 ") for ln in bad)
    assert any(f"models.{C.RF}.rmse.mean" in ln for ln in bad)


def test_missing_metric_key_is_a_fail(tmp_path, capsys):
    res = ok_results()
    del res["FD004"]["models"][C.XGB]["rmse"]
    code, lines = run(tmp_path, capsys, res)
    assert code == 1
    bad = fail_lines(lines)
    assert len(bad) == 1
    assert bad[0].startswith(f"FAIL FD004 models.{C.XGB}.rmse.mean: value missing ")


# ---------------------------------------------------------------------------------------------
# WARN never changes the exit code
# ---------------------------------------------------------------------------------------------
def _flip_ordering(r):
    set_metric(r, "FD002", C.XGB, "rmse", 27.30)


def _rf_sd(r):
    set_metric(r, "FD001", C.RF, "rmse", 0.31, stat="std")


def _no_leader(r):
    r["FD003"]["top_gain_features"] = {"s7_slope": 3.0, "s9_std": 2.0, "op1": 1.0}


def _capped_off(r):
    capped = C.REF_TUNED["FD004"][C.XGB]["rmse_capped_truth"]
    set_metric(r, "FD004", C.XGB, "rmse_capped_truth", capped + 1.5)


def _raw_gap(r):
    set_metric(r, "FD001", C.XGB, "rmse", 18.70)
    set_metric(r, "FD001", C.XGB_RAW, "rmse", 17.60)


WARN_CASES = [
    ("ordering_flip", _flip_ordering, "WARN FD002 ordering: "),
    ("rf_rmse_sd", _rf_sd, f"WARN FD001 {C.RF} rmse std: "),
    ("no_gain_leader_in_top3", _no_leader, "WARN FD003 top-3 gain: "),
    ("capped_truth_off", _capped_off, f"WARN FD004 {C.XGB} rmse_capped_truth: "),
    ("raw_sensor_gap", _raw_gap, "WARN FD001 raw-sensor gap: "),
]


@pytest.mark.parametrize("mutate, prefix", [c[1:] for c in WARN_CASES], ids=[c[0] for c in WARN_CASES])
def test_warn_only_cases_keep_exit_zero(tmp_path, capsys, mutate, prefix):
    res = ok_results()
    mutate(res)
    code, lines = run(tmp_path, capsys, res)
    assert code == 0
    assert any(ln.startswith(prefix) for ln in lines), lines
    assert not fail_lines(lines)
    assert lines[-1].startswith("RESULT: PASS (40 checks, ")
    assert not lines[-1].endswith(", 0 warnings)")


def test_equal_xgb_and_rf_rmse_counts_as_an_ordering_warn(tmp_path, capsys):
    res = ok_results()
    set_metric(res, "FD003", C.XGB, "rmse", 18.0)
    set_metric(res, "FD003", C.RF, "rmse", 18.0)
    code, lines = run(tmp_path, capsys, res)
    assert code == 0
    assert any(ln.startswith("WARN FD003 ordering: ") for ln in lines)


def test_ordering_in_the_expected_direction_is_info_only(tmp_path, capsys):
    code, lines = run(tmp_path, capsys, ok_results())
    assert code == 0
    assert [ln for ln in lines if " ordering: " in ln] == [
        f"INFO {s} ordering: value XGB {C.REF_TUNED[s][C.XGB]['rmse']:.4f} RF {C.REF_TUNED[s][C.RF]['rmse']:.4f} "
        f"reference XGB < RF band {C.BAND_ORDERING}" for s in C.SUBSETS]


# ---------------------------------------------------------------------------------------------
# unreadable input is exit 2
# ---------------------------------------------------------------------------------------------
def test_nonexistent_results_file_is_exit_2(tmp_path, capsys):
    assert C.main([str(tmp_path / "no_such_results.json")]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_invalid_json_is_exit_2(tmp_path, capsys):
    path = tmp_path / "results.json"
    path.write_text("{not json", encoding="utf-8")
    assert C.main([str(path)]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_json_list_at_top_level_is_exit_2(tmp_path, capsys):
    path = tmp_path / "results.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert C.main([str(path)]) == 2
    assert "not a JSON object" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# --subsets
# ---------------------------------------------------------------------------------------------
def test_subsets_flag_with_only_fd001_present(tmp_path, capsys):
    res = {"FD001": ok_results()["FD001"]}
    code, lines = run(tmp_path, capsys, res, "--subsets", "FD001")
    assert code == 0
    assert lines[-1].startswith("RESULT: PASS (13 checks, ")


def test_subsets_flag_ignores_other_subsets(tmp_path, capsys):
    res = ok_results()
    res["FD002"]["n_features"] = 70  # would FAIL if judged
    code, lines = run(tmp_path, capsys, res, "--subsets", "FD001", "FD003")
    assert code == 0
    assert not any(" FD002 " in ln for ln in lines)
    assert lines[-1].startswith("RESULT: PASS (22 checks, ")


# ---------------------------------------------------------------------------------------------
# line formats
# ---------------------------------------------------------------------------------------------
def test_format_row_precision_rules():
    assert C.format_row(C.Row("FAIL", "FD002", "train_rows", 1, 53759, "exact")) == \
        "FAIL FD002 train_rows: value 1 reference 53759 band exact"
    assert C.format_row(C.Row("PASS", "FD001", "x rmse", 18.04223, 17.77, "+-1.0 cycle")) == \
        "PASS FD001 x rmse: value 18.0422 reference 17.77 band +-1.0 cycle"
    assert C.format_row(C.Row("FAIL", "FD003", "subset", "missing", "present", "exact")) == \
        "FAIL FD003 subset: value missing reference present band exact"


def test_summarise_counts_only_pass_and_fail_rows():
    rows = [C.Row("PASS", "FD001", "a", 1, 1, "exact"), C.Row("WARN", "FD001", "b", 1.0, 1.0, "x"),
            C.Row("INFO", "FD001", "c", 1.0, 1.0, "x")]
    assert C.summarise(rows) == "RESULT: PASS (1 checks, 1 warnings)"
    rows.append(C.Row("FAIL", "FD001", "d", 2, 1, "exact"))
    assert C.summarise(rows) == "RESULT: FAIL (1 of 2 checks)"


# ---------------------------------------------------------------------------------------------
# the reference and band data are pinned (D-06): a literal copy typed here
# ---------------------------------------------------------------------------------------------
def test_reference_data_is_pinned():
    assert C.COUNTS == {
        "FD001": {"train_engines": 100, "train_rows": 20631, "test_engines": 100, "kept_sensors": 14, "n_features": 57},
        "FD002": {"train_engines": 260, "train_rows": 53759, "test_engines": 259, "kept_sensors": 17, "n_features": 69},
        "FD003": {"train_engines": 100, "train_rows": 24720, "test_engines": 100, "kept_sensors": 16, "n_features": 65},
        "FD004": {"train_engines": 249, "train_rows": 61249, "test_engines": 248, "kept_sensors": 17, "n_features": 69},
    }
    assert C.REF_TUNED == {
        "FD001": {"XGBoost (rolling features)": {"rmse": 17.77, "nasa": 582, "rmse_capped_truth": 16.60},
                  "Random Forest (rolling features)": {"rmse": 18.18, "nasa": 590, "rmse_capped_truth": 17.01}},
        "FD002": {"XGBoost (rolling features)": {"rmse": 26.85, "nasa": 8618, "rmse_capped_truth": 14.82},
                  "Random Forest (rolling features)": {"rmse": 27.19, "nasa": 8996, "rmse_capped_truth": 15.20}},
        "FD003": {"XGBoost (rolling features)": {"rmse": 17.36, "nasa": 617, "rmse_capped_truth": 15.96},
                  "Random Forest (rolling features)": {"rmse": 18.85, "nasa": 778, "rmse_capped_truth": 17.39}},
        "FD004": {"XGBoost (rolling features)": {"rmse": 28.36, "nasa": 5667, "rmse_capped_truth": 17.18},
                  "Random Forest (rolling features)": {"rmse": 28.49, "nasa": 5574, "rmse_capped_truth": 17.26}},
    }
    assert C.REF_CANARY == {"rmse": 20.08, "nasa": 786}
    assert C.REF_ABLATION == {"XGBoost (raw sensors only)": {"rmse": 17.78, "nasa": 566},
                              "Random Forest (raw sensors only)": {"rmse": 17.88, "nasa": 526}}
    assert C.REF_CV_RMSE == {"FD001": {"xgb": 16.25, "rf": 16.45}, "FD002": {"xgb": 16.17, "rf": 16.45},
                             "FD003": {"xgb": 14.69, "rf": 15.06}, "FD004": {"xgb": 16.11, "rf": 16.30}}
    assert C.REF_FD001_XGB_N_ESTIMATORS == 255
    assert C.SHAP_LEADERS == ("cycle", "s15_mean", "s4_mean", "s2_mean", "s3_mean", "s11_mean")
    assert C.BANDS == {"rmse_cycles": 1.0, "nasa_relative": 0.25, "capped_warn_cycles": 1.0,
                       "sd_warn_cycles": 0.3, "raw_gap_warn_cycles": 1.0}
    assert C.EPS == 1e-9
    assert C.SUBSETS == ("FD001", "FD002", "FD003", "FD004")
    assert (C.XGB, C.RF, C.XGB_DEFAULT, C.XGB_RAW, C.RF_RAW) == (
        "XGBoost (rolling features)", "Random Forest (rolling features)",
        "XGBoost default params (rolling features)", "XGBoost (raw sensors only)", "Random Forest (raw sensors only)")


# ---------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------
def test_markdown_new_file_has_title_markers_and_every_subset(tmp_path, capsys):
    md = tmp_path / "out" / "recon.md"
    code, _ = run(tmp_path, capsys, ok_results(), "--markdown", str(md))
    assert code == 0
    raw = md.read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert text.startswith(C.MD_TITLE + "\n\n" + C.MARK_BEGIN)
    assert C.MARK_END in text
    assert "RESULT: PASS" in text
    for s in C.SUBSETS:
        assert s in text
    assert "Bands (declared before the full run)" in text


def test_markdown_source_label_never_an_absolute_path_and_hash_ignores_cr(tmp_path, capsys):
    results_path = tmp_path / "results.json"
    body = json.dumps(ok_results(), indent=1).replace("\n", "\r\n")
    results_path.write_bytes(body.encode("utf-8"))
    md = tmp_path / "recon.md"
    assert C.main([str(results_path), "--markdown", str(md)]) == 0
    capsys.readouterr()
    text = md.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert "Source: `results.json`" in text
    assert hashlib.sha256(body.replace("\r", "").encode("utf-8")).hexdigest() in text


def test_markdown_existing_file_keeps_text_outside_markers(tmp_path, capsys):
    md = tmp_path / "recon.md"
    md.write_text(f"intro line\n\n{C.MARK_BEGIN}\nold generated text\n{C.MARK_END}\n\n## Findings\n\nhand written\n",
                  encoding="utf-8", newline="\n")
    code, _ = run(tmp_path, capsys, ok_results(), "--markdown", str(md))
    assert code == 0
    text = md.read_text(encoding="utf-8")
    assert text.startswith("intro line\n\n" + C.MARK_BEGIN)
    assert text.endswith(C.MARK_END + "\n\n## Findings\n\nhand written\n")
    assert "old generated text" not in text
    assert "RESULT: PASS" in text
    code, _ = run(tmp_path, capsys, ok_results(), "--markdown", str(md))  # idempotent
    assert md.read_text(encoding="utf-8") == text


def test_markdown_file_without_markers_is_exit_2_and_unchanged(tmp_path, capsys):
    md = tmp_path / "recon.md"
    md.write_text("hand written only\n", encoding="utf-8", newline="\n")
    code, _ = run(tmp_path, capsys, ok_results(), "--markdown", str(md))
    assert code == 2
    assert md.read_text(encoding="utf-8") == "hand written only\n"


def test_bands_only_markdown_needs_no_results_and_has_no_result_line(tmp_path, capsys):
    md = tmp_path / "recon.md"
    assert C.main(["--bands-only", "--markdown", str(md)]) == 0
    out = capsys.readouterr().out
    text = md.read_text(encoding="utf-8")
    assert "Bands (declared before the full run)" in out and "Bands (declared before the full run)" in text
    assert "Reference values (Claude_Code_Handoff.pdf section 4)" in text
    assert C.MARK_BEGIN in text and C.MARK_END in text
    assert "RESULT:" not in text and "RESULT:" not in out
    assert b"\r" not in md.read_bytes()


def test_bands_only_never_reads_the_results_file(tmp_path, capsys):
    assert C.main([str(tmp_path / "no_such_results.json"), "--bands-only"]) == 0
    capsys.readouterr()


# ---------------------------------------------------------------------------------------------
# real process
# ---------------------------------------------------------------------------------------------
def _run_process(path):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_reference.py"), str(path)],
                          capture_output=True, text=True, timeout=60)


def test_subprocess_exit_code_zero_on_in_band_results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(ok_results()), encoding="utf-8")
    proc = _run_process(path)
    assert proc.returncode == 0
    assert proc.stdout.splitlines()[-1] == "RESULT: PASS (40 checks, 0 warnings)"


def test_subprocess_exit_code_one_on_out_of_band_results(tmp_path):
    res = ok_results()
    set_metric(res, "FD001", C.XGB_DEFAULT, "rmse", 20.5)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(res), encoding="utf-8")
    proc = _run_process(path)
    assert proc.returncode == 1
    assert "FAIL FD001 " + C.XGB_DEFAULT + " rmse: value 20.5000 reference 20.08 band exact (2 dp)" in proc.stdout


def test_importing_the_checker_loads_no_kernel_or_numpy():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import check_reference; "
            "bad = {'rul_lib', 'run_experiments', 'numpy', 'xgboost'} & set(sys.modules); "
            "sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "scripts")],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
