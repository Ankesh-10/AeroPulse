"""D-10 / T-02-08 / T-02-09: the run wrapper's preflight checks, single-call launcher and manifest privacy.

Files are written only under tmp_path; the child process is a tiny `python -c`, never the kernel.
"""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

import run_baseline as RB
from conftest import ROOT

pytestmark = pytest.mark.unit

CHILD = [sys.executable, "-c", "print('hello from child'); raise SystemExit(3)"]


# ------------------------------------------------------------------------------------- subsets

@pytest.mark.parametrize("bad", [["FD005"], ["FD001;echo"], ["FD001", "FD001"], []])
def test_check_subsets_rejects_bad_input(bad):
    assert RB.check_subsets(bad)


def test_check_subsets_names_the_offender():
    assert any("FD005" in p for p in RB.check_subsets(["FD005"]))


def test_check_subsets_accepts_all_four():
    assert RB.check_subsets(["FD001", "FD002", "FD003", "FD004"]) == []


# ------------------------------------------------------------------------------------- outputs

def _tree(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "models").mkdir()
    return tmp_path


def test_check_outputs_accepts_empty_dirs_and_missing_dirs(tmp_path):
    assert RB.check_outputs(_tree(tmp_path), ["FD001"], resume=False) == []
    assert RB.check_outputs(tmp_path / "nowhere", ["FD001"], resume=False) == []


def test_check_outputs_names_stale_files(tmp_path):
    root = _tree(tmp_path)
    (root / "outputs" / "results.json").write_text("{}")
    (root / "models" / "xgb_FD001.json").write_text("{}")
    problems = " | ".join(RB.check_outputs(root, ["FD001"], resume=False))
    assert "outputs/results.json" in problems
    assert "models/xgb_FD001.json" in problems


def test_check_outputs_resume_needs_a_results_file(tmp_path):
    assert RB.check_outputs(_tree(tmp_path), ["FD002"], resume=True)


def test_check_outputs_resume_rejects_present_subset_and_accepts_new(tmp_path):
    root = _tree(tmp_path)
    (root / "outputs" / "results.json").write_text(json.dumps({"FD001": {}}))
    assert RB.check_outputs(root, ["FD001"], resume=True)
    assert RB.check_outputs(root, ["FD002", "FD003"], resume=True) == []


def test_check_outputs_resume_rejects_unparseable_results(tmp_path):
    root = _tree(tmp_path)
    (root / "outputs" / "results.json").write_text("not json")
    assert RB.check_outputs(root, ["FD002"], resume=True)
    (root / "outputs" / "results.json").write_text("[1, 2]")
    assert RB.check_outputs(root, ["FD002"], resume=True)


# ------------------------------------------------------------------------------------- command

def test_build_command_is_the_d10_command():
    assert RB.build_command(["FD001"]) == [sys.executable, "-u", "src/run_experiments.py", "FD001"]
    assert RB.build_command(["FD001", "FD003"])[-2:] == ["FD001", "FD003"]


def test_check_interpreter_accepts_the_repo_venv_only():
    # a tmp root has no .venv, so the running interpreter can never be its venv
    assert any(".venv" in p for p in RB.check_interpreter(Path(ROOT) / "no_such_dir"))


# ------------------------------------------------------------------------------------- launcher

def _run(tmp_path, append, meta=None):
    return RB.run_and_record(
        CHILD, tmp_path, tmp_path / "outputs" / "run_log.txt", tmp_path / "outputs" / "run_manifest.json",
        meta or {"git_sha": "abc", "git_dirty": [], "bands_commit": "def", "subsets": ["FD001"],
                 "resume": append, "keep_awake": False},
        append=append,
    )


def test_run_and_record_returns_child_code_and_records(tmp_path):
    assert _run(tmp_path, append=False) == 3
    log = (tmp_path / "outputs" / "run_log.txt").read_text()
    assert "hello from child" in log
    doc = json.loads((tmp_path / "outputs" / "run_manifest.json").read_text())
    assert len(doc["invocations"]) == 1
    rec = doc["invocations"][0]
    assert rec["exit_code"] == 3
    assert rec["command"][0] == "python" and rec["command"][1:3] == ["-c", CHILD[2]]
    assert rec["wall_seconds"] >= 0
    assert rec["started_at"].endswith("Z") and rec["ended_at"].endswith("Z")
    assert isinstance(rec["started_epoch"], int) and rec["ended_epoch"] >= rec["started_epoch"]
    assert rec["kernel_n_jobs"] == 2 and set(rec["packages"]) == set(RB.PACKAGES)
    assert rec["git_sha"] == "abc" and rec["bands_commit"] == "def"


def test_run_and_record_append_keeps_both_invocations_and_log_lines(tmp_path):
    _run(tmp_path, append=False)
    _run(tmp_path, append=True)
    doc = json.loads((tmp_path / "outputs" / "run_manifest.json").read_text())
    assert len(doc["invocations"]) == 2
    assert (tmp_path / "outputs" / "run_log.txt").read_text().count("hello from child") == 2


def test_fresh_run_truncates_the_log(tmp_path):
    _run(tmp_path, append=False)
    _run(tmp_path, append=False)
    assert (tmp_path / "outputs" / "run_log.txt").read_text().count("hello from child") == 1


def test_manifest_is_written_even_when_the_child_cannot_start(tmp_path):
    with pytest.raises(OSError):
        RB.run_and_record(
            [str(tmp_path / "no_such_python.exe"), "-c", "pass"], tmp_path, tmp_path / "outputs" / "l.txt",
            tmp_path / "outputs" / "m.json", {"subsets": []}, append=False)
    rec = json.loads((tmp_path / "outputs" / "m.json").read_text())["invocations"][0]
    assert rec["exit_code"] is None


def test_manifest_uses_lf_line_endings(tmp_path):
    _run(tmp_path, append=False)
    assert b"\r" not in (tmp_path / "outputs" / "run_manifest.json").read_bytes()


def test_interpreter_label_is_relative_inside_the_repo_and_a_name_outside(tmp_path):
    inside = tmp_path / ".venv" / "Scripts" / "python.exe"
    assert RB._interpreter_label(str(inside), tmp_path) == ".venv/Scripts/python.exe"
    assert RB._interpreter_label(str(Path(sys.executable)), tmp_path) == Path(sys.executable).name


# ------------------------------------------------------------------------------------- privacy

def _forbidden(tmp_path):
    items = {str(tmp_path), str(ROOT), os.path.expanduser("~"), socket.gethostname(), os.environ.get("USERNAME", ""),
             os.environ.get("USER", "")}
    out = set()
    for s in items:
        if len(s) >= 3:
            out |= {s, json.dumps(s)[1:-1], s.replace("\\", "/")}
    return out


def test_manifest_holds_no_path_user_or_host(tmp_path):
    _run(tmp_path, append=False)
    _run(tmp_path, append=True)
    text = (tmp_path / "outputs" / "run_manifest.json").read_text().lower()
    leaks = [s for s in _forbidden(tmp_path) if s.lower() in text]
    assert leaks == []


# ------------------------------------------------------------------------------------- main

def test_main_rejects_unknown_subset_with_code_2(capsys):
    assert RB.main(["--check-only", "FD009"]) == 2
    assert "PREFLIGHT FAIL" in capsys.readouterr().err
