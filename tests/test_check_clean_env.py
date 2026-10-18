"""D-15 / T-03-11..T-03-14: the clean-environment check's helpers, markdown privacy and writer.

Files are written only under tmp_path (or a scratch directory the test removes). The one slow test runs
the whole check on Python 3.13 and 3.14: it builds throwaway venvs and takes several minutes.
"""
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import check_clean_env as C
from conftest import ROOT

# ------------------------------------------------------------------------------------- versions

@pytest.mark.parametrize("bad", [[], ["3.11"], ["3.13;echo"], ["3.13", "3.13"], ["3.15"]])
def test_check_pythons_rejects_bad_input(bad):
    assert C.check_pythons(bad)


def test_check_pythons_accepts_both_default_versions():
    assert C.check_pythons(["3.13", "3.14"]) == []
    assert C.check_pythons(list(C.DEFAULT_PYTHONS)) == []


def test_main_rejects_unsupported_python_before_any_subprocess(capsys):
    assert C.main(["--pythons", "3.11"]) == 2
    assert "PREFLIGHT FAIL" in capsys.readouterr().err


def test_pytest_pin_equals_requirements_dev_line():
    lines = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    expected = [ln.strip() for ln in lines if ln.strip().startswith("pytest==")]
    assert expected and C.pytest_pin(ROOT) == expected[0]


def test_pytest_pin_missing_raises(tmp_path):
    (tmp_path / "requirements-dev.txt").write_text("-r requirements.txt\noptuna==5.0.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        C.pytest_pin(tmp_path)


# ------------------------------------------------------------------------------------- commands

def test_venv_python_path_follows_os():
    p = C.venv_python(Path("x") / "v")
    tail = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    assert p.parts[-2:] == tail


def test_venv_and_install_commands():
    assert C.venv_command("uv", "3.13", Path("v")) == ["uv", "venv", "--python", "3.13", "v"]
    cmd = C.install_command("uv", Path("v/py"), ROOT, "pytest==9.1.1")
    assert all(isinstance(a, str) for a in cmd)
    assert cmd[:3] == ["uv", "pip", "install"]
    assert cmd[cmd.index("--only-binary") + 1] == ":all:"
    assert cmd[cmd.index("--python") + 1] == str(Path("v/py"))
    assert cmd[cmd.index("-r") + 1] == str(ROOT / "requirements.txt")
    assert cmd[-1] == "pytest==9.1.1"


def test_pytest_command_names_only_the_demo_files():
    cmd = C.pytest_command(Path("v/py"))
    assert all(isinstance(a, str) for a in cmd)
    for f in ("tests/test_demo_lib.py", "tests/test_demo_hygiene.py", "tests/test_demo_app.py"):
        assert f in cmd
    joined = " ".join(cmd)
    assert "test_smoke" not in joined and "test_artifacts" not in joined
    assert "no:cacheprovider" in cmd and "-rs" in cmd


def test_streamlit_command_is_headless_with_script_health_check():
    cmd = C.streamlit_command(Path("v/py"), 8765)
    assert all(isinstance(a, str) for a in cmd)
    assert cmd[1:6] == ["-m", "streamlit", "run", "app.py", "--server.headless"]
    assert cmd[cmd.index("--server.port") + 1] == "8765"
    assert cmd[cmd.index("--server.address") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--server.scriptHealthCheckEnabled") + 1] == "true"


def test_linux_resolve_command_resolves_without_installing():
    cmd = C.linux_resolve_command("uv", "3.14", ROOT, Path("out.txt"))
    assert all(isinstance(a, str) for a in cmd)
    assert "compile" in cmd and "install" not in cmd
    assert cmd[cmd.index("--python-platform") + 1] == "x86_64-manylinux_2_31"
    assert cmd[cmd.index("--python-version") + 1] == "3.14"
    assert cmd[cmd.index("--only-binary") + 1] == ":all:"
    assert cmd[cmd.index("-o") + 1] == "out.txt"


def test_free_port_is_bindable():
    port = C.free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


def test_child_env_forces_utf8_and_drops_venv_variables(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "somewhere")
    monkeypatch.setenv("PYTHONPATH", "elsewhere")
    monkeypatch.setenv("PYTHONHOME", "base-of-another-python")
    monkeypatch.setenv("UV_INTERNAL__PYTHONHOME", "base-of-another-python")
    env = C.child_env()
    assert env["PYTHONUTF8"] == "1" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    for name in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_INTERNAL__PYTHONHOME"):
        assert name not in env


# ------------------------------------------------------------------------------------- pins

def test_requirement_pins_are_exactly_the_seven_runtime_pins():
    assert C.requirement_pins(ROOT) == {
        "xgboost": "3.4.1", "scikit-learn": "1.9.1", "pandas": "3.0.5", "numpy": "2.5.3",
        "shap": "0.53.0", "streamlit": "1.65.0", "matplotlib": "3.11.2"}


def _compiled(pins, extra="scipy==1.18.1\n"):
    return "".join(f"{n}=={v}\n" for n, v in pins.items()) + extra


def test_linux_resolve_problems():
    required = C.requirement_pins(ROOT)
    assert C.linux_resolve_problems(required, C.resolved_pins(_compiled(required))) == []
    missing = {k: v for k, v in required.items() if k != "shap"}
    probs = C.linux_resolve_problems(required, C.resolved_pins(_compiled(missing)))
    assert len(probs) == 1 and "shap" in probs[0]
    other = dict(required, pandas="3.0.6")
    probs = C.linux_resolve_problems(required, C.resolved_pins(_compiled(other)))
    assert len(probs) == 1 and "pandas" in probs[0]


def test_resolved_pins_skips_comments_and_normalises_names():
    text = "# comment\n\nScikit_Learn==1.9.1\nnumpy==2.5.3 ; python_version >= '3.12'\n"
    assert C.resolved_pins(text) == {"scikit-learn": "1.9.1", "numpy": "2.5.3"}


# ------------------------------------------------------------------------------------- pytest summary

@pytest.mark.parametrize("text, counts, ok", [
    ("25 passed in 3.10s", {"passed": 25, "failed": 0, "errors": 0, "skipped": 0}, True),
    ("3 failed, 20 passed, 1 skipped in 9.0s", {"passed": 20, "failed": 3, "errors": 0, "skipped": 1}, False),
    ("1 error in 0.5s", {"passed": 0, "failed": 0, "errors": 1, "skipped": 0}, False),
    ("SKIPPED [1] tests/x.py:3: reason\n109 passed, 2 skipped in 121.90s (0:02:01)\n",
     {"passed": 109, "failed": 0, "errors": 0, "skipped": 2}, False),
    ("", {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}, False),
])
def test_parse_pytest_summary_and_tests_ok(text, counts, ok):
    parsed = C.parse_pytest_summary(text)
    assert parsed == counts
    assert C.tests_ok(parsed) is ok


# ------------------------------------------------------------------------------------- scratch

def test_make_scratch_is_outside_the_repo():
    scratch = C.make_scratch(ROOT)
    try:
        assert scratch.is_dir()
        assert ROOT.resolve() not in scratch.parents and scratch != ROOT.resolve()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_make_scratch_refuses_a_directory_inside_the_repo(monkeypatch, tmp_path):
    fake_root = tmp_path / "repo"
    inside = fake_root / "scratch"
    inside.mkdir(parents=True)
    monkeypatch.setattr(C.tempfile, "mkdtemp", lambda prefix="": str(inside))
    with pytest.raises(ValueError):
        C.make_scratch(fake_root)


# ------------------------------------------------------------------------------------- markdown

def _result(version, load, imp=5.0, linux_ok=True, ok=True):
    return {
        "request": version, "python": f"{version}.9", "install_ok": True,
        "packages": {n: "1.0.0" for n in C.PACKAGES},
        "linux": {"platform": C.LINUX_PLATFORM, "ok": linux_ok, "packages": 45, "problems": []},
        "tests": {"passed": 109, "failed": 0, "errors": 0, "skipped": 0}, "tests_ok": True,
        "probe_ok": True,
        "timings": {s: {"import_s": imp, "load_s": load + i} for i, s in enumerate(C.SUBSETS)},
        "ok": ok and linux_ok,
    }


def test_render_markdown_pass(monkeypatch):
    monkeypatch.setattr(C.platform, "system", lambda: "Windows")
    md = C.render_markdown([_result("3.13", 1.0), _result("3.14", 2.0)], "2026-10-18")
    assert "RESULT: PASS" in md
    assert "D-01: worst local cold load 5.00 s (threshold 30 s): precompute not triggered" in md
    rows = [ln for ln in md.splitlines() if ln.startswith("| 3.1") and C.LINUX_PLATFORM in ln]
    assert len(rows) == 2
    assert "local Windows measurement" in md
    assert "not the Streamlit Cloud cold start" in md
    assert "Measured on Windows" in md and "2026-10-18" in md


def test_render_markdown_threshold_exceeded_and_linux_failure():
    over = C.render_markdown([_result("3.13", 40.0)], "2026-10-18")
    assert "PRECOMPUTE THRESHOLD EXCEEDED" in over
    bad = C.render_markdown([_result("3.13", 1.0), _result("3.14", 1.0, linux_ok=False)], "2026-10-18")
    assert "RESULT: FAIL" in bad and "RESULT: PASS" not in bad


def test_render_markdown_holds_no_local_identity():
    md = C.render_markdown([_result("3.13", 1.0), _result("3.14", 2.0)], "2026-10-18")
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", md)
    assert str(Path.home()) not in md
    for ident in (os.environ.get("USERNAME", ""), os.environ.get("USER", ""), socket.gethostname()):
        if len(ident) >= 3:
            assert ident.lower() not in md.lower()


def test_write_markdown_creates_new_file(tmp_path):
    target = tmp_path / "docs" / "DEPLOY.md"
    C.write_markdown(target, "body one")
    text = target.read_bytes().decode("utf-8")
    assert text.startswith(C.MD_TITLE) and C.MARK_BEGIN in text and C.MARK_END in text and "body one" in text
    assert b"\r" not in target.read_bytes()


def test_write_markdown_replaces_only_the_block(tmp_path):
    target = tmp_path / "DEPLOY.md"
    before, after = "# Title\n\nhand written before\n\n", "\n\nhand written after\n"
    target.write_bytes((before + f"{C.MARK_BEGIN}\n\nold\n\n{C.MARK_END}" + after).encode("utf-8"))
    C.write_markdown(target, "new body")
    text = target.read_bytes().decode("utf-8")
    assert text.startswith(before) and text.endswith(after)
    assert "new body" in text and "old" not in text


def test_write_markdown_without_markers_raises_and_leaves_file(tmp_path):
    target = tmp_path / "DEPLOY.md"
    target.write_bytes(b"# my own notes\n")
    with pytest.raises(C.MarkdownError):
        C.write_markdown(target, "body")
    assert target.read_bytes() == b"# my own notes\n"
    assert C.check_markdown_target(target)


def test_source_uses_no_shell_and_no_local_identity_calls():
    src = (ROOT / "scripts" / "check_clean_env.py").read_text(encoding="utf-8")
    for banned in ("shell=True", "os.system", "gethostname", "getlogin", "getuser", "Path.home", "expanduser"):
        assert banned not in src


# ------------------------------------------------------------------------------------- slow end to end

@pytest.mark.slow
def test_clean_env_check_passes_on_3_13_and_3_14():
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run([sys.executable, "scripts/check_clean_env.py"], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2400)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert lines and lines[-1] == "RESULT: PASS"


# Every test except the slow end-to-end one belongs to the unit tier.
for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and _name != "test_clean_env_check_passes_on_3_13_and_3_14":
        globals()[_name] = pytest.mark.unit(_fn)
