"""Prove that requirements.txt alone installs and runs the demo in clean Python environments (D-15, DEMO-07).

Usage (from the repo root, with the project venv's interpreter):
    python scripts/check_clean_env.py [--pythons 3.13 3.14] [--markdown docs/DEPLOY.md] [--keep-scratch]

For every requested Python version the script, in a scratch directory outside the repository:

  1. resolves requirements.txt for Streamlit Cloud's platform (Linux x86_64) with `uv pip compile`
     without installing anything, and checks that all seven pins come back at their pinned version;
  2. creates a throwaway venv with uv, installs only requirements.txt plus the pytest pin read
     from requirements-dev.txt (binary wheels only), and records the package versions;
  3. runs tests/test_demo_lib.py, tests/test_demo_hygiene.py and tests/test_demo_app.py in that venv
     (a failure, an error or a skip counts as a failed check);
  4. starts `streamlit run app.py` headlessly and requires HTTP 200 from /_stcore/health and
     /_stcore/script-health-check;
  5. times a cold demo_lib.load_bundle for FD001-FD004, each in a fresh process (the D-01 measurement).

--markdown PATH writes the result between the check_clean_env.py markers of PATH (a new file gets a
title plus the block, an existing file keeps everything outside the markers). The record holds
versions, statuses, timings, the OS name and release, the CPU count and the date, and never a path,
a user name or a host name.

The script never touches the project .venv, never writes a venv or a lock file inside the repo and
never pushes or deploys anything. Child processes are started from argument lists, never through a
shell, and the scratch directory and the server process tree are always removed.

Exit codes: 0 every interpreter passed; 1 a check failed; 2 the preflight failed (uv missing, an
interpreter not found, a version outside 3.12/3.13/3.14, a markdown target without markers).

Standard library only, no import-time side effects: it never imports the kernel or the demo seam.
"""
import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
ALLOWED_PYTHONS = ("3.12", "3.13", "3.14")
DEFAULT_PYTHONS = ("3.13", "3.14")
DEMO_TESTS = ("tests/test_demo_lib.py", "tests/test_demo_hygiene.py", "tests/test_demo_app.py")
PACKAGES = ("streamlit", "shap", "xgboost", "scikit-learn", "pandas", "numpy", "matplotlib", "joblib", "scipy",
            "starlette")
PRECOMPUTE_THRESHOLD_S = 30.0  # D-01: above this cold load the demo would need precomputed artefacts
MARK_BEGIN = "<!-- BEGIN GENERATED: check_clean_env.py -->"
MARK_END = "<!-- END GENERATED: check_clean_env.py -->"
MD_TITLE = "# Deploying the demo to Streamlit Community Cloud"
# Streamlit Community Cloud runs Debian 11, whose glibc is 2.31, so this uv target accepts exactly
# the Linux x86_64 wheels Cloud can install.
LINUX_PLATFORM = "x86_64-manylinux_2_31"

HEALTH_WAIT_S = 90
SCRIPT_CHECK_TIMEOUT_S = 180

# Run as `python -c TIMING_SNIPPET <subset>` from the repo root: times the heavy imports, then the
# cold demo_lib.load_bundle, and prints one JSON object. Lines are quoted strings so that no source
# line of this script starts with an import statement of a project or runtime module.
TIMING_SNIPPET = "\n".join((
    "import json, sys, time",
    "sys.path.insert(0, 'src')",
    "t0 = time.perf_counter()",
    "import joblib, shap, xgboost, sklearn",
    "import demo_lib",
    "t1 = time.perf_counter()",
    "demo_lib.load_bundle(sys.argv[1])",
    "t2 = time.perf_counter()",
    "print(json.dumps({'import_s': round(t1 - t0, 2), 'load_s': round(t2 - t1, 2)}))",
))

# Run as `python -c VERSIONS_SNIPPET <package> ...`: prints the interpreter and package versions.
VERSIONS_SNIPPET = "\n".join((
    "import json, platform, sys",
    "from importlib import metadata",
    "print(json.dumps({'python': platform.python_version(),"
    " 'packages': {n: metadata.version(n) for n in sys.argv[1:]}}))",
))


class MarkdownError(Exception):
    """The markdown target exists but lacks the generated markers."""


# ---------------------------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------------------------
def check_pythons(versions):
    """Problems with the requested Python versions (empty list: all fine)."""
    versions = list(versions)
    if not versions:
        return ["no Python versions requested"]
    problems = []
    for v in versions:
        if v not in ALLOWED_PYTHONS:
            problems.append(f"unsupported Python version {v!r} (allowed: {', '.join(ALLOWED_PYTHONS)})")
    for v in sorted(set(versions)):
        if versions.count(v) > 1:
            problems.append(f"duplicate Python version {v!r}")
    return problems


def pytest_pin(root):
    """The exact `pytest==` line of requirements-dev.txt."""
    for line in (Path(root) / "requirements-dev.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("pytest=="):
            return line
    raise ValueError("requirements-dev.txt has no pytest== pin")


def make_scratch(root):
    """A fresh scratch directory outside the repository (ValueError if it would lie inside it)."""
    scratch = Path(tempfile.mkdtemp(prefix="aeropulse-cleanenv-")).resolve()
    repo = Path(root).resolve()
    if scratch == repo or repo in scratch.parents:
        try:
            os.rmdir(scratch)  # only succeeds when it is empty
        except OSError:
            pass
        raise ValueError("scratch directory would lie inside the repository")
    return scratch


def venv_python(venv_dir):
    venv_dir = Path(venv_dir)
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def venv_command(uv, version, venv_dir):
    return [str(uv), "venv", "--python", version, str(venv_dir)]


def install_command(uv, venv_py, root, pin):
    return [str(uv), "pip", "install", "--python", str(venv_py), "--only-binary", ":all:",
            "-r", str(Path(root) / "requirements.txt"), pin]


def pytest_command(venv_py):
    return [str(venv_py), "-m", "pytest", *DEMO_TESTS, "-p", "no:cacheprovider", "-q", "-rs"]


def streamlit_command(venv_py, port):
    return [str(venv_py), "-m", "streamlit", "run", "app.py", "--server.headless", "true",
            "--server.port", str(port), "--server.address", "127.0.0.1",
            "--server.scriptHealthCheckEnabled", "true", "--browser.gatherUsageStats", "false"]


def linux_resolve_command(uv, version, root, out_file):
    """Resolve only: nothing is installed and no lock file is written."""
    return [str(uv), "pip", "compile", str(Path(root) / "requirements.txt"),
            "--python-platform", LINUX_PLATFORM, "--python-version", version,
            "--only-binary", ":all:", "--no-header", "--no-annotate", "-q", "-o", str(out_file)]


_SUMMARY_TIME = re.compile(r"\bin \d+(?:\.\d+)?s\b")
_SUMMARY_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped)\b")


def parse_pytest_summary(text):
    """Counts of passed, failed, errors, skipped from the last pytest summary line."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for line in reversed(str(text).splitlines()):
        found = _SUMMARY_COUNT.findall(line)
        if found and _SUMMARY_TIME.search(line):
            for n, word in found:
                key = "errors" if word.startswith("error") else word
                counts[key] += int(n)
            break
    return counts


def tests_ok(counts):
    """Passed above 0 and nothing failed, errored or skipped (a skip means a tier measured nothing)."""
    return counts["passed"] > 0 and counts["failed"] == 0 and counts["errors"] == 0 and counts["skipped"] == 0


def child_env():
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    return env


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _norm_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


_PIN_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)")


def _pins_from_text(text):
    pins = {}
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PIN_LINE.match(line)
        if m:
            pins[_norm_name(m.group(1))] = m.group(2)
    return pins


def requirement_pins(root):
    """name -> version for every `name==version` line of requirements.txt."""
    return _pins_from_text((Path(root) / "requirements.txt").read_text(encoding="utf-8"))


def resolved_pins(text):
    """The same mapping parsed from `uv pip compile` output."""
    return _pins_from_text(text)


def linux_resolve_problems(required, resolved):
    """One problem per required pin that is absent from the resolved set or at another version."""
    problems = []
    for name, version in sorted(required.items()):
        got = resolved.get(name)
        if got is None:
            problems.append(f"{name}=={version} is absent from the resolved set")
        elif got != version:
            problems.append(f"{name}=={version} resolved to {got}")
    return problems


def ascii_safe(text):
    return str(text).encode("ascii", "replace").decode("ascii")


def _tail(text, n):
    lines = [ln.rstrip() for ln in str(text).splitlines() if ln.strip()]
    return "\n".join(ascii_safe(ln) for ln in lines[-n:])


def worst_cold_load(results):
    """Largest load_s over all interpreters and subsets, or None when nothing was timed."""
    vals = [t["load_s"] for r in results for t in r.get("timings", {}).values() if "load_s" in t]
    return max(vals) if vals else None


def d01_line(results):
    worst = worst_cold_load(results)
    if worst is None:
        return "D-01: no cold-load timing recorded"
    threshold = f"{PRECOMPUTE_THRESHOLD_S:g} s"
    if worst >= PRECOMPUTE_THRESHOLD_S:
        return f"D-01: worst local cold load {worst:.2f} s (threshold {threshold}): PRECOMPUTE THRESHOLD EXCEEDED"
    return f"D-01: worst local cold load {worst:.2f} s (threshold {threshold}): precompute not triggered"


def all_ok(results):
    return bool(results) and all(r.get("ok") for r in results)


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _ok(flag):
    return "OK" if flag else "FAIL"


def _tests_text(counts):
    return f"{counts['passed']} passed / {counts['failed'] + counts['errors']} failed / {counts['skipped']} skipped"


def render_markdown(results, when):
    """Markdown for the generated block: versions, statuses, timings and the D-01 line only."""
    system = platform.system()
    date = when.strftime("%Y-%m-%d") if hasattr(when, "strftime") else str(when)
    reqs = [r["request"] for r in results]

    local_rows = [[r["request"], r.get("python") or "-", _ok(r["install_ok"]), _tests_text(r["tests"]),
                   _ok(r["probe_ok"])] for r in results]
    pkg_rows = [[name] + [(r.get("packages") or {}).get(name, "-") for r in results] for name in PACKAGES]
    linux_rows = [[r["request"], LINUX_PLATFORM, _ok(r["linux"].get("ok")), r["linux"].get("packages", 0)]
                  for r in results]
    load_rows = []
    for s in SUBSETS:
        load_rows.append([s] + [f"{r['timings'][s]['load_s']:.2f}" if s in r.get("timings", {}) else "-"
                                for r in results])
    import_row = ["imports (once per process)"] + [
        f"{max(t['import_s'] for t in r['timings'].values()):.2f}" if r.get("timings") else "-" for r in results]

    parts = [
        f"### Clean environments: local {system} run",
        "",
        _table(["Requested Python", "Python", "Install from requirements.txt", "Demo tests", "Headless start"],
               local_rows),
        "",
        "Package versions installed from requirements.txt:",
        "",
        _table(["Package"] + [f"Python {v}" for v in reqs], pkg_rows),
        "",
        "### Streamlit Cloud target: Linux x86_64 resolve (no install)",
        "",
        _table(["Requested Python", "Platform", "Resolve", "Resolved packages"], linux_rows),
        "",
        "This resolve proves that every pin and its dependencies have a Linux x86_64 wheel for that Python. "
        "Nothing was installed or run on Linux.",
        "",
        f"### Cold load in seconds, local {system} measurement (this machine, not the Streamlit Cloud cold start)",
        "",
        _table(["Subset"] + [f"Python {v}" for v in reqs], load_rows + [import_row]),
        "",
        d01_line(results),
        "",
        f"Measured on {system} {platform.release()}, {os.cpu_count()} logical CPUs, {date}",
        "",
        f"RESULT: {'PASS' if all_ok(results) else 'FAIL'}",
    ]
    return "\n".join(parts)


def check_markdown_target(path):
    """Problems with the markdown target (an existing file must hold both markers)."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_bytes().decode("utf-8")
    start, end = text.find(MARK_BEGIN), text.find(MARK_END)
    if start < 0 or end < 0 or end < start:
        return [f"{path.name} exists but lacks the check_clean_env.py markers"]
    return []


def write_markdown(path, body):
    """New file: title plus the marker block. Existing file: replace only the block between the
    markers. Existing file without both markers: untouched, MarkdownError."""
    path = Path(path)
    block = f"{MARK_BEGIN}\n\n{body.strip(chr(10))}\n\n{MARK_END}"
    if path.exists():
        text = path.read_bytes().decode("utf-8")
        start = text.find(MARK_BEGIN)
        end = text.find(MARK_END)
        if start < 0 or end < 0 or end < start:
            raise MarkdownError(f"{path.name} exists but lacks the generated markers; left unchanged")
        new_text = text[:start] + block + text[end + len(MARK_END):]
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_text = f"{MD_TITLE}\n\n{block}\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)


# ---------------------------------------------------------------------------------------------
# process helpers
# ---------------------------------------------------------------------------------------------
def run(cmd, timeout, cwd=None):
    """Run an argument list (no shell) and capture its output; a timeout or a missing program
    becomes a non-zero return code instead of an exception. The default cwd is the repo root:
    uv looks for a .venv in the cwd and its parents, and a scratch directory under the user profile
    can hit an unrelated, broken .venv of a parent directory."""
    try:
        return subprocess.run(cmd, cwd=str(cwd or ROOT), env=child_env(), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, stdout="", stderr=f"timed out after {timeout} s")
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=f"could not start: {exc.__class__.__name__}")


def stop_process(proc):
    """Stop the server and its children. The venv python.exe on Windows is a launcher stub whose
    child is the real server, so the whole tree is killed."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        pass


def _http_status(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError):
        return None


def probe(venv_py, scratch):
    """Start the app headlessly and require 200 from /_stcore/health and /_stcore/script-health-check."""
    port = free_port()
    log_path = Path(scratch) / f"streamlit-{port}.log"
    base = f"http://127.0.0.1:{port}"
    health = script = None
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(streamlit_command(venv_py, port), cwd=str(ROOT), env=child_env(),
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + HEALTH_WAIT_S
            while time.monotonic() < deadline and proc.poll() is None:
                health = _http_status(f"{base}/_stcore/health", 5)
                if health == 200:
                    break
                time.sleep(1)
            if health == 200:
                script = _http_status(f"{base}/_stcore/script-health-check", SCRIPT_CHECK_TIMEOUT_S)
        finally:
            stop_process(proc)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    ok = health == 200 and script == 200 and "Traceback" not in log_text
    if not ok:
        print("STREAMLIT LOG TAIL")
        print(_tail(log_text, 30))
    return {"health": health, "script": script, "ok": ok}


def resolve_linux(uv, version, scratch, root):
    """Resolve requirements.txt for Cloud's platform into the scratch directory; install nothing."""
    out_file = Path(scratch) / f"linux-{version}.txt"
    proc = run(linux_resolve_command(uv, version, root, out_file), 300)
    packages, problems = 0, []
    if proc.returncode == 0 and out_file.exists():
        resolved = resolved_pins(out_file.read_text(encoding="utf-8", errors="replace"))
        packages = len(resolved)
        problems = linux_resolve_problems(requirement_pins(root), resolved)
    else:
        problems = [f"uv pip compile exited {proc.returncode}"]
        print(f"LINUX RESOLVE ERROR TAIL {version}")
        print(_tail(proc.stderr or proc.stdout, 15))
    for p in problems:
        print(f"LINUX-RESOLVE-PROBLEM {version} {ascii_safe(p)}")
    return {"platform": LINUX_PLATFORM, "ok": proc.returncode == 0 and not problems,
            "packages": packages, "problems": problems}


def _last_json(text):
    for line in reversed(str(text).splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def run_one(uv, version, scratch, root, pin):
    """All checks for one interpreter. A failing step is recorded, not raised."""
    rec = {"request": version, "python": None, "packages": {}, "linux": {}, "install_ok": False,
           "tests": {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}, "tests_ok": False,
           "probe_ok": False, "timings": {}, "ok": False}
    print(f"STEP {version} resolve for {LINUX_PLATFORM}", file=sys.stderr, flush=True)
    rec["linux"] = resolve_linux(uv, version, scratch, root)

    venv_dir = Path(scratch) / f"venv{version}"
    venv_py = venv_python(venv_dir)
    print(f"STEP {version} create venv and install", file=sys.stderr, flush=True)
    made = run(venv_command(uv, version, venv_dir), 600)
    if made.returncode == 0:
        inst = run(install_command(uv, venv_py, root, pin), 1200)
        rec["install_ok"] = inst.returncode == 0 and venv_py.exists()
        if not rec["install_ok"]:
            print(f"INSTALL ERROR TAIL {version}")
            print(_tail(inst.stderr or inst.stdout, 20))
    else:
        print(f"VENV ERROR TAIL {version}")
        print(_tail(made.stderr or made.stdout, 20))
    if rec["install_ok"]:
        info = _last_json(run([str(venv_py), "-c", VERSIONS_SNIPPET, *PACKAGES], 120, cwd=scratch).stdout)
        if info:
            rec["python"] = info.get("python")
            rec["packages"] = info.get("packages", {})
        print(f"STEP {version} demo tests", file=sys.stderr, flush=True)
        t = run(pytest_command(venv_py), 1200)
        rec["tests"] = parse_pytest_summary(t.stdout)
        rec["tests_ok"] = t.returncode == 0 and tests_ok(rec["tests"])
        if not rec["tests_ok"]:
            print(f"PYTEST OUTPUT TAIL {version}")
            print(_tail(t.stdout + "\n" + t.stderr, 40))
        print(f"STEP {version} headless start", file=sys.stderr, flush=True)
        rec["probe_ok"] = probe(venv_py, scratch)["ok"]
        print(f"STEP {version} cold-load timing", file=sys.stderr, flush=True)
        for s in SUBSETS:
            timing = _last_json(run([str(venv_py), "-c", TIMING_SNIPPET, s], 600).stdout)
            if timing and "load_s" in timing and "import_s" in timing:
                rec["timings"][s] = {"import_s": timing["import_s"], "load_s": timing["load_s"]}
    rec["ok"] = (rec["linux"].get("ok", False) and rec["install_ok"] and rec["tests_ok"] and rec["probe_ok"]
                 and all(s in rec["timings"] for s in SUBSETS))
    return rec


def print_record(rec):
    v = rec["request"]
    lin = rec["linux"]
    print(f"LINUX-RESOLVE {v} {lin.get('platform', LINUX_PLATFORM)} {_ok(lin.get('ok'))} "
          f"{lin.get('packages', 0)} packages", flush=True)
    c = rec["tests"]
    print(f"CLEAN-ENV {v} python {rec['python'] or '?'} install {_ok(rec['install_ok'])} "
          f"tests {c['passed']}/{c['failed'] + c['errors']}/{c['skipped']} {_ok(rec['tests_ok'])} "
          f"probe {_ok(rec['probe_ok'])}", flush=True)
    for s in SUBSETS:
        t = rec["timings"].get(s)
        if t:
            print(f"COLD-LOAD {v} {s} import {t['import_s']:.2f} load {t['load_s']:.2f}", flush=True)
        else:
            print(f"COLD-LOAD {v} {s} FAIL", flush=True)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def preflight(versions, markdown):
    """Problems that stop the run before any scratch work (empty list: go)."""
    problems = check_pythons(versions)
    if problems:
        return problems
    uv = shutil.which("uv")
    if not uv:
        return ["uv is not on PATH"]
    for v in versions:
        found = run([uv, "python", "find", v], 60)
        if found.returncode != 0:
            problems.append(f"no local Python {v} found by uv (try: uv python install {v})")
    try:
        pytest_pin(ROOT)
    except (ValueError, OSError) as exc:
        problems.append(f"pytest pin unavailable: {exc}")
    if markdown:
        problems += check_markdown_target(markdown)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Clean-environment check for requirements.txt and the demo.")
    ap.add_argument("--pythons", nargs="+", default=list(DEFAULT_PYTHONS), help="Python versions (3.12/3.13/3.14)")
    ap.add_argument("--markdown", help="write the result between the markers of this markdown file")
    ap.add_argument("--keep-scratch", action="store_true", help="debug: keep the scratch directory")
    args = ap.parse_args(argv)

    markdown = None
    if args.markdown:
        markdown = Path(args.markdown)
        if not markdown.is_absolute():
            markdown = ROOT / markdown

    problems = preflight(args.pythons, markdown)
    if problems:
        for p in problems:
            print(f"PREFLIGHT FAIL: {ascii_safe(p)}", file=sys.stderr)
        return 2

    uv = shutil.which("uv")
    pin = pytest_pin(ROOT)
    scratch = make_scratch(ROOT)
    results = []
    try:
        for version in args.pythons:
            rec = run_one(uv, version, scratch, ROOT, pin)
            results.append(rec)
            print_record(rec)
        print(d01_line(results), flush=True)
        if markdown:
            write_markdown(markdown, render_markdown(results, datetime.now(timezone.utc)))
    finally:
        if args.keep_scratch:
            print(f"scratch kept at {scratch}", file=sys.stderr)
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    passed = all_ok(results)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
