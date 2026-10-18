"""Gate, launch and record the full baseline run (D-10, D-05).

Usage (from the repo root, inside the project venv):
    python scripts/run_baseline.py [FD001 FD002 FD003 FD004] [--resume] [--check-only]

It checks the start conditions (valid subset names, the repo .venv on Python 3.13, a clean git tree
for the kernel / scripts / tests / config / data, a committed bands file, no stale files in outputs/
or models/), then issues the single D-10 command once:

    python -u src/run_experiments.py <subsets>

with stdout and stderr redirected to outputs/run_log.txt, and appends one invocation record to
outputs/run_manifest.json whatever the child's exit code. The manifest holds the git SHA, the bands
commit, timings and the package versions, and never an absolute path, user name or host name.

    --resume       a first run was interrupted: run only the missing subsets, append to the log and
                   the manifest (outputs/results.json must exist and lack the requested subsets)
    --check-only   run the preflight and print PREFLIGHT OK, write nothing

Exit codes: the child's exit code (1 when it was interrupted before returning); 2 on a failed preflight.
Results are written to outputs/ and models/ by the kernel, as it decides.

Standard library only, no import-time side effects: it must never import the kernel (importing
run_experiments creates outputs/ and models/ and loads xgboost and optuna).
"""
import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
PACKAGES = ("xgboost", "scikit-learn", "pandas", "numpy", "shap", "streamlit", "matplotlib", "optuna", "joblib")
KERNEL_N_JOBS = 2  # recorded, not set: the kernel hard-codes n_jobs=2
GIT_CLEAN_PATHS = ("src", "scripts", "tests", "pytest.ini", "requirements.txt", "requirements-dev.txt", "data_raw")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def check_subsets(subsets):
    """Problems with the requested subset names (empty list: all fine)."""
    problems = []
    if not subsets:
        return ["no subsets requested"]
    for s in subsets:
        if s not in SUBSETS:
            problems.append(f"unknown subset {s!r} (allowed: {', '.join(SUBSETS)})")
    for s in sorted(set(subsets)):
        if list(subsets).count(s) > 1:
            problems.append(f"duplicate subset {s!r}")
    return problems


def check_outputs(root, subsets, resume):
    """A fresh run needs empty outputs/ and models/; a resume needs results.json lacking the subsets."""
    root = Path(root)
    problems = []
    if not resume:
        for sub in ("outputs", "models"):
            base = root / sub
            if base.is_dir():
                for p in sorted(base.rglob("*")):
                    if p.is_file():
                        problems.append(f"stale file for a fresh run: {p.relative_to(root).as_posix()}")
        return problems
    rpath = root / "outputs" / "results.json"
    if not rpath.is_file():
        return ["resume needs outputs/results.json, which does not exist"]
    try:
        data = json.loads(rpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"outputs/results.json is not readable JSON: {exc}"]
    if not isinstance(data, dict):
        return ["outputs/results.json is not a JSON object"]
    for s in subsets:
        if s in data:
            problems.append(f"subset {s} is already in outputs/results.json")
    return problems


def check_interpreter(root):
    """The project venv on Python 3.13, never the global Python."""
    problems = []
    if sys.version_info[:2] != (3, 13):
        problems.append(f"Python {sys.version_info[0]}.{sys.version_info[1]} is not 3.13")
    if Path(sys.prefix).resolve() != (Path(root) / ".venv").resolve():
        problems.append("interpreter is not the repo .venv (sys.prefix differs)")
    return problems


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def check_git(root):
    """Return (problems, info): clean tree for the kernel and its inputs, HEAD, and the bands commit."""
    problems = []
    info = {"git_sha": "", "git_dirty": [], "bands_commit": ""}
    st = _git(root, "status", "--porcelain", "--", *GIT_CLEAN_PATHS)
    if st.returncode != 0:
        problems.append(f"git status failed: {st.stderr.strip()}")
    else:
        info["git_dirty"] = [ln for ln in st.stdout.splitlines() if ln.strip()]
        problems.extend(f"uncommitted change: {ln}" for ln in info["git_dirty"])
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0 or not head.stdout.strip():
        problems.append(f"git rev-parse HEAD failed: {head.stderr.strip()}")
    else:
        info["git_sha"] = head.stdout.strip()
    bands = _git(root, "log", "-1", "--format=%H", "--", "scripts/check_reference.py")
    if bands.returncode != 0:
        problems.append(f"git log for scripts/check_reference.py failed: {bands.stderr.strip()}")
    elif not bands.stdout.strip():
        problems.append("scripts/check_reference.py was never committed (no bands commit)")
    else:
        info["bands_commit"] = bands.stdout.strip()
    return problems, info


def build_command(subsets):
    return [sys.executable, "-u", "src/run_experiments.py", *subsets]


def keep_awake():
    """Ask Windows to keep the system awake while this thread runs; True when the OS accepted."""
    if sys.platform != "win32":
        return False
    try:
        fn = ctypes.windll.kernel32.SetThreadExecutionState
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_uint32
        return fn(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    except Exception:
        return False


def _utc(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _interpreter_label(exe, root):
    """Repo-relative POSIX path when the interpreter lies inside the repo, else only its file name."""
    try:
        return Path(os.path.abspath(exe)).relative_to(Path(os.path.abspath(root))).as_posix()
    except ValueError:
        return Path(exe).name


def _package_versions():
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_and_record(cmd, root, log_path, manifest_path, meta, append):
    """Run cmd once from root (no shell, output to log_path), then append a record to the manifest."""
    log_path, manifest_path = Path(log_path), Path(manifest_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    t0 = time.monotonic()
    code = None
    try:
        with open(log_path, "ab" if append else "wb") as log:
            code = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
    finally:
        ended = time.time()
        wall = time.monotonic() - t0
        record = {
            **meta,
            "command": ["python", *cmd[1:]],
            "python_executable": _interpreter_label(cmd[0], root),
            "started_at": _utc(started),
            "ended_at": _utc(ended),
            "started_epoch": int(started),
            "ended_epoch": int(ended),
            "wall_seconds": round(wall, 1),
            "exit_code": code,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "kernel_n_jobs": KERNEL_N_JOBS,
            "packages": _package_versions(),
        }
        doc = {"invocations": []}
        if manifest_path.exists():
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        doc["invocations"].append(record)
        manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8", newline="\n")
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gate, launch and record the full baseline run.")
    ap.add_argument("subsets", nargs="*", help="subset names (default: all four)")
    ap.add_argument("--resume", action="store_true", help="run only missing subsets, append to log and manifest")
    ap.add_argument("--check-only", action="store_true", help="run the preflight only")
    args = ap.parse_args(argv)
    subsets = list(args.subsets) or list(SUBSETS)

    git_problems, info = check_git(ROOT)
    problems = (
        check_subsets(subsets)
        + check_interpreter(ROOT)
        + git_problems
        + check_outputs(ROOT, subsets, args.resume)
    )
    if problems:
        for p in problems:
            print(f"PREFLIGHT FAIL: {p}", file=sys.stderr)
        return 2
    print(f"PREFLIGHT OK git {info['git_sha']} bands {info['bands_commit']} subsets {' '.join(subsets)}")
    if args.check_only:
        return 0

    awake = keep_awake()
    code = run_and_record(
        build_command(subsets),
        ROOT,
        ROOT / "outputs" / "run_log.txt",
        ROOT / "outputs" / "run_manifest.json",
        {**info, "subsets": subsets, "resume": args.resume, "keep_awake": awake},
        append=args.resume,
    )
    print(f"EXIT {code}")
    return 1 if code is None else code


if __name__ == "__main__":
    sys.exit(main())
