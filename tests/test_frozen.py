"""D-12: the frozen kernel and the baseline check themselves against docs/FROZEN.md (unit tier).

Kernel files are hashed as raw bytes (.gitattributes keeps *.py at LF). The baseline is hashed after
removing every CR byte: Windows writes CRLF, git stores LF, and the CR-removed hash is the same everywhere.
"""
import hashlib
import re

import pytest
from conftest import ROOT

pytestmark = pytest.mark.unit

HASH_LINE = re.compile(r"^([0-9a-f]{64})  (\S+)$")
KERNEL = ["src/rul_lib.py", "src/run_experiments.py"]
BASELINE = "outputs/results_baseline.json"


def hashes_in(doc):
    """Return {path: hash} for every `<64 hex>  <path>` line of docs/<doc> (CR stripped)."""
    out = {}
    for line in (ROOT / "docs" / doc).read_text(encoding="utf-8").splitlines():
        m = HASH_LINE.match(line.rstrip("\r"))
        if m:
            assert m.group(2) not in out, f"duplicate hash line for {m.group(2)} in {doc}"
            out[m.group(2)] = m.group(1)
    return out


def sha256_raw(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def sha256_without_cr(rel):
    return hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r", b"")).hexdigest()


def test_frozen_md_records_exactly_the_kernel_and_baseline():
    assert sorted(hashes_in("FROZEN.md")) == sorted(KERNEL + [BASELINE])


@pytest.mark.parametrize("rel", KERNEL)
def test_kernel_file_matches_frozen_hash(rel):
    assert sha256_raw(rel) == hashes_in("FROZEN.md")[rel], f"{rel} changed since the freeze (D-12)"


def test_baseline_matches_frozen_hash_after_cr_removal():
    assert sha256_without_cr(BASELINE) == hashes_in("FROZEN.md")[BASELINE], \
        "outputs/results_baseline.json changed since the freeze (D-09, D-11)"


def test_rul_lib_unchanged_since_verbatim_rebuild():
    assert hashes_in("FROZEN.md")["src/rul_lib.py"] == hashes_in("REBUILD_CHECK.md")["src/rul_lib.py"]


def test_run_experiments_record_differs_from_verbatim():
    # the D-01 edit is recorded by design: FROZEN.md holds the post-edit bytes, REBUILD_CHECK.md the verbatim ones
    assert hashes_in("FROZEN.md")["src/run_experiments.py"] != hashes_in("REBUILD_CHECK.md")["src/run_experiments.py"]
