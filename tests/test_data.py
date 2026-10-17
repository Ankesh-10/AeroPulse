"""REBUILD-02 / REBUILD-03: data counts, checksums, kept sensors, feature counts (needs data_raw/)."""
import ast
import hashlib
import inspect
import textwrap

import pandas as pd
import pytest
from conftest import DATA_FILES, ROOT, SUBSETS

import rul_lib as L
import run_experiments as R   # kernel source of the sensor rule; import creates empty outputs/ and models/

pytestmark = pytest.mark.data

EXPECTED = {  # train/test engines+rows, kept sensors, features, dropped sensors
    "FD001": dict(tr_e=100, tr_r=20631, te_e=100, te_r=13096, kept=14, feats=57,
                  dropped=["s1", "s5", "s6", "s10", "s16", "s18", "s19"]),
    "FD002": dict(tr_e=260, tr_r=53759, te_e=259, te_r=33991, kept=17, feats=69,
                  dropped=["s1", "s5", "s18", "s19"]),
    "FD003": dict(tr_e=100, tr_r=24720, te_e=100, te_r=16596, kept=16, feats=65,
                  dropped=["s1", "s5", "s16", "s18", "s19"]),
    "FD004": dict(tr_e=249, tr_r=61249, te_e=248, te_r=41214, kept=17, feats=69,
                  dropped=["s1", "s5", "s18", "s19"]),
}
RULE_TARGETS = ["nunique", "norm_all", "within_sd", "cond", "raw_sd", "keep", "dropped"]


def _kernel_sensor_rule():
    """Compile run_subset's own sensor-selection statements (nunique .. dropped) from the kernel source."""
    fn = ast.parse(textwrap.dedent(inspect.getsource(R.run_subset))).body[0]
    targets = [s.targets[0].id if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) else None
               for s in fn.body]
    start, end = targets.index("nunique"), targets.index("dropped")
    block = fn.body[start:end + 1]
    assert [s.targets[0].id for s in block] == RULE_TARGETS, "sensor rule in run_subset changed"
    return compile(ast.Module(body=block, type_ignores=[]), "run_experiments.run_subset (sensor rule)", "exec")


@pytest.mark.parametrize("name", SUBSETS)
def test_counts_shape_order_and_rangeindex(name):
    e = EXPECTED[name]
    tr, te, rul = L.load_subset(name)
    assert (tr["unit"].nunique(), len(tr), te["unit"].nunique(), len(te)) == (e["tr_e"], e["tr_r"], e["te_e"], e["te_r"])
    assert len(rul) == e["te_e"] and list(tr.columns) == L.COLS and tr.shape[1] == 26
    for df in (tr, te):
        assert not df.isna().any().any()
        assert df.sort_values(["unit", "cycle"]).equals(df)
        assert df.index.equals(pd.RangeIndex(len(df)))            # last_cycle_rows .iloc precondition


@pytest.mark.parametrize("name", SUBSETS)
def test_kept_sensors_and_feature_counts(name):
    """Runs run_subset's own sensor-rule statements (kernel source), then the real make_features."""
    e = EXPECTED[name]
    n_cond, _ = L.SUBSETS[name]
    tr, _, _ = L.load_subset(name)
    tr = L.add_rul(tr)
    ns = {"L": L, "n_cond": n_cond, "tr": tr}
    exec(_kernel_sensor_rule(), ns)
    keep = ns["keep"]
    assert len(keep) == e["kept"] and ns["dropped"] == e["dropped"]
    X = L.make_features(L.ConditionNormaliser(n_cond, keep).fit(tr).transform(tr), keep)
    assert X.shape[1] == e["feats"] == 1 + 4 * len(keep)


def _parse_sums(text):
    out = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = digest.lower()
    return out


def test_sha256sums_match_files_byte_for_byte():
    sums = _parse_sums((ROOT / "data_raw" / "SHA256SUMS").read_text())
    assert sorted(sums) == sorted(DATA_FILES)
    for name, digest in sums.items():
        assert hashlib.sha256((ROOT / "data_raw" / name).read_bytes()).hexdigest() == digest, name
