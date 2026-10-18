"""Judge a results file against the handoff section 4 reference, with bands declared before the full run.

The bands and the reference values below are data in this file. The proof that they were declared
before the full four-subset run is the git commit order, not a claim in a document (D-05). The
reference values are comparison targets copied from Claude_Code_Handoff.pdf section 4 (checked
against the rendered pages, D-06); they are never reported results.

Usage (from the repo root, inside the project venv):
    python scripts/check_reference.py [RESULTS_JSON] [--subsets FD00N ...] [--markdown PATH] [--bands-only]

    RESULTS_JSON   defaults to outputs/results.json
    --subsets      judge only these subsets (default: all four)
    --markdown     write the bands and the comparison into the generated block of a markdown file
    --bands-only   print (and optionally write) the bands and reference tables without reading results

Exit codes:
    0  every FAIL-band check passes (WARN and INFO lines never change the exit code)
    1  at least one FAIL: a band violation, or a missing subset, model or metric key
    2  unreadable input (missing file, invalid JSON, top level not an object), or a markdown file
       that lacks the generated markers

Standard library only. It must never import the training kernel: importing run_experiments creates
outputs/ and models/ and loads xgboost and optuna, which would couple the checker to what it judges.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs" / "results.json"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

XGB = "XGBoost (rolling features)"
RF = "Random Forest (rolling features)"
XGB_DEFAULT = "XGBoost default params (rolling features)"
XGB_RAW = "XGBoost (raw sensors only)"
RF_RAW = "Random Forest (raw sensors only)"

# Handoff section 2 (kept sensors, features per row) and section 5.2 (train / test counts).
COUNTS = {
    "FD001": {"train_engines": 100, "train_rows": 20631, "test_engines": 100, "kept_sensors": 14, "n_features": 57},
    "FD002": {"train_engines": 260, "train_rows": 53759, "test_engines": 259, "kept_sensors": 17, "n_features": 69},
    "FD003": {"train_engines": 100, "train_rows": 24720, "test_engines": 100, "kept_sensors": 16, "n_features": 65},
    "FD004": {"train_engines": 249, "train_rows": 61249, "test_engines": 248, "kept_sensors": 17, "n_features": 69},
}

# Handoff section 4.2: mean over three seeds, test set.
REF_TUNED = {
    "FD001": {XGB: {"rmse": 17.77, "nasa": 582, "rmse_capped_truth": 16.60},
              RF: {"rmse": 18.18, "nasa": 590, "rmse_capped_truth": 17.01}},
    "FD002": {XGB: {"rmse": 26.85, "nasa": 8618, "rmse_capped_truth": 14.82},
              RF: {"rmse": 27.19, "nasa": 8996, "rmse_capped_truth": 15.20}},
    "FD003": {XGB: {"rmse": 17.36, "nasa": 617, "rmse_capped_truth": 15.96},
              RF: {"rmse": 18.85, "nasa": 778, "rmse_capped_truth": 17.39}},
    "FD004": {XGB: {"rmse": 28.36, "nasa": 5667, "rmse_capped_truth": 17.18},
              RF: {"rmse": 28.49, "nasa": 5574, "rmse_capped_truth": 17.26}},
}

# Handoff section 4.1 (FD001 only). The default-parameter XGBoost is the golden canary.
REF_CANARY = {"rmse": 20.08, "nasa": 786}
# RMSE is a FAIL band, NASA is INFO only (D-04a).
REF_ABLATION = {XGB_RAW: {"rmse": 17.78, "nasa": 566},
                RF_RAW: {"rmse": 17.88, "nasa": 526}}

# Handoff section 4.4 (capped-target CV RMSE, XGBoost / Random Forest); INFO only.
REF_CV_RMSE = {
    "FD001": {"xgb": 16.25, "rf": 16.45},
    "FD002": {"xgb": 16.17, "rf": 16.45},
    "FD003": {"xgb": 14.69, "rf": 15.06},
    "FD004": {"xgb": 16.11, "rf": 16.30},
}
REF_FD001_XGB_N_ESTIMATORS = 255  # INFO only

# Handoff section 4.4 SHAP ranking (FD001): used only for the top-3 gain WARN (D-04a).
SHAP_LEADERS = ("cycle", "s15_mean", "s4_mean", "s2_mean", "s3_mean", "s11_mean")

BANDS = {"rmse_cycles": 1.0, "nasa_relative": 0.25, "capped_warn_cycles": 1.0,
         "sd_warn_cycles": 0.3, "raw_gap_warn_cycles": 1.0}
EPS = 1e-9

Row = namedtuple("Row", "level subset check value reference band")

MISSING_TEXT = "missing"
MARK_BEGIN = "<!-- BEGIN GENERATED: check_reference.py -->"
MARK_END = "<!-- END GENERATED: check_reference.py -->"
MD_TITLE = "# Reconciliation: baseline re-run vs handoff reference"

BAND_EXACT = "exact"
BAND_RMSE = f"+-{BANDS['rmse_cycles']} cycle"
BAND_NASA = f"+-{round(BANDS['nasa_relative'] * 100)}%"
BAND_CANARY_RMSE = "exact (2 dp)"
BAND_CANARY_NASA = "exact (0 dp)"
BAND_ORDERING = "XGB < RF expected; gaps under ~1 cycle are noise"
BAND_INFO = "informational"

COUNT_KEYS = ("train_engines", "train_rows", "test_engines", "kept_sensors", "n_features")

_MISSING = object()


class MarkdownError(Exception):
    """The markdown target exists but lacks the generated markers."""


# ---------------------------------------------------------------------------------------------
# result access
# ---------------------------------------------------------------------------------------------
def _get(node, *path):
    cur = node
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return _MISSING
    return cur


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num(node, *path):
    v = _get(node, *path)
    return v if _is_num(v) else _MISSING


def _within(value, ref, band):
    # NaN compares False, so a NaN value is a FAIL, never a pass.
    return abs(value - ref) <= band + EPS


def _canary_rmse_ok(value):
    return math.isfinite(value) and round(value, 2) == REF_CANARY["rmse"]


def _canary_nasa_ok(value):
    return math.isfinite(value) and round(value) == REF_CANARY["nasa"]


def _missing_row(level, subset, path_text, reference, band):
    return Row(level, subset, path_text, MISSING_TEXT, reference, band)


def _graded(subset, check_name, path_text, value, reference, band, ok_fn):
    """A PASS / FAIL row; a missing value is a FAIL row that names the dotted path."""
    if value is _MISSING:
        return _missing_row("FAIL", subset, path_text, reference, band)
    return Row("PASS" if ok_fn(value) else "FAIL", subset, check_name, value, reference, band)


# ---------------------------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------------------------
def _fail_band_rows(subset, sub):
    rows = []
    # counts (exact)
    for key in COUNT_KEYS:
        ref = COUNTS[subset][key]
        if key == "kept_sensors":
            raw = _get(sub, "kept_sensors")
            value = len(raw) if isinstance(raw, (list, tuple)) else _MISSING
        else:
            value = _num(sub, key)
        rows.append(_graded(subset, key, key, value, ref, BAND_EXACT, lambda v, r=ref: v == r))

    # tuned XGBoost and RF: mean RMSE (+-1.0 cycle) and mean NASA (+-25% relative)
    for model in (XGB, RF):
        ref = REF_TUNED[subset][model]
        rmse_path = f"models.{model}.rmse.mean"
        rows.append(_graded(subset, f"{model} rmse", rmse_path, _num(sub, "models", model, "rmse", "mean"),
                            ref["rmse"], BAND_RMSE,
                            lambda v, r=ref["rmse"]: _within(v, r, BANDS["rmse_cycles"])))
        nasa_path = f"models.{model}.nasa.mean"
        rows.append(_graded(subset, f"{model} nasa", nasa_path, _num(sub, "models", model, "nasa", "mean"),
                            ref["nasa"], BAND_NASA,
                            lambda v, r=ref["nasa"]: _within(v, r, BANDS["nasa_relative"] * r)))

    if subset == "FD001":
        # the golden canary (RERUN-01, post-run half)
        rows.append(_graded(subset, f"{XGB_DEFAULT} rmse", f"models.{XGB_DEFAULT}.rmse.mean",
                            _num(sub, "models", XGB_DEFAULT, "rmse", "mean"),
                            REF_CANARY["rmse"], BAND_CANARY_RMSE, _canary_rmse_ok))
        rows.append(_graded(subset, f"{XGB_DEFAULT} nasa", f"models.{XGB_DEFAULT}.nasa.mean",
                            _num(sub, "models", XGB_DEFAULT, "nasa", "mean"),
                            REF_CANARY["nasa"], BAND_CANARY_NASA, _canary_nasa_ok))
        # raw-sensor ablations: RMSE is a FAIL band
        for model in (XGB_RAW, RF_RAW):
            ref = REF_ABLATION[model]["rmse"]
            rows.append(_graded(subset, f"{model} rmse", f"models.{model}.rmse.mean",
                                _num(sub, "models", model, "rmse", "mean"), ref, BAND_RMSE,
                                lambda v, r=ref: _within(v, r, BANDS["rmse_cycles"])))
    return rows


def _warn_info_rows(subset, sub):
    rows = []
    # rmse_capped_truth per tuned model: WARN beyond +-1.0, INFO inside
    for model in (XGB, RF):
        ref = REF_TUNED[subset][model]["rmse_capped_truth"]
        value = _num(sub, "models", model, "rmse_capped_truth", "mean")
        if value is _MISSING:
            rows.append(_missing_row("WARN", subset, f"models.{model}.rmse_capped_truth.mean", ref, BAND_RMSE))
        else:
            level = "INFO" if _within(value, ref, BANDS["capped_warn_cycles"]) else "WARN"
            rows.append(Row(level, subset, f"{model} rmse_capped_truth", value, ref, BAND_RMSE))

    # ordering
    x_rmse = _num(sub, "models", XGB, "rmse", "mean")
    r_rmse = _num(sub, "models", RF, "rmse", "mean")
    if x_rmse is _MISSING or r_rmse is _MISSING:
        rows.append(_missing_row("WARN", subset, "ordering", "XGB < RF", BAND_ORDERING))
    else:
        level = "WARN" if x_rmse >= r_rmse else "INFO"
        rows.append(Row(level, subset, "ordering", f"XGB {x_rmse:.4f} RF {r_rmse:.4f}", "XGB < RF", BAND_ORDERING))

    # RMSE seed SD above 0.3, for every model entry present
    models = _get(sub, "models")
    if isinstance(models, dict):
        for model in models:
            sd = _num(sub, "models", model, "rmse", "std")
            if sd is _MISSING:
                rows.append(_missing_row("WARN", subset, f"models.{model}.rmse.std",
                                         BANDS["sd_warn_cycles"], f"SD <= {BANDS['sd_warn_cycles']}"))
            elif not sd <= BANDS["sd_warn_cycles"]:
                rows.append(Row("WARN", subset, f"{model} rmse std", sd, BANDS["sd_warn_cycles"],
                                f"SD <= {BANDS['sd_warn_cycles']}"))

    # gain features
    leaders = ", ".join(SHAP_LEADERS)
    gain = _get(sub, "top_gain_features")
    if isinstance(gain, dict) and gain:
        keys = list(gain)
        rows.append(Row("INFO", subset, "top-5 gain", ", ".join(keys[:5]), leaders, BAND_INFO))
        if not any(k in SHAP_LEADERS for k in keys[:3]):
            rows.append(Row("WARN", subset, "top-3 gain", ", ".join(keys[:3]), leaders,
                            "one of the SHAP leaders expected"))
    else:
        rows.append(_missing_row("WARN", subset, "top_gain_features", leaders, BAND_INFO))

    # CV RMSE
    for key, ref_key in (("xgb_cv_rmse", "xgb"), ("rf_cv_rmse", "rf")):
        ref = REF_CV_RMSE[subset][ref_key]
        value = _num(sub, key)
        if value is _MISSING:
            rows.append(_missing_row("WARN", subset, key, ref, BAND_INFO))
        else:
            rows.append(Row("INFO", subset, key, value, ref, BAND_INFO))

    if subset == "FD001":
        for model in (XGB_RAW, RF_RAW):
            ref = REF_ABLATION[model]["nasa"]
            value = _num(sub, "models", model, "nasa", "mean")
            if value is _MISSING:
                rows.append(_missing_row("WARN", subset, f"models.{model}.nasa.mean", ref, BAND_INFO))
            else:
                rows.append(Row("INFO", subset, f"{model} nasa", value, ref, BAND_INFO))
        raw_x = _num(sub, "models", XGB_RAW, "rmse", "mean")
        if x_rmse is _MISSING or raw_x is _MISSING:
            rows.append(_missing_row("WARN", subset, "raw-sensor gap", 0.0, f"<= {BANDS['raw_gap_warn_cycles']}"))
        else:
            gap = x_rmse - raw_x
            level = "WARN" if gap > BANDS["raw_gap_warn_cycles"] else "INFO"
            rows.append(Row(level, subset, "raw-sensor gap", gap, 0.0,
                            f"tuned XGB above raw-sensor XGB by <= {BANDS['raw_gap_warn_cycles']} expected"))
        n_est = _num(sub, "xgb_n_estimators")
        if n_est is _MISSING:
            rows.append(_missing_row("WARN", subset, "xgb_n_estimators", REF_FD001_XGB_N_ESTIMATORS, BAND_INFO))
        else:
            rows.append(Row("INFO", subset, "xgb_n_estimators", n_est, REF_FD001_XGB_N_ESTIMATORS, BAND_INFO))
    return rows


def check(results, subsets=SUBSETS):
    """Judge `results` and return a list of Row, in subset order. Never skips: absent means FAIL."""
    rows = []
    for subset in subsets:
        sub = results.get(subset) if isinstance(results, dict) else None
        if not isinstance(sub, dict):
            rows.append(Row("FAIL", subset, "subset", MISSING_TEXT, "present", BAND_EXACT))
            continue
        rows.extend(_fail_band_rows(subset, sub))
        rows.extend(_warn_info_rows(subset, sub))
    return rows


# ---------------------------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------------------------
def _fmt_value(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _fmt_reference(r):
    if isinstance(r, bool):
        return str(r)
    if isinstance(r, int):
        return str(r)
    if isinstance(r, float):
        return f"{r:.2f}"
    return str(r)


def format_row(row):
    return (f"{row.level} {row.subset} {row.check}: value {_fmt_value(row.value)} "
            f"reference {_fmt_reference(row.reference)} band {row.band}")


def summarise(rows):
    graded = [r for r in rows if r.level in ("PASS", "FAIL")]
    failed = [r for r in graded if r.level == "FAIL"]
    if failed:
        return f"RESULT: FAIL ({len(failed)} of {len(graded)} checks)"
    warnings = sum(1 for r in rows if r.level == "WARN")
    return f"RESULT: PASS ({len(graded)} checks, {warnings} warnings)"


# ---------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------
def _cell(x):
    return str(x).replace("|", "\\|")


def _table(headers, body_rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in body_rows:
        lines.append("| " + " | ".join(_cell(c) for c in r) + " |")
    return "\n".join(lines)


def bands_markdown():
    """Bands and reference tables, rendered from the module data only (no typed numbers)."""
    sd = BANDS["sd_warn_cycles"]
    band_rows = [
        ("FAIL", "Counts per subset: train_engines, train_rows, test_engines, len(kept_sensors), n_features",
         BAND_EXACT, "exit 1"),
        ("FAIL", "FD001 default-parameter XGBoost mean RMSE (golden canary)", BAND_CANARY_RMSE, "exit 1"),
        ("FAIL", "FD001 default-parameter XGBoost mean NASA (golden canary)", BAND_CANARY_NASA, "exit 1"),
        ("FAIL", "Tuned XGBoost and RF mean RMSE on every subset; FD001 raw-sensor XGBoost and RF mean RMSE",
         BAND_RMSE, "exit 1"),
        ("FAIL", "Tuned XGBoost and RF mean NASA on every subset", BAND_NASA + " relative", "exit 1"),
        ("FAIL", "A missing subset, model or metric key (never a silent skip)", "present", "exit 1"),
        ("WARN", "XGB vs RF RMSE ordering flips on a subset", BAND_ORDERING, "none (exit 0)"),
        ("WARN", "RMSE seed SD of any model", f"SD <= {sd}", "none (exit 0)"),
        ("WARN", "rmse_capped_truth of a tuned model", f"{BAND_RMSE} of the handoff value", "none (exit 0)"),
        ("WARN", "None of the top-3 gain features is a SHAP leader", "one of " + ", ".join(SHAP_LEADERS),
         "none (exit 0)"),
        ("WARN", "FD001 tuned XGBoost RMSE above raw-sensor XGBoost RMSE",
         f"gap <= {BANDS['raw_gap_warn_cycles']} cycle", "none (exit 0)"),
        ("INFO", "rmse_capped_truth inside its band, top-5 gain features, FD001 ablation NASA, "
                 "FD001 xgb_n_estimators, CV RMSE", BAND_INFO, "none (exit 0)"),
    ]
    count_rows = [(s,) + tuple(COUNTS[s][k] for k in COUNT_KEYS) for s in SUBSETS]
    tuned_rows = []
    for s in SUBSETS:
        for model in (XGB, RF):
            ref = REF_TUNED[s][model]
            rmse_lo, rmse_hi = ref["rmse"] - BANDS["rmse_cycles"], ref["rmse"] + BANDS["rmse_cycles"]
            nasa_lo = ref["nasa"] * (1 - BANDS["nasa_relative"])
            nasa_hi = ref["nasa"] * (1 + BANDS["nasa_relative"])
            tuned_rows.append((s, model, f"{ref['rmse']:.2f}", f"[{rmse_lo:.2f}, {rmse_hi:.2f}]",
                               ref["nasa"], f"[{nasa_lo:.1f}, {nasa_hi:.1f}]", f"{ref['rmse_capped_truth']:.2f}"))
    ablation_rows = [(XGB_DEFAULT, f"{REF_CANARY['rmse']:.2f}", BAND_CANARY_RMSE, REF_CANARY["nasa"], BAND_CANARY_NASA)]
    for model in (XGB_RAW, RF_RAW):
        ref = REF_ABLATION[model]
        ablation_rows.append((model, f"{ref['rmse']:.2f}", BAND_RMSE, ref["nasa"], BAND_INFO))
    cv_rows = [(s, f"{REF_CV_RMSE[s]['xgb']:.2f}", f"{REF_CV_RMSE[s]['rf']:.2f}") for s in SUBSETS]

    parts = [
        "### Bands (declared before the full run)",
        "",
        _table(["Tier", "Checks", "Band", "Effect on exit code"], band_rows),
        "",
        "### Reference values (Claude_Code_Handoff.pdf section 4)",
        "",
        "Comparison targets, not reported results.",
        "",
        "Counts per subset (sections 2 and 5.2):",
        "",
        _table(["Subset", "Train engines", "Train rows", "Test engines", "Kept sensors", "n_features"], count_rows),
        "",
        "Tuned models, mean over three seeds (section 4.2):",
        "",
        _table(["Subset", "Model", "RMSE", "RMSE band", "NASA", "NASA band", "RMSE vs capped truth"], tuned_rows),
        "",
        "FD001 default-parameter XGBoost (canary) and raw-sensor ablations (section 4.1):",
        "",
        _table(["Model", "RMSE", "RMSE band", "NASA", "NASA band"], ablation_rows),
        "",
        "Capped-target CV RMSE (section 4.4, informational):",
        "",
        _table(["Subset", "XGBoost", "Random Forest"], cv_rows),
        "",
        f"FD001 tuned XGBoost trees (section 4.1, informational): {REF_FD001_XGB_N_ESTIMATORS}",
        "",
        "SHAP leaders on FD001 (section 4.4): " + ", ".join(SHAP_LEADERS),
    ]
    return "\n".join(parts)


def comparison_markdown(rows, source, source_sha256):
    body_rows = [(r.level, r.subset, r.check, _fmt_value(r.value), _fmt_reference(r.reference), r.band)
                 for r in rows]
    parts = [
        "### Comparison",
        "",
        f"Source: `{source}`",
        "",
        f"SHA256 of the source with CR bytes removed: `{source_sha256}`",
        "",
        _table(["Level", "Subset", "Check", "Value", "Reference", "Band"], body_rows),
        "",
        summarise(rows),
    ]
    return "\n".join(parts)


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
            raise MarkdownError(f"{path} exists but lacks the generated markers; left unchanged")
        new_text = text[:start] + block + text[end + len(MARK_END):]
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_text = f"{MD_TITLE}\n\n{block}\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _source_label(arg):
    p = Path(arg)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="check_reference.py",
        description="Judge a results file against the handoff section 4 reference and its declared bands.")
    parser.add_argument("results", nargs="?", default=None,
                        help="results JSON (default: outputs/results.json)")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS),
                        help="subsets to judge (default: all four)")
    parser.add_argument("--markdown", metavar="PATH", default=None,
                        help="write the bands and comparison into the generated block of this file")
    parser.add_argument("--bands-only", action="store_true",
                        help="print the bands and reference tables without reading results")
    args = parser.parse_args(argv)

    if args.bands_only:
        text = bands_markdown()
        print(text)
        if args.markdown:
            try:
                write_markdown(args.markdown, text)
            except (MarkdownError, OSError, UnicodeDecodeError) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
        return 0

    results_arg = args.results if args.results is not None else str(DEFAULT_RESULTS)
    try:
        raw = Path(results_arg).read_bytes()
        results = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:  # UnicodeDecodeError and JSONDecodeError are ValueError
        print(f"ERROR: cannot read results from {results_arg}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(results, dict):
        print(f"ERROR: {results_arg}: top level is not a JSON object", file=sys.stderr)
        return 2

    subsets = tuple(s for s in SUBSETS if s in args.subsets)
    rows = check(results, subsets)
    for row in rows:
        print(format_row(row))
    summary = summarise(rows)
    print(summary)

    if args.markdown:
        digest = hashlib.sha256(raw.replace(b"\r", b"")).hexdigest()
        body = bands_markdown() + "\n\n" + comparison_markdown(rows, _source_label(results_arg), digest)
        try:
            write_markdown(args.markdown, body)
        except (MarkdownError, OSError, UnicodeDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    return 1 if any(r.level == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
