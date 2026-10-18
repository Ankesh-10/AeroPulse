"""Unit tier: the DESIGN.md look and the dependency and import rules, as failing tests.

Reads text files only (no data, no models). Colours are compared to the DESIGN.md tokens
parsed from its front matter, never to hex values typed here.
"""
import ast
import json
import re
import subprocess
import tomllib

import pytest

import demo_lib
from conftest import ROOT, SUBSETS

pytestmark = pytest.mark.unit

APP = ROOT / "app.py"
LIB = ROOT / "src" / "demo_lib.py"
CONFIG = ROOT / ".streamlit" / "config.toml"
DESIGN = ROOT / "DESIGN.md"
PY_SOURCES = [APP, LIB]


def _text(path):
    return path.read_text(encoding="utf-8")


def _tokens():
    """DESIGN.md front-matter colours as {token name: upper-case hex}."""
    out, in_colors = {}, False
    for line in _text(DESIGN).splitlines():
        if line.rstrip() == "colors:":
            in_colors = True
            continue
        if in_colors:
            m = re.match(r'^  ([a-z-]+): "(#[0-9A-Fa-f]{6})"\s*$', line)
            if m:
                out[m.group(1)] = m.group(2).upper()
            elif line.strip() and not line.startswith(" "):
                break
    return out


def _theme():
    return tomllib.loads(_text(CONFIG))["theme"]


def _imports(tree_body):
    """(module name, lineno) for every import statement in a list of statements."""
    found = []
    for node in tree_body:
        if isinstance(node, ast.Import):
            found += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module, node.lineno))
    return found


# ---------------------------------------------------------------- config.toml and fonts
def test_config_has_only_a_valid_theme_table():
    import streamlit.config as sc

    cfg = tomllib.loads(_text(CONFIG))
    assert set(cfg) == {"theme"}
    valid = {
        k.split(".")[1]
        for k in sc._config_options_template
        if k.startswith("theme.") and k.count(".") == 1
    }
    theme = cfg["theme"]
    assert set(theme) <= valid, sorted(set(theme) - valid)
    assert not [k for k in theme if k.startswith(("orange", "violet"))]


def test_theme_colours_match_design_tokens():
    t, tok = _theme(), _tokens()
    assert tok, "no colour tokens parsed from DESIGN.md"

    def same(key, token):
        assert t[key].upper() == tok[token], f"{key} != {token}"

    for key in ("primaryColor", "linkColor"):
        same(key, "action")
    same("backgroundColor", "canvas")
    same("secondaryBackgroundColor", "canvas-warm")
    same("textColor", "ink")
    same("codeTextColor", "ink")
    same("borderColor", "hairline")
    same("codeBackgroundColor", "surface")
    for colour, ink, bg in (
        ("red", "late-ink", "late-bg"),
        ("blue", "early-ink", "early-bg"),
        ("green", "healthy-ink", "healthy-bg"),
        ("yellow", "capped-ink", "capped-bg"),
    ):
        same(f"{colour}Color", ink)
        same(f"{colour}TextColor", ink)
        same(f"{colour}BackgroundColor", bg)
    same("grayColor", "ink-mute")
    same("grayTextColor", "ink-mute")
    same("grayBackgroundColor", "canvas-warm")
    expected = [
        tok[n] for n in ("ink", "early-ink", "late-ink", "healthy-ink", "capped-ink", "ink-mute")
    ]
    assert [c.upper() for c in t["chartCategoricalColors"]] == expected


def test_fonts_are_single_name_url_entries():
    t = _theme()
    for key, family in (("font", "Geist"), ("headingFont", "Newsreader"), ("codeFont", "Geist Mono")):
        m = re.fullmatch(r"([^:,]+):(https://fonts\.googleapis\.com/css2\?[^,]+)", t[key])
        assert m and m.group(1) == family, (key, t[key])
    banned = re.compile(r"\b(Inter|Roboto|Open Sans)\b", re.IGNORECASE)
    for path in (CONFIG, APP, LIB):
        assert not banned.search(_text(path)), path.name


# ---------------------------------------------------------------- dependencies (D-00b)
def test_forbidden_dependency_files_absent():
    names = (
        "packages.txt",
        "runtime.txt",
        ".python-version",
        "uv.lock",
        "Pipfile",
        "pyproject.toml",
        "environment.yml",
    )
    assert [n for n in names if (ROOT / n).exists()] == []


def test_requirements_hold_exactly_the_seven_runtime_pins():
    lines = [ln.split("#")[0].strip() for ln in _text(ROOT / "requirements.txt").splitlines()]
    lines = [ln for ln in lines if ln]
    pins = {ln.split("==")[0].strip() for ln in lines if "==" in ln}
    assert pins == {"xgboost", "scikit-learn", "pandas", "numpy", "shap", "streamlit", "matplotlib"}
    assert len(lines) == 7
    assert not [ln for ln in lines if re.match(r"(optuna|pytest)", ln, re.IGNORECASE)]


def test_secrets_file_ignored_and_untracked():
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".streamlit/secrets.toml"], cwd=ROOT, capture_output=True
    )
    assert ignored.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".streamlit/secrets.toml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == ""


# ---------------------------------------------------------------- copy and markup rules
def test_sources_are_ascii_and_free_of_cliches():
    for path in (APP, LIB, CONFIG):
        path.read_bytes().decode("ascii")
    cliches = re.compile(r"elevate|seamless|unleash|next-gen|game-changer|delve", re.IGNORECASE)
    for path in PY_SOURCES:
        assert not cliches.search(_text(path)), path.name


def test_page_icon_is_a_material_icon():
    assert re.search(r'page_icon=":material/[a-z0-9_]+:"', _text(APP))


def test_no_raw_html_or_css():
    for path in PY_SOURCES:
        text = _text(path)
        assert "unsafe_allow_html" not in text and "<style" not in text, path.name


def test_single_title_no_other_headings_tabs_uploads_or_images():
    tree = ast.parse(_text(APP))
    attrs = [
        n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert attrs.count("title") == 1
    banned = {"header", "subheader", "tabs", "file_uploader", "image", "balloons", "snow"}
    assert not banned & set(attrs), banned & set(attrs)


def test_every_pyplot_is_closed():
    text = _text(APP)
    shown, closed = text.count("st.pyplot("), text.count("plt.close(")
    assert shown == closed and shown >= 3


# ---------------------------------------------------------------- imports (D-13)
def test_app_imports():
    text = _text(APP)
    tree = ast.parse(text)
    assert "from demo_lib import" not in text
    mods = _imports(tree.body)
    allowed = {"logging", "sys", "pathlib", "streamlit", "rul_lib", "demo_lib", "matplotlib.pyplot", "numpy", "pandas"}
    assert {m for m, _ in mods} <= allowed, {m for m, _ in mods} - allowed
    line = dict(mods)
    assert line["streamlit"] < line["matplotlib.pyplot"]
    cached = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        for d in n.decorator_list
        if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "cache_resource"
    ]
    assert len(cached) == 1
    deco = cached[0].decorator_list[0]
    assert {k.arg: getattr(k.value, "value", None) for k in deco.keywords}.get("max_entries") == 1


def test_demo_lib_imports():
    tree = ast.parse(_text(LIB))
    mods = _imports(tree.body)
    allowed = {
        "matplotlib",
        "matplotlib.figure",
        "dataclasses",
        "json",
        "math",
        "threading",
        "pathlib",
        "numpy",
        "pandas",
        "rul_lib",
    }
    assert {m for m, _ in mods} <= allowed, {m for m, _ in mods} - allowed
    # Heavy modules are imported inside functions only.
    inside = {m for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for m, _ in _imports(n.body)}
    for heavy in ("joblib", "shap", "xgboost", "matplotlib.pyplot"):
        assert heavy in inside, heavy
    everything = {m for n in ast.walk(tree) for m, _ in _imports([n])}
    assert not [m for m in everything if m and ("run_experiments" in m or "optuna" in m)]
    # matplotlib.use("Agg") runs before every other module-level matplotlib import.
    use_line = next(
        n.lineno
        for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "matplotlib.use"
    )
    first_import = min(line for m, line in mods if m == "matplotlib")
    assert first_import < use_line
    assert all(line > use_line for m, line in mods if m.startswith("matplotlib.") or m == "matplotlib.pyplot")


# ---------------------------------------------------------------- numbers and models (DEMO-05, DEMO-08)
def test_nasa_cost_comes_from_rul_lib():
    text = _text(LIB)
    assert "L.nasa_score(" in text
    assert not re.search(r"exp" + r"\(", text)


def _standalone(number, text):
    return re.search(r"(?<![0-9.])" + re.escape(number) + r"(?![0-9]|\.[0-9])", text)


def test_no_hard_coded_evaluation_numbers():
    path = ROOT / "outputs" / "results_baseline.json"
    if not path.is_file():
        pytest.skip("outputs/results_baseline.json absent")
    res = json.loads(_text(path))
    for subset in SUBSETS:
        m = res[subset]["models"][demo_lib.MODEL_KEY]
        for number in (f"{m['rmse']['mean']:.2f}", f"{m['nasa']['mean']:.0f}"):
            for src in PY_SOURCES:
                assert not _standalone(number, _text(src)), (subset, number, src.name)


def test_only_baseline_models_are_served():
    for src in PY_SOURCES:
        text = _text(src)
        assert not re.search(r"\brf_|results_variants|no_?cycle", text), src.name
    patterns = set(re.findall(r"(?:prep|xgb|rf)_\{subset\}\.(?:joblib|json)", _text(LIB)))
    assert patterns == {"prep_{subset}.joblib", "xgb_{subset}.json"}


def test_colour_literals_are_design_tokens():
    tok = _tokens()
    allowed = set(tok.values()) | {v.upper() for v in demo_lib.SHAP_TINTS.values()}
    for src in PY_SOURCES:
        literals = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", _text(src))}
        assert literals <= allowed, (src.name, literals - allowed)
    style = demo_lib.plot_style()
    expected = {
        "figure.facecolor": "canvas",
        "axes.facecolor": "canvas",
        "axes.edgecolor": "hairline",
        "grid.color": "hairline",
        "xtick.color": "ink-mute",
        "ytick.color": "ink-mute",
        "axes.labelcolor": "ink",
        "text.color": "ink",
    }
    for key, token in expected.items():
        assert style[key].upper() == tok[token], key
