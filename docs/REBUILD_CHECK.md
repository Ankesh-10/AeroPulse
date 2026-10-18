# Rebuild check: src/rul_lib.py, src/run_experiments.py, app.py

This file records how the three source files were rebuilt from the project handoff and how the rebuild was checked. Later checks that the kernel files are unchanged read the hashes below.

Rebuild commit: 1c3469a feat(01-01): rebuild rul_lib, run_experiments and app verbatim from handoff appendices

## Source

`Claude_Code_Handoff.pdf` (225,591 bytes), code listings in the appendices:

| Appendix | PDF pages | Rebuilt file |
|----------|-----------|--------------|
| A | 7-8 | `src/rul_lib.py` |
| B | 9-11 | `src/run_experiments.py` |
| C | 12 | `app.py` |

## Method

1. The PDF text dump loses indentation and collapses runs of spaces, so it was not used.
2. PyMuPDF 1.28.2, installed in a throwaway venv outside the repository (never in `.venv` or a requirements file), read every glyph of the monospace code font with its position. Each glyph was matched to the nearest line number in the PDF's own line-number column, and its column was computed as `round((x - x0) / advance)` with advance = 0.6 x font size.
3. The output was written with LF endings and no edits: no character was substituted, normalised or fixed by hand. The files are byte copies of the reconstruction.
4. The 150 dpi renders of pages 7 to 12 were read and compared with the reconstruction page by page (table below).
5. The reconstruction SHA256 values equal the values of an independent earlier reconstruction (hash path A, no corrections needed).

## Line counts

Line counts: src/rul_lib.py 140, src/run_experiments.py 215, app.py 86

## SHA256 of the committed files

Check with `grep -E '^[0-9a-f]{64}  ' docs/REBUILD_CHECK.md | sha256sum -c -` from the repository root.

```
1f48cf7020e52612243e476cb8d9167443faaae249ac914bfc70a5f47a772044  src/rul_lib.py
f530a48b2a6fabd4428ff0341c3245437183a0bbe875ce50cd78ac2655c943da  src/run_experiments.py
ba0d034877e0199e8f482c97203952470d68d5c844db9025ab9ffb41d5ad79ad  app.py
```

## Page-by-page visual check

| Page | Appendix | Lines | Result |
|------|----------|-------|--------|
| 7 | A | 1-84 | matches render |
| 8 | A | 85-140 | matches render |
| 9 | B | 1-84 | matches render |
| 10 | B | 85-172 | matches render |
| 11 | B | 173-215 | matches render |
| 12 | C | 1-86 | matches render |

## Indentation and spacing checks (settled from the rendered pages)

- `evaluate`: `r_cap = L.metrics(...)`, `runs[-1]["rmse_capped_truth"] = r_cap["rmse"]` and `if first is None:` sit inside the `for s in seeds:` loop, at 8 spaces (page 10, lines 118-121).
- `__main__`: `rpath.write_text(...)` and `log("saved", n)` sit inside the `for n in names:` loop, at 8 spaces (page 11, lines 214-215).
- `tune_xgb` and `tune_rf`: the nested `objective` function bodies are indented 8 spaces under a 4 space `def objective(trial):` (page 9 lines 66-77, page 10 lines 85-98); the `st = optuna.create_study(...)` lines are at 4 spaces, outside `objective`.
- `run_subset`: the `if name == "FD001":  # ablations` block sits at 4 spaces with its body at 8 spaces; the `for k, v in res["models"].items():` loop sits at 8 spaces inside it with its `log(...)` call at 12 spaces (page 11, lines 184-192).
- `run_subset`: the `# persist artefacts` block sits at 4 spaces (page 11, lines 194-205).
- Spacing the text dump collapses and the reconstruction keeps: `log("XGB   ", ...)` has 3 spaces, `log("RF    ", ...)` has 4 spaces, `log("  ablation", ...)` has 2 spaces, and inline comments such as `within_sd = norm_all.sd  # replaced zeros ...` have 2 spaces before the `#`.
- Known warts kept on purpose (verbatim means verbatim): unused `within_sd`, hard-coded `n_jobs=2`, the `predict_seconds_100_engines` key name.

## Mechanical checks

- Reconstruction output: `A 140 lines`, `B 215 lines`, `C 86 lines`.
- Byte scan (`LC_ALL=C grep` for bytes 0x80-0xff and tab): no match in any of the three files, so they are ASCII with no tabs.
- No trailing whitespace in any of the three files.
- `python -m py_compile` of all three files in the project venv (Python 3.13): clean.
- `src/__init__.py` does not exist, so the module imports as `rul_lib`, the name the saved `prep_*.joblib` files reference.
- `.gitattributes` holds `*.py text eol=lf`, so these files stay byte-identical to the hashes above in working trees on Windows with `core.autocrlf=true`.

## Changes to the reconstruction

None. The reconstruction matched the research hashes exactly, so no line was changed.

## Post-rebuild change (Phase 2, RERUN-04)

The only deviation from Appendix B is commit c0cb307, confined to the persist block of `run_subset`: the gain-importance keys are mapped to feature names, and two additive seed-mean prediction columns are written to `pred_FD00N.csv`. `src/run_experiments.py` therefore no longer matches its hash line above, by design, while `src/rul_lib.py` and `app.py` still do. The verbatim bytes stay checkable with `git show 1c3469a:src/run_experiments.py | sha256sum`. The post-edit hash and the full diff are in `docs/FROZEN.md`.
