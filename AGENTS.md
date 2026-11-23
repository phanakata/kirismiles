# Working on this repository

Rules for anyone changing the code here, people and coding agents alike. They come from
problems found in this code base; each rule names an example to copy.

## 1. Files and stored results: safety first

- **Never delete or overwrite something the code did not create.** Before removing a folder,
  check a marker that proves it is ours, and stop with a message otherwise.
  Example: `scripts/build_kdb.py` `write()` replaces `--out` only if it holds `report.json`
  and `index.sqlite` (before this check, `--out .` would have deleted the repository).
- **Never delete stored runs** in `results/runs/`. Runs named `nohint*`, `cloud*` and
  `test12h*` belong to the collective experiments.
- **Keep paths inside their folder.** Turn user-supplied run names into folder names with
  `jobs.safe_name()`, use `jobs.unique_name()` for new runs, and resolve-and-check a path
  before reading it (`mcp_server/server.py` `run_dir()`).
- **Write atomically when a reader may look at the file mid-write:** write a temporary file,
  then `os.replace()` it (`kirigami/jobs.py` `_save()`).
- **Temporary files** go in `tempfile` locations and are removed in a `finally` block
  (`mcp_server/server.py` `import_cut_file`). Deletions are scoped to exactly those files.
- **Tracked outputs** (figures, `results/...` files in git) are not regenerated as a side
  effect. If a check rewrites one, compare it with the committed version and restore it
  unless the change was intended.
- **Tests never start real simulations or write into `results/runs/`.** They test input
  checks before the solver is reached, or use a temporary folder.
- **One solve at a time.** Simulations go through the shared queue (`kirigami/jobs.py`).
  Ask before starting a simulation; a laptop with 8 GB of RAM cannot run two.
- **No personal data or secrets** in code, commits, logs or requests to outside services.

## 2. Types

- `from __future__ import annotations` at the top of every typed module.
- Type hints on every public function and method, including the return type.
- Shared aliases live in `kirigami/_types.py` (`Point`, `Polyline`, `Pattern`, `Spec`,
  `Summary`).
- Optional heavy imports (dolfinx and others) go under `if TYPE_CHECKING:`.
- A `# type: ignore[code]` names the error code and the reason, at the end of the line it
  applies to (`scripts/run_config.py` `np.savez`).
- A newly typed module is added to `[tool.mypy] files` in `pyproject.toml`.

## 3. Input checks: right type, right range

- Check inputs where they enter: function arguments, CLI options, HTTP and MCP requests,
  files read from disk.
- Raise `ValueError` with the value and the allowed range:
  `ValueError(f"the hinge t must be between 0 and 16 mm, not {t}")`.
- Use the shared argparse types in `kirigami/argtypes.py` (`positive`, `nonneg`, `count`,
  `count0`): they reject NaN and infinity as well as out-of-range values.
- Database records state their ranges with Pydantic `Field(gt=0)` and similar
  (`kirigami/kdb.py`).
- `spec.validate()` returns a list of problems and never raises.
- **Poisson ratio:** keep only the physical bound −1 < ν < 0.5. Do not add tighter bounds.

## 4. Errors

- Catch the specific exceptions you expect (`OSError`, `ValueError`, `KeyError`, ...). A
  broad `except Exception` is allowed only as a last-resort guard (a worker thread, a server
  request), with a comment saying why.
- One bad input file must not stop a batch: record it as a problem and carry on
  (`scripts/build_kdb.py` `_read_json()`).
- The CLI always prints one JSON line. On failure it prints `{"ok": false, "errors": [...]}`
  and exits with status 1.
- The servers answer a bad request with 400 and the reason. They read the request body
  before replying (Windows resets the connection otherwise).

## 5. Structure

- Shared logic lives in `kirigami/`; scripts, the web app and the MCP server are thin layers
  over it.
- Small functions with one job and a docstring that states units (mm, MPa, kPa) and ranges.
- Small classes or dataclasses that check their own fields (`fvk.Material.__post_init__`).

## 6. Tests and checks before every commit

```
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy
.venv/Scripts/python.exe -m pytest
```

- Every new check gets a test (`tests/test_input_checks.py`).
- A refactor of numerical code must give identical outputs: compare old and new results
  (numbers, database builds, figures) before committing.
- Keep each file's line endings (most files here use CRLF).
- Commit and push only when the repository owner asks.
