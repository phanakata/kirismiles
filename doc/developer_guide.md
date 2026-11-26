# Developer guide: how the code fits together

This chapter is for anyone who wants to read, review or extend the KiriSMILES code. It follows
one design from a line of text to a finished simulation, names the file that handles each step,
and suggests an order for reading the code. It describes release **v0.2.0**.

## 1. The code at a glance

| Part | Lines (v0.2.0) | What it is |
|---|---:|---|
| `kirigami/` | 5,750 | the library: notation, specs, meshing, plate and shell models, job queue |
| `scripts/` | 6,600 | command line, studies of the paper, figures, MuJoCo and database tools |
| `tests/` | 1,130 | pytest suite (no test starts a real simulation) |
| `mcp_server/` | 520 | the MCP server for AI assistants |
| `webapp/` | 880 | the local web app (Python server and one HTML page) |

About 14,300 lines of Python in total, of which about 12,100 are not blank or comments. The part
worth learning well is smaller: the library minus the experimental `tube.py`, about 5,000
lines. Most of `scripts/` are one-off studies and figure scripts that use the library.

## 2. Two places the code runs

Docker is used for one thing only: the finite element solver (FEniCSx, PETSc, SLEPc, gmsh), which
lives in the image `dolfinx/dolfinx:v0.9.0`. Everything else runs directly on your computer.

```
 your computer (host)                                  Docker container
 -----------------------------------------             -------------------------------
 notation, specs, checks, previews                     kirigami/api.py
 command line, web app, MCP server                       |- geometry.py  (gmsh mesh)
 job queue, stored results, plots          --run-->      |- fvk.py       (FvK plate)
 SVG/DXF import and export, MuJoCo                       '- shell.py     (KL shell)
                                           <-files--   results/runs/<name>/
```

You never log into the container. When a command needs the solver, the code starts the container
with `docker run`, mounts the repository at `/work`, runs one command and lets the container exit.
The results are written to `results/runs/<name>/` in the repository, where the host reads them.
If `dolfinx` can be imported on the host (a conda install, for example), `scripts/kirigami_cli.py`
runs the solver directly instead. The job queue behind the web app and the MCP server still
requires Docker in v0.2.0.

## 3. The flow of one design

```
 "P5x45S1/2O0[V42@0,0]N18E3"                         a KiriSMILES string
        |
        |  kirigami/kirismiles.py   parse(): string -> slits, cuts, holes, solids
        v
 JSON spec: pattern + sheet (L, W) + material (E, nu, h) + mesh + loading
        |
        |  kirigami/spec.py         load(), validate(), pattern(), save()
        v                           every route into the code passes through a spec
 front ends on the host
        |  scripts/kirigami_cli.py  validate | preview | buckle | simulate | export | ...
        |  webapp/server.py  ---+
        |  mcp_server/server.py -+-> kirigami/jobs.py  Runner: a queue, one solve at a time
        |
        |  kirigami_cli.in_container():  docker run ... python3 -m kirigami.api <command>
        v
 inside the container
        |  kirigami/api.py          buckle(), simulate(), stiffness(), export_mesh(), ...
        |     kirigami/geometry.py  mesh_pattern(): gmsh mesh with the cuts as free edges
        |     kirigami/fvk.py       Material, Sheet: plane stress, linear buckling (buckle)
        |     kirigami/shell.py     ShellSheet: nonlinear Kirchhoff-Love shell (simulate)
        v
 results/runs/<name>/   spec.json, provenance.json, buckling.json, history.npz, PNG, GIF
        |
        v
 back on the host
        kirigami/strain.py    local material strain from saved displacements
        kirigami/export.py    laser-cut files (SVG, DXF) and STL
        kirigami/mjcf.py      the same mesh as a MuJoCo model
        kirigami/kirchhoff_bend.py   corrected bending for MuJoCo (Kirchhoff plate)
        kirigami/kdb.py       database records (scripts/build_kdb.py builds the database)
        kirigami/provenance.py   which commit and image produced the run
```

Two analyses use two models:

- **`buckle`** (seconds) uses the Föppl–von Kármán plate in `fvk.py`. It solves the in-plane
  problem for the stiffness, then the eigenvalue problem for the buckling onset $\Delta_c$ and
  the first modes.
- **`simulate`** (minutes) uses the geometrically exact Kirchhoff–Love shell in `shell.py`. It
  seeds the first buckling mode as a small imperfection, then ramps the end displacement and
  saves every step to `history.npz`.

## 4. Where to start reading

Read in this order. Each step builds on the one before.

1. **`kirigami/spec.py`** (240 lines): the data model. A spec is a plain dictionary holding a
   pattern, the sheet, the material, the mesh and the loading. Read `load()`, `validate()` and
   `pattern()`. The `GENERATORS` table shows the ways a pattern can be given; the general one
   is `"kirismiles"`.
2. **`kirigami/kirismiles.py`** (860 lines): the notation. `parse()` turns a string into
   pieces. Read it next to `KIRISMILES.md`, which defines the notation. `grammar.py` is the
   lattice-only notation that `kirismiles.py` builds on.
3. **`scripts/kirigami_cli.py`**, especially `in_container()`: how the host hands work to the
   container, and the rule that every command prints one JSON line.
4. **`kirigami/api.py`** (330 lines): what happens inside the container. Follow `buckle()` from
   the spec to `buckling.json`, then `simulate()`.
5. **`kirigami/geometry.py`**, then **`kirigami/fvk.py`**: meshing, then the physics. The forms
   in `fvk.py` (`bending_energy()`, `solve_inplane()`, `linear_buckling()`) are the equations of
   the engine section of the paper.
6. **`kirigami/shell.py`**: the nonlinear shell used by `simulate()`.
7. **`kirigami/jobs.py`**, then `mcp_server/server.py` and `webapp/server.py`: the queue,
   the solver lock, the store of runs indexed by design (`design_key()`, `find_runs()`), and the
   two front ends built on them.
8. As needed: `importer.py` (SVG/DXF to a string), `export.py`, `mjcf.py`, `kdb.py`,
   `strain.py`, `provenance.py`. `tube.py` is experimental and not part of the paper.

**Learn by tracing one call.** Run a buckling analysis and follow it through steps 3 → 4 → 5:

```
python scripts/kirigami_cli.py validate specs/alternating.json   # host only, instant
python scripts/kirigami_cli.py buckle   specs/alternating.json   # starts the container, ~20 s
```

then open `results/runs/alternating/buckling.json` and `provenance.json`.

## 5. Rules for changing the code

`AGENTS.md` lists the rules for everyone who changes the code, people and coding agents alike:
guarded deletions, type hints, input checks with clear messages, specific exceptions, and the
checks to run before every commit:

```
python -m ruff check .
python -m mypy
python -m pytest
```

A change to numerical code must give identical results before and after; compare numbers,
database builds and figures.

## 6. Versions

Releases are git tags following semantic versioning: `v0.2.0` is the version described in the
first preprint, and `v0.1.0` the release before it. Commits between releases need no tag. A bug fix gives the next patch release
(`v0.1.1`); new features or changed results give the next minor release (`v0.2.0`). A tag is
never moved once published.
