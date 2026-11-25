# Tutorial: the command line

This walkthrough stretches the staggered-slit sheet of the [tutorials page](README.md) by 5 %
and measures its stiffness, using only the terminal. Every output below is the real output of
these commands.

**Before you start.** Install the code as in the [README](../README.md#installation), with
`pip install -e .`, so that the `kirismiles` command exists. Start Docker before step 3. Then
open a terminal in the `kirismiles` folder and activate the environment:

```bash
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
kirismiles --help                  # lists the commands
docker info > /dev/null && echo "Docker is running"
```

All paths below are relative to the `kirismiles` folder.

## 1. Describe the design: a JSON spec

A **spec** is a small JSON file that says everything about one simulation: the sheet, the
cuts, the material, how the sheet is held, and how far it is pulled. Create
`tutorial_staggered.json`:

```bash
cat > tutorial_staggered.json <<'EOF'
{
  "name": "tutorial_staggered",
  "sheet": {"L": 120, "W": 60},
  "pattern": {"generator": "kirismiles",
              "params": {"code": "P5x45S1/2O0[V42@0,0]N18E3"}},
  "material": {"E": 3500, "nu": 0.38, "h": 0.127},
  "boundary": {"type": "clamped", "in_plane": false},
  "load": {"Dmax_over_L": 0.05, "steps": 12}
}
EOF
```

| Field | Meaning |
|---|---|
| `name` | the run's name: results go to `results/runs/<name>/`. Letters, digits, `_` and `-` |
| `sheet` | length `L` along the pull and width `W`, in mm. The sheet is pulled along x |
| `pattern` | the cuts. Here a KiriSMILES string; a spec can also list cuts, holes and solids as coordinates (see [TOOLS.md](../TOOLS.md)) |
| `material` | Young's modulus `E` (MPa), Poisson's ratio `nu`, thickness `h` (mm). These values are PET, the default |
| `boundary` | `clamped`: the end x = 0 is held and the end x = L is pulled, both gripped so they cannot rotate, as in a tensile tester. Also `roller` (the ends may slide sideways) and `biaxial` (all four edges pulled out). `in_plane: true` keeps the sheet flat, with no buckling |
| `load` | the final stretch as a fraction of L (`0.05` = 5 %, so Δ = 6 mm) and the number of load steps |

Everything except `sheet` and `pattern` is optional and falls back to these same defaults
(with 40 load steps to 10 %). An optional `mesh` section sets the element size.

## 2. Check it

**Validate** the spec. This checks the numbers and the geometry (no crossing cuts, no cut
touching a clamped end) in well under a second, without Docker:

```console
$ kirismiles validate tutorial_staggered.json
{"name": "tutorial_staggered", "ok": true, "errors": [], "counts": {"cuts": 45, "holes": 0, "solids": 0}}
```

**Look at it.** `preview` draws the pattern:

```console
$ kirismiles preview tutorial_staggered.json
{"name": "tutorial_staggered", "ok": true, "files": ["results/runs/tutorial_staggered/pattern.png"]}
```

![pattern.png](img/pattern.png)

**Quick check** (about 15 s, needs Docker). `buckle` meshes the sheet and solves the linear
problems: how stiff the sheet is before it buckles, and how far it can be pulled before it
buckles out of plane (Δ_c):

```console
$ kirismiles buckle tutorial_staggered.json
Job started  2026-11-25 09:51:37   buckle tutorial_staggered
log: results/runs/tutorial_staggered/log.txt
mesh: 31,386 triangles
stiffness: 0.3527 N/mm
buckling: Delta_c = 0.09998, 0.1126, 0.1295, 0.147, 0.1656 mm
files: results/runs/tutorial_staggered/buckling.npz
       results/runs/tutorial_staggered/log.txt
Job ended    2026-11-25 09:51:53   completed in 16 s
{"ok": true, "name": "tutorial_staggered", "analysis": "buckle", "n_cells": 31386, "stiffness_N_per_mm": 0.3526782069679313, "Delta_c_mm": 0.09998007048050003, ...}
```

The sheet buckles after only 0.1 mm of a 6 mm pull: almost the whole test happens in the
buckled state, which is what the simulation in the next step follows. The five Δ_c values
are the first five buckling modes.

## 3. Simulate

```console
$ kirismiles simulate tutorial_staggered.json
Job started  2026-11-25 09:54:00   simulate tutorial_staggered
log: results/runs/tutorial_staggered/log.txt
mesh: 31,386 triangles   material: E = 3500 MPa, nu = 0.38, h = 0.127 mm
stiffness: 0.3527 N/mm   buckling: Delta_c = 0.09998 mm (strain 0.0833 %)
load: 13 steps planned to Delta = 6 mm (the solver adds smaller steps where Newton needs them)
 step   Delta(mm)  strain(%)     F(N)   max|w|(mm)    e_loc   time(s)
    1      0.0500      0.042    0.0176       0.0000    0.0002      22.3
    2      0.1050      0.087    0.0354       0.0804    0.0007     102.3
    3      0.1516      0.126    0.0356       0.2591    0.0014     219.4
    4      0.2191      0.183    0.0357       0.3928    0.0019     269.9
    5      0.3164      0.264    0.0358       0.5285    0.0025     309.3
    6      0.4571      0.381    0.0360       0.6774    0.0031     349.0
    7      0.6603      0.550    0.0362       0.8456    0.0038     394.9
    8      0.9539      0.795    0.0365       1.0379    0.0046     440.3
    9      1.3779      1.148    0.0369       1.2585    0.0055     485.7
   10      1.9905      1.659    0.0375       1.5108    0.0066     537.0
   11      2.8753      2.396    0.0383       1.8135    0.0080     594.9
   12      4.1536      3.461    0.0394       2.1655    0.0095     652.2
   13      6.0000      5.000    0.0410       2.5672    0.0114     708.7
status: completed
rendering deformation.gif ...
wrote results/runs/tutorial_staggered/deformation.gif
files: results/runs/tutorial_staggered/history.npz
       results/runs/tutorial_staggered/deformation.gif
       results/runs/tutorial_staggered/log.txt
Job ended    2026-11-25 10:06:01   completed in 12 min 2 s
{"ok": true, "name": "tutorial_staggered", "analysis": "simulate", "status": "completed", "Delta_c_mm": 0.09998, "stiffness_N_per_mm": 0.3527, "steps": 13, "Delta_final_mm": 6.0, "uz_max_final_mm": 2.5672, ...}
```

The progress (everything above the last line) goes to stderr, and the last line, on stdout,
is the result as JSON, for scripts and agents. Add `--quiet` to print only that line, or
`--print-every 5` for a shorter table.

**Running it in the background.** A simulation takes minutes, so you may prefer to start it
and keep the terminal:

```bash
nohup kirismiles simulate tutorial_staggered.json > simulate.out 2>&1 &
```

Running it again with the same `name` replaces that run's files; change the name to keep both.

## 4. Follow it

The table above is the progress: one row per load step, with the stretch Δ, the strain Δ/L,
the force F, the largest out-of-plane displacement w and the peak local strain e_loc. Where
Newton's method struggles, for example at the buckling point of other designs, the solver
inserts smaller steps, so there can be more rows than planned steps.

For a run in the background, follow the log, which has the same rows:

```bash
tail -f results/runs/tutorial_staggered/log.txt     # Ctrl+C stops watching, not the run
```

The solver runs in a Docker container. To see it, and to stop it:

```bash
docker ps --filter label=kirigami=1                # the running solver, if any
docker stop <CONTAINER ID>                         # stop it (the ID from docker ps)
```

Stopping the `kirismiles` command alone (Ctrl+C or `kill`) may leave the container
running, so check `docker ps` afterwards.

## 5. Analyse

The run folder holds everything:

```console
$ ls results/runs/tutorial_staggered/
buckling.json  buckling.npz  deformation.gif  history.npz  log.txt  pattern.png
provenance.json  spec.json  summary.json
```

| File | Contents |
|---|---|
| `spec.json` | the spec as simulated, with every default filled in |
| `provenance.json` | the code version and solver image that produced the run |
| `summary.json` | the key numbers: buckling stretch, force at each step, final shape |
| `history.npz` | the full history: Δ, F, w and the displacement of every mesh node at every step |
| `deformation.gif` | the sheet as it is stretched |
| `log.txt` | everything the solver printed |
| `pattern.png`, `buckling.npz` | from `preview` and `buckle` in step 2 |

![deformation.gif](img/deformation.gif)

**The stiffness.** `tutorials/fit_stiffness.py` reads `history.npz` and fits two straight lines
to the force against the stretch: one through the steps before buckling, and one over the
buckled part (from 1 % strain to the end, by default):

```console
$ python tutorials/fit_stiffness.py tutorial_staggered
buckling at Delta_c = 0.1000 mm (strain 0.083 %)
stiffness before buckling  k0 = 0.3527 N/mm   (1 step below Delta_c)
stiffness after buckling   k  = 0.0008816 N/mm   (fit over 5 steps, 1 to 5 % strain; F0 = 0.03572 N)
the sheet is 400 times softer once it has buckled
wrote results/runs/tutorial_staggered/stiffness.png
```

![stiffness.png](img/stiffness.png)

**Reading the result.**

- Before buckling the sheet is a stiff spring, **k0 = 0.353 N/mm**. This is the same as the
  quick check's 0.3527 N/mm in step 2, which the fit reproduces from the simulation's first step.
- At Δ_c = 0.1 mm the ligaments between the slits buckle out of plane (the max |w| column
  leaves zero), and the force stops rising: it is 0.0354 N at 0.105 mm and only 0.0410 N at
  6 mm.
- After buckling the sheet is **k = 0.00088 N/mm**, 400 times softer. The pull now mostly
  rotates the ligaments out of plane instead of stretching them, which is what makes kirigami
  so stretchable.

`--from` and `--to` set the fit range in percent strain, for example
`python tutorials/fit_stiffness.py tutorial_staggered --from 2 --to 5`.

**Your own analysis.** `history.npz` loads with numpy:

```python
import numpy as np
d = np.load("results/runs/tutorial_staggered/history.npz")
d["Delta"], d["F"]          # stretch (mm) and force (N), one value per load step
d["uz_max"]                 # largest out-of-plane displacement (mm) per step
d["x"], d["cells"]          # the mesh: node coordinates (mm) and triangles
d["U"]                      # displacement of every node at every step, shape (steps, nodes, 3)
```

**Stress–strain and stretchability.** The run engine of the web app and the MCP server
also writes a stress–strain plot and estimates how far the sheet can be stretched before the
material yields. The same function works on a command-line run:

```console
$ python -c "from kirigami import jobs; print(jobs.analyse_run('tutorial_staggered'))"
{'stretchability': 0.05, 'stretchability_note': 'not reached within the simulated range', 'stress_at_stretchability_MPa': 0.005378422232603635, 'initial_modulus_MPa': 5.553987511306004, 'plot': 'results/runs/tutorial_staggered/stress_strain.png', 'stiffness_before_buckling_N_per_mm': 0.35270581919846417, 'stiffness_after_buckling_N_per_mm': 0.0008815860384972348, 'stiffness_fit_range_pct': [1.0, 5.0]}
```

It writes `results/runs/tutorial_staggered/stress_strain.png` and gives the same two stiffnesses.
The **stretchability** is the strain at which the peak local strain in the material first
reaches 2 %, the yield strain of PET. Here it is not reached within the 5 % simulated, so the
sheet stays elastic all the way; stretch further to find it.

## Next

- Change the design: edit the string, or try another of the [examples](../KIRISMILES.md).
- Compare designs: give each spec its own `name`, simulate them, and fit each one.
- Make it: `kirismiles export tutorial_staggered.json` writes laser-cut SVG and DXF files.
- Every command and option: [TOOLS.md](../TOOLS.md).
