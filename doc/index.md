# KiriSMILES documentation

Where to find each topic.

| Topic | Contents | Document |
|---|---|---|
| Introduction | what KiriSMILES is, citing it | `README.md` |
| Installation | Python, Docker, the FEniCSx image, optional parts | `README.md` § Installation |
| The KiriSMILES notation | the line notation, grammar, identity | `KIRISMILES.md` |
| Specs | the JSON spec: sheet, material, mesh, loading, boundary | `schemas/` |
| Command line, web app and MCP server | the platform for people and AI agents | `TOOLS.md`, `README.md` § 4 |
| Theory of the engine | plate and shell models, buckling, loading | the paper, Sec. IV |
| Validation and benchmarks | comparison with published results | `BENCHMARK.md` |
| Python API | functions and classes of `kirigami/` | the docstrings in `kirigami/` |
| Developer guide | how the code fits together, where to start | [developer_guide.md](developer_guide.md) |
| Contributing | rules for changing the code | `AGENTS.md` |

## Building the PDF of the developer guide

```
pandoc doc/developer_guide.md -o doc/developer_guide.pdf --pdf-engine=lualatex \
  -V geometry:margin=2.2cm -V fontsize=11pt -V colorlinks=true -V monofont=Consolas \
  -V title="KiriSMILES developer guide" -V subtitle="Release v0.2.0"
```
