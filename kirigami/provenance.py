"""What code produced a run: recorded next to every run as provenance.json.

  code_version   short git commit, with "-dirty" when tracked files under kirigami/ or
                 scripts/ differ from it, or when they hold files git does not track
  git_commit     full commit (None outside a git checkout or without git, e.g. in the
                 solver container)
  changed_files  the changed or untracked files under kirigami/ and scripts/ (at most 20)
  source_hash    sha1 over every kirigami/*.py and scripts/*.py: identifies the exact
                 code even for a dirty tree or when git is not available
  image          solver container image, and image_id its exact digest (None when
                 Docker is not reachable, e.g. from inside the container)

Only the standard library is used, so this works on the host and in the container.
"""
import datetime
import hashlib
import json
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "dolfinx/dolfinx:v0.9.0"
CODE_DIRS = ("kirigami", "scripts")


def _run(cmd):
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=20)
        # rstrip only: porcelain status lines start with a space (' M path')
        return r.stdout.rstrip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def source_hash():
    h = hashlib.sha1()
    for d in CODE_DIRS:
        for f in sorted((ROOT / d).rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            h.update(f.relative_to(ROOT).as_posix().encode())
            h.update(b"\0")
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


@lru_cache(maxsize=None)
def image_id(image=IMAGE):
    return _run(["docker", "image", "inspect", "--format", "{{.Id}}", image])


def collect(image=IMAGE):
    commit = _run(["git", "rev-parse", "HEAD"])
    changed = None
    if commit:
        st = _run(["git", "status", "--porcelain", "--untracked-files=all", "--", *CODE_DIRS])
        changed = [l[3:] for l in (st or "").splitlines()
                   if l[3:].endswith(".py") and "__pycache__" not in l]
    dirty = bool(changed)
    return {
        "code_version": (commit[:7] + ("-dirty" if dirty else "")) if commit else None,
        "git_commit": commit,
        "dirty": dirty if commit else None,
        "changed_files": changed[:20] if changed else [],
        "source_hash": source_hash(),
        "image": image,
        "image_id": image_id(image),
        "python": platform.python_version(),
        "recorded": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def write(run_dir, image=IMAGE):
    """Write provenance.json into a run folder and return its contents."""
    p = collect(image)
    Path(run_dir, "provenance.json").write_text(json.dumps(p, indent=1))
    return p


if __name__ == "__main__":
    print(json.dumps(collect(), indent=1))
