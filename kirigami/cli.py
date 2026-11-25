"""The `kirismiles` command, installed by `pip install -e .` from a checkout.

It runs scripts/kirigami_cli.py, so `kirismiles simulate specs/x.json` is the same as
`python scripts/kirigami_cli.py simulate specs/x.json`. The command needs the checkout
(the scripts, the specs and the results/ folder live there), which is why the install is
an editable one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    """Entry point of the `kirismiles` command; returns the exit status."""
    root = Path(__file__).resolve().parent.parent
    if not (root / "scripts" / "kirigami_cli.py").exists():
        print(json.dumps(dict(ok=False, errors=[
            f"scripts/kirigami_cli.py not found next to {root / 'kirigami'}: install from a "
            "checkout with `pip install -e .` (an editable install), not from a built wheel"])))
        return 1
    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import kirigami_cli
    return kirigami_cli.main()


if __name__ == "__main__":
    sys.exit(main())
