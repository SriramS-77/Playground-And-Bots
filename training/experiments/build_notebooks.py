"""Turn the `nb_*.py` cell-marked scripts into runnable notebooks.

Each script uses the usual percent format:

    # %% [markdown]
    # Some prose
    # %%
    code

Run `python build_notebooks.py` to (re)generate every `.ipynb`, or pass names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse(text: str):
    cells, kind, buf = [], "code", []

    def flush():
        src = "\n".join(buf).strip("\n")
        if src:
            cells.append((kind, src))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
            continue
        if kind == "markdown":
            buf.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        else:
            buf.append(line)
    flush()
    return cells


def build(path: Path) -> Path:
    cells = []
    for i, (kind, src) in enumerate(parse(path.read_text(encoding="utf-8"))):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            "id": f"c{i:03d}",
        }
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = path.with_suffix(".ipynb").with_name(path.stem.replace("nb_", "") + ".ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(str(p) for p in HERE.glob("nb_*.py"))
    for t in targets:
        print("built", build(Path(t)).name)
