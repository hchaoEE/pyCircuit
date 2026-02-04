from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LinxISASpec:
    root: Path
    json_path: Path
    data: dict[str, Any]


def find_linxisa_root() -> Path:
    env = os.getenv("LINXISA_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"LINXISA_ROOT is set but not a directory: {p}")

    # Default to a sibling repo checkout: <...>/pyCircuit/../linxisa
    here = Path(__file__).resolve()
    repo_root = here.parents[4]  # binding/python/pycircuit/linxisa/spec.py -> repo root
    cand = (repo_root / ".." / "linxisa").resolve()
    if cand.is_dir():
        return cand
    raise FileNotFoundError(
        "Could not locate linxisa repo. Set LINXISA_ROOT=/path/to/linxisa (expected a sibling checkout at ../linxisa)."
    )


def load_spec(root: Path | None = None) -> LinxISASpec:
    if root is None:
        root = find_linxisa_root()
    json_path = root / "isa" / "spec" / "linxisa-v0.1.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return LinxISASpec(root=root, json_path=json_path, data=data)
