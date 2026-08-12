"""Projection utility: internal dependency signal = activation · unit_direction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def project(act: np.ndarray, direction: np.ndarray) -> float:
    vhat = direction / np.linalg.norm(direction)
    return float(act @ vhat)


def project_rollouts(rollout_dir: str | Path, direction_path: str | Path) -> list[dict]:
    """Adds 'projection' to every turn record (from its saved activation)."""
    v = np.load(direction_path)
    out = []
    for p in sorted(Path(rollout_dir).glob("*.jsonl")):
        if ".judged." in p.name:
            continue
        for line in open(p):
            r = json.loads(line)
            act = np.load(r["activation_ref"])
            if act.ndim == 2:                     # all-layer acts: slice
                act = act[r["activation_layer"]]
            r["projection"] = project(act, v)
            out.append(r)
    return out
