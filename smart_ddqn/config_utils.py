from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        cfg = json.load(stream)

    _validate(cfg)
    cfg["_config_path"] = str(path)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    for section in ("run", "habitat", "observation", "action", "model", "agent", "aux"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")

    if int(cfg["run"]["num_envs"]) <= 0:
        raise ValueError("run.num_envs must be positive")
    if int(cfg["action"]["num_actions"]) <= 1:
        raise ValueError("action.num_actions must be greater than 1 for DDQN")

    dims = cfg["observation"]["dims"]
    for key in ("img", "goal", "graph", "teacher_orientation"):
        if key not in dims:
            raise ValueError(f"Missing observation.dims.{key}")
        if int(dims[key]) <= 0:
            raise ValueError(f"observation.dims.{key} must be positive")

    if int(dims["teacher_orientation"]) != 1:
        raise ValueError("teacher_orientation must have dimension 1 and contain yaw in radians")
