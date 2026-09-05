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


def _require_positive(mapping: dict, key: str, prefix: str) -> None:
    if key not in mapping:
        raise ValueError(f"Missing {prefix}.{key}")
    if int(mapping[key]) <= 0:
        raise ValueError(f"{prefix}.{key} must be positive")


def _validate(cfg: dict[str, Any]) -> None:
    for section in (
        "run",
        "habitat",
        "observation",
        "action",
        "model",
        "agent",
        "aux",
    ):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")

    if int(cfg["run"]["num_envs"]) <= 0:
        raise ValueError("run.num_envs must be positive")
    if not str(cfg["run"].get("device", "")).strip():
        raise ValueError("run.device must be a non-empty device string")
    if int(cfg["action"]["num_actions"]) <= 1:
        raise ValueError("action.num_actions must be greater than 1 for DDQN")

    clip_device = cfg["habitat"].get("clip_device")
    if clip_device is not None and not str(clip_device).strip():
        raise ValueError("habitat.clip_device must be a non-empty device string")
    simulator_gpu_device_id = int(
        cfg["habitat"].get("simulator_gpu_device_id", 0)
    )
    if simulator_gpu_device_id < 0:
        raise ValueError("habitat.simulator_gpu_device_id must be non-negative")

    dims = cfg["observation"]["dims"]
    for key in ("img", "goal", "emb_obj", "graph", "teacher_orientation"):
        _require_positive(dims, key, "observation.dims")

    if int(dims["teacher_orientation"]) != 1:
        raise ValueError(
            "teacher_orientation must have dimension 1 and contain yaw in radians"
        )

    model = cfg["model"]
    for key in (
        "dqn_img_hidden",
        "goal_hidden",
        "emb_obj_hidden",
        "graph_embedding_dim",
        "orientation_img_hidden",
        "orientation_hidden",
        "orientation_bins",
    ):
        _require_positive(model, key, "model")

    if not isinstance(model.get("q_hidden"), list) or not model["q_hidden"]:
        raise ValueError("model.q_hidden must be a non-empty list")
    if any(int(value) <= 0 for value in model["q_hidden"]):
        raise ValueError("Every model.q_hidden value must be positive")

    orientation_source = str(model.get("orientation_source", "pred"))
    if orientation_source not in {"gt", "pred"}:
        raise ValueError("model.orientation_source must be either 'gt' or 'pred'")

    aux = cfg["aux"]
    for key in (
        "learning_rate_graph",
        "learning_rate_orientation",
        "batch_size",
    ):
        if key not in aux:
            raise ValueError(f"Missing aux.{key}")
        if float(aux[key]) <= 0:
            raise ValueError(f"aux.{key} must be positive")
