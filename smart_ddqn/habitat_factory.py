from __future__ import annotations

from typing import Any

from habitat_adapter import HabitatGymnasiumAdapter


def create_homerobot_env(task_config_path: str, data_path: str, index: int):
    """Top-level worker factory; top-level placement keeps it multiprocessing-pickleable."""
    try:
        from habitat.config import read_write
        from utils.habitat_utils import setup_env_config
        from curriculum_habitat.curriculum_wrapper import ObjRLNav
    except ImportError as exc:
        raise ImportError(
            "Cannot import your Habitat project modules. Add the project root to "
            "run.python_paths in config.json and verify the imports in habitat_factory.py."
        ) from exc

    config = setup_env_config(
        params_path=data_path,
        default_config_path=task_config_path,
    )
    with read_write(config):
        config.habitat.seed = int(config.habitat.seed) + int(index)

    # TODO: If ObjRLNav requires dataset, rank, or additional constructor fields,
    # pass them here. This signature is copied from the provided Habitat template.
    return ObjRLNav(config=config)


def build_habitat_env(cfg: dict[str, Any]) -> HabitatGymnasiumAdapter:
    try:
        from curriculum_habitat.curriculum_wrapper import CurriculumVectorEnv
    except ImportError as exc:
        raise ImportError(
            "Cannot import CurriculumVectorEnv. Verify run.python_paths and the import path."
        ) from exc

    num_envs = int(cfg["run"]["num_envs"])
    task_path = str(cfg["habitat"]["task_config_path"])
    data_path = str(cfg["habitat"]["data_path"])

    # TODO: Confirm that CurriculumVectorEnv automatically resets only the finished
    # worker and returns the reset observation on done. If it does not, implement
    # per-worker reset before using this adapter with replay-based learning.
    vec_env = CurriculumVectorEnv(
        make_env_fn=create_homerobot_env,
        env_fn_args=[
            (task_path, data_path, index)
            for index in range(num_envs)
        ],
    )

    if bool(cfg["habitat"].get("use_clip_wrapper", True)):
        try:
            from curriculum_habitat.helper_wrappers import CLIPWrapper
        except ImportError as exc:
            vec_env.close()
            raise ImportError("Cannot import CLIPWrapper from helper_wrappers.py") from exc

        # TODO: Confirm the key produced by CLIPWrapper. config.json currently
        # assumes that the batched feature is available as observation["clip"].
        vec_env = CLIPWrapper(vec_env, device=str(cfg["habitat"].get("clip_device", "cuda")))

    return HabitatGymnasiumAdapter(vec_env, cfg)
