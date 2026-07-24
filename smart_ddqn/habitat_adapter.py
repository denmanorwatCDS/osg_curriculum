from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch


class HabitatGymnasiumAdapter(gym.Env):
    """Convert a Habitat-style vector environment to the Gymnasium contract used by skrl.

    Output observation (batched):
        {
            "img": [num_envs, img_dim],
            "goal": [num_envs, goal_dim],
            "graph": [num_envs, graph_dim],
            "teacher_orientation": [num_envs, 1],
        }

    The policy deliberately ignores ``teacher_orientation``. It remains in the
    flattened state stored by skrl's replay memory and is consumed only by the
    auxiliary trainer.
    """

    metadata = {"render_modes": []}

    def __init__(self, habitat_env: Any, cfg: dict[str, Any]):
        super().__init__()
        self.env = habitat_env
        self.cfg = cfg
        self.num_envs = int(cfg["run"]["num_envs"])
        self.device = torch.device(cfg["run"].get("device", "cuda"))

        reported_num_envs = getattr(habitat_env, "num_envs", None)
        if reported_num_envs is not None and int(reported_num_envs) != self.num_envs:
            raise ValueError(
                f"Configured num_envs={self.num_envs}, but Habitat reports {reported_num_envs}"
            )

        dims = cfg["observation"]["dims"]
        self.single_observation_space = gym.spaces.Dict(
            {
                "img": gym.spaces.Box(-np.inf, np.inf, (int(dims["img"]),), np.float32),
                "goal": gym.spaces.Box(-np.inf, np.inf, (int(dims["goal"]),), np.float32),
                "graph": gym.spaces.Box(-np.inf, np.inf, (int(dims["graph"]),), np.float32),
                "teacher_orientation": gym.spaces.Box(
                    -np.pi, np.pi, (int(dims["teacher_orientation"]),), np.float32
                ),
            }
        )
        # skrl's Gymnasium wrapper reads observation_space/action_space as the
        # single-agent spaces and uses num_envs for the batch dimension.
        self.observation_space = self.single_observation_space

        self.single_action_space = gym.spaces.Discrete(int(cfg["action"]["num_actions"]))
        self.action_space = self.single_action_space

        self._source_keys = dict(cfg["observation"]["source_keys"])
        self._dims = {key: int(value) for key, value in dims.items()}
        self._teacher_required = bool(cfg["observation"].get("teacher_required", True))
        self._actions_as_list = bool(cfg["habitat"].get("actions_as_list", True))
        self._action_id_map = cfg["habitat"].get("action_id_map")
        self._truncation_info_key = str(
            cfg["habitat"].get("truncation_info_key", "TimeLimit.truncated")
        )

        if self._action_id_map is not None:
            if len(self._action_id_map) != self.action_space.n:
                raise ValueError("habitat.action_id_map length must equal action.num_actions")
            self._action_id_map = [int(value) for value in self._action_id_map]

    @property
    def unwrapped(self):
        return self.env

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Habitat seeding is normally configured per worker in habitat_factory.py.
        # TODO: forward seed/options here if your CurriculumVectorEnv supports them.
        result = self.env.reset()
        raw_obs, info = self._split_reset_result(result)
        return self._convert_observation(raw_obs), self._normalize_info(info)

    def step(self, actions):
        habitat_actions = self._convert_actions(actions)
        result = self.env.step(habitat_actions)
        raw_obs, rewards, terminated, truncated, info = self._split_step_result(result)

        observations = self._convert_observation(raw_obs)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(self.num_envs)
        terminated = np.asarray(terminated, dtype=np.bool_).reshape(self.num_envs)
        truncated = np.asarray(truncated, dtype=np.bool_).reshape(self.num_envs)
        return observations, rewards, terminated, truncated, self._normalize_info(info)

    def close(self):
        self.env.close()

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def get_metrics(self):
        if hasattr(self.env, "get_metrics"):
            return self.env.get_metrics()
        return {}

    def _convert_actions(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(-1)
        if actions.size != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} discrete actions, got shape {tuple(actions.shape)}"
            )

        action_ids = [int(action) for action in actions]
        for action in action_ids:
            if not self.action_space.contains(action):
                raise ValueError(f"Policy action {action} is outside Discrete({self.action_space.n})")

        if self._action_id_map is not None:
            action_ids = [self._action_id_map[action] for action in action_ids]

        # TODO: If ObjRLNav expects dictionaries such as {"action": "move_forward"},
        # replace this identity conversion with the exact Habitat action payload.
        if self._actions_as_list:
            return action_ids
        return np.asarray(action_ids, dtype=np.int64)

    @staticmethod
    def _split_reset_result(result):
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        return result, {}

    def _split_step_result(self, result):
        # Habitat VectorEnv commonly returns a list of per-environment tuples.
        if (
            isinstance(result, Sequence)
            and len(result) == self.num_envs
            and all(isinstance(item, tuple) for item in result)
        ):
            tuple_size = len(result[0])
            if not all(len(item) == tuple_size for item in result):
                raise ValueError("Habitat step returned per-env tuples with different lengths")
            columns = list(zip(*result))
            if tuple_size == 4:
                obs, rewards, dones, infos = columns
                terminated, truncated = self._split_dones(dones, infos)
                return list(obs), rewards, terminated, truncated, list(infos)
            if tuple_size == 5:
                obs, rewards, terminated, truncated, infos = columns
                return list(obs), rewards, terminated, truncated, list(infos)

        if not isinstance(result, tuple):
            raise TypeError(
                "Unsupported Habitat step result. Expected tuple or list of per-env tuples."
            )
        if len(result) == 4:
            obs, rewards, dones, infos = result
            terminated, truncated = self._split_dones(dones, infos)
            return obs, rewards, terminated, truncated, infos
        if len(result) == 5:
            return result
        raise ValueError(f"Unsupported Habitat step tuple length: {len(result)}")

    def _split_dones(self, dones, infos):
        dones = np.asarray(dones, dtype=np.bool_).reshape(self.num_envs)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        for index in range(self.num_envs):
            info = self._info_at(infos, index)
            truncated[index] = bool(self._get_dotted(info, self._truncation_info_key, False))
        terminated = np.logical_and(dones, np.logical_not(truncated))
        return terminated, truncated

    def _convert_observation(self, raw_obs) -> dict[str, np.ndarray]:
        return {
            target_key: self._extract_field(raw_obs, target_key)
            for target_key in ("img", "goal", "graph", "teacher_orientation")
        }

    def _extract_field(self, raw_obs, target_key: str) -> np.ndarray:
        source_key = self._source_keys.get(target_key)
        expected_dim = self._dims[target_key]

        if source_key is None:
            if target_key == "teacher_orientation" and self._teacher_required:
                raise KeyError("teacher_orientation source key is null while teacher_required=true")
            return np.zeros((self.num_envs, expected_dim), dtype=np.float32)

        if isinstance(raw_obs, Mapping):
            value = self._get_dotted(raw_obs, source_key)
            if value is None:
                return self._missing_field(target_key, source_key, expected_dim)
            array = self._to_numpy(value)
            return self._reshape_batched(array, target_key, source_key, expected_dim)

        if isinstance(raw_obs, Sequence) and not isinstance(raw_obs, (str, bytes, np.ndarray)):
            if len(raw_obs) != self.num_envs:
                raise ValueError(
                    f"Expected {self.num_envs} per-env observations, got {len(raw_obs)}"
                )
            rows = []
            for index, observation in enumerate(raw_obs):
                if not isinstance(observation, Mapping):
                    raise TypeError(f"Observation {index} is not a mapping: {type(observation)}")
                value = self._get_dotted(observation, source_key)
                if value is None:
                    if target_key == "teacher_orientation" and not self._teacher_required:
                        value = np.zeros(expected_dim, dtype=np.float32)
                    else:
                        raise KeyError(
                            f"Observation {index} has no source key {source_key!r} for {target_key!r}"
                        )
                row = self._to_numpy(value).astype(np.float32, copy=False).reshape(-1)
                if row.size != expected_dim:
                    raise ValueError(
                        f"Observation {index}, field {source_key!r}: expected {expected_dim} values, "
                        f"got shape {tuple(row.shape)}"
                    )
                rows.append(row)
            return np.stack(rows, axis=0)

        raise TypeError(f"Unsupported observation container: {type(raw_obs)}")

    def _missing_field(self, target_key: str, source_key: str, expected_dim: int):
        if target_key == "teacher_orientation" and not self._teacher_required:
            return np.zeros((self.num_envs, expected_dim), dtype=np.float32)
        raise KeyError(
            f"Habitat observation has no key {source_key!r} required for {target_key!r}. "
            "Update observation.source_keys in config.json."
        )

    def _reshape_batched(
        self, array: np.ndarray, target_key: str, source_key: str, expected_dim: int
    ) -> np.ndarray:
        array = array.astype(np.float32, copy=False)
        if array.ndim > 0 and array.shape[0] == self.num_envs:
            array = array.reshape(self.num_envs, -1)
        elif self.num_envs == 1:
            array = array.reshape(1, -1)
        elif array.size == self.num_envs * expected_dim:
            array = array.reshape(self.num_envs, expected_dim)
        else:
            raise ValueError(
                f"Field {source_key!r} for {target_key!r} cannot be interpreted as a batch of "
                f"{self.num_envs}: source shape={tuple(array.shape)}"
            )

        if array.shape[1] != expected_dim:
            raise ValueError(
                f"Field {source_key!r} for {target_key!r}: configured dim={expected_dim}, "
                f"actual flattened dim={array.shape[1]}. Update config.json."
            )
        return array

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _get_dotted(mapping: Any, path: str, default=None):
        value = mapping
        for part in str(path).split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    @staticmethod
    def _info_at(infos, index: int):
        if isinstance(infos, Sequence) and not isinstance(infos, (str, bytes, np.ndarray)):
            return infos[index] if index < len(infos) else {}
        if isinstance(infos, Mapping):
            result = {}
            for key, value in infos.items():
                try:
                    result[key] = value[index]
                except Exception:
                    result[key] = value
            return result
        return {}

    @staticmethod
    def _normalize_info(info):
        if isinstance(info, Mapping):
            return dict(info)
        return {"per_env": info}
