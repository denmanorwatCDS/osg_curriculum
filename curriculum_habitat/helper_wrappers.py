from pathlib import Path
import re

import cv2
import gym
import numpy as np

import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPWrapper:
    def __init__(self, env, device = "cuda"):
        self.env, self.device = env, device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        spaces = dict(env.observation_space.spaces)
        spaces["observation"] = gym.spaces.Box(-1, 1, (env.num_envs, 512), np.float32)
        self.observation_space, self.action_space = gym.spaces.Dict(spaces), env.action_space

    def _extract_images_from_obs(self, obs):
        images = [obs["observation"][i, ..., :3] for i in range(obs["observation"].shape[0])]
        return images

    def _encode(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        embeddings = torch.nn.functional.normalize(embeddings).cpu().numpy()
        return embeddings

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs['observation'] = self._encode(self._extract_images_from_obs(obs))
        return obs
    
    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)

        if np.any(done):
            done_obs = []
            for idx, d in enumerate(done):
                if d:
                    done_obs.append(info[idx]["done_observation"]['observation'])
            done_obs = self._encode(done_obs)

            next_done_obs_idx = 0
            for idx, d in enumerate(done):
                if d:
                    info[idx]["done_observation"]['observation'] = done_obs[next_done_obs_idx]
                    next_done_obs_idx += 1

        obs['observation'] = self._encode(self._extract_images_from_obs(obs))
        return (obs, reward, done, info)
    
    def __getattr__(self, name): 
        return getattr(self.env, name)


class MemoryWrapper:
    def __init__(self, env):
        self.env, n = env, env.num_envs
        self.embeddings, self.actions = np.zeros((n, 13, 512), np.float32), np.zeros((n, 13, 2), np.float32)
        self.initialized = np.zeros(n, bool)
        spaces = dict(env.observation_space.spaces)
        spaces["memory"] = gym.spaces.Box(-np.inf, np.inf, (n, 2056), np.float32)
        self.observation_space, self.action_space = gym.spaces.Dict(spaces), env.action_space

    def _update(self, embedding, action):
        cold = ~self.initialized
        self.embeddings[cold], self.actions[cold], self.initialized[cold] = embedding[cold, None], 0, True
        self.embeddings, self.actions = np.roll(self.embeddings, 1, 1), np.roll(self.actions, 1, 1)
        self.embeddings[:, 0], self.actions[:, 0] = embedding, action

    def _memory(self):
        actions = np.stack([self.actions[:, i:min(i + 4, 13)].sum(1) for i in (0, 4, 8, 12)], 1)
        return np.concatenate((self.embeddings[:, (0, 4, 8, 12)], actions), 2).reshape(self.env.num_envs, -1)

    def _observe(self, obs, action):
        self._update(obs["observation"], action)
        obs["memory"] = self._memory()
        return obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.embeddings.fill(0); self.actions.fill(0); self.initialized.fill(False)
        return self._observe(obs, np.zeros((self.env.num_envs, 2), np.float32))

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        executed = np.asarray([info["executed_action"]["action"] for info in infos])
        action = np.stack((executed == "move_forward", (executed == "turn_left").astype(float) - (executed == "turn_right")), 1).astype(np.float32)
        self._update(obs["observation"], action)
        self.embeddings[dones], self.actions[dones], self.initialized[dones] = obs["observation"][dones, None], 0, True
        obs["memory"] = self._memory()
        return obs, rewards, dones, infos

    def __getattr__(self, name):
        return getattr(self.env, name)


class DebugVideoWrapper:
    """Record one side-by-side debug video for every vector-env episode.

    This wrapper must receive RGB observations, so it should be placed directly
    around ``CurriculumVectorEnv`` and inside wrappers (such as ``CLIPWrapper``)
    that replace the RGB observation with an embedding.
    """

    DEBUG_STATE_KEY = "_debug_video_state"

    def __init__(
        self,
        env,
        output_dir="debug_videos",
        fps=10,
        panel_size=(640, 480),
        map_resolution=512,
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.panel_width, self.panel_height = map(int, panel_size)
        self.map_resolution = int(map_resolution)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.panel_width <= 0 or self.panel_height <= 0:
            raise ValueError("panel_size must contain positive (width, height)")
        if self.map_resolution <= 0:
            raise ValueError("map_resolution must be positive")

        # H.264 encoders commonly require even frame dimensions.
        self.footer_height = 156
        self.frame_width = 2 * self.panel_width
        self.frame_height = self.panel_height + self.footer_height
        if self.frame_width % 2:
            self.frame_width += 1
        if self.frame_height % 2:
            self.frame_height += 1

        self._writers = [None] * self.num_envs
        self._base_maps = [None] * self.num_envs
        self._map_bounds = [None] * self.num_envs
        self._episode_metadata = [None] * self.num_envs
        self._episode_returns = [0.0] * self.num_envs
        self._last_rewards = [None] * self.num_envs
        self._episode_statuses = [None] * self.num_envs
        self._episode_indices = [
            self._last_existing_episode_index(env_index)
            for env_index in range(self.num_envs)
        ]

    def _last_existing_episode_index(self, env_index):
        pattern = re.compile(
            rf"^env_{env_index:03d}_episode_(\d+)\.mp4$"
        )
        indices = []
        for path in self.output_dir.glob(
            f"env_{env_index:03d}_episode_*.mp4"
        ):
            match = pattern.match(path.name)
            if match is not None:
                indices.append(int(match.group(1)))
        return max(indices, default=0)

    def _video_path(self, env_index):
        return self.output_dir / (
            f"env_{env_index:03d}_episode_"
            f"{self._episode_indices[env_index]:06d}.mp4"
        )

    def _open_episode(self, env_index, state):
        self._release_writer(env_index)
        self._episode_indices[env_index] += 1
        path = self._video_path(env_index)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open debug video writer for {path}")

        top_down_map = state.get("top_down_map")
        if top_down_map is None:
            writer.release()
            raise RuntimeError(
                "get_debug_video_state() did not return a top_down_map"
            )
        self._writers[env_index] = writer
        self._base_maps[env_index] = self._as_rgb(top_down_map, "top_down_map")
        lower_bound = np.asarray(state.get("map_lower_bound"), dtype=float)
        upper_bound = np.asarray(state.get("map_upper_bound"), dtype=float)
        if lower_bound.shape != (3,) or upper_bound.shape != (3,):
            writer.release()
            self._writers[env_index] = None
            self._base_maps[env_index] = None
            raise RuntimeError(
                "get_debug_video_state() did not return valid map bounds"
            )
        self._map_bounds[env_index] = (lower_bound, upper_bound)
        self._episode_metadata[env_index] = {
            "expert_controlled": env_index in self.env.expert_envs,
            "target_object": str(state.get("target_object", "unknown")),
        }
        self._episode_returns[env_index] = 0.0
        self._last_rewards[env_index] = None
        self._episode_statuses[env_index] = "running"

    def _release_writer(self, env_index):
        writer = self._writers[env_index]
        if writer is not None:
            writer.release()
            self._writers[env_index] = None
        self._base_maps[env_index] = None
        self._map_bounds[env_index] = None
        self._episode_metadata[env_index] = None
        self._episode_returns[env_index] = 0.0
        self._last_rewards[env_index] = None
        self._episode_statuses[env_index] = None

    @staticmethod
    def _as_rgb(image, name):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                f"{name} must be an HxWx3 RGB image, got shape {image.shape}"
            )
        image = image[..., :3]
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size and np.nanmax(image) <= 1.0:
                image = image * 255.0
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _fit_panel(self, image):
        """Letterbox an RGB image without changing its aspect ratio."""
        height, width = image.shape[:2]
        scale = min(self.panel_width / width, self.panel_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        panel = np.zeros(
            (self.panel_height, self.panel_width, 3), dtype=np.uint8
        )
        x = (self.panel_width - resized_width) // 2
        y = (self.panel_height - resized_height) // 2
        panel[y:y + resized_height, x:x + resized_width] = resized
        return panel

    def _compose_frame(self, observation, state, env_index):
        fpv = self._fit_panel(self._as_rgb(observation, "observation"))
        top_down = self._base_maps[env_index]
        if top_down is None:
            raise RuntimeError(f"No top-down map cached for env {env_index}")
        top_down = top_down.copy()

        agent_coord = self._world_to_map_coord(
            state["agent_position"], env_index
        )
        goal_coord = self._world_to_map_coord(
            state["goal_position"], env_index
        )
        dot_radius = max(4, min(top_down.shape[:2]) // 80)
        # Habitat coordinates are (row, column); OpenCV expects (x, y).
        cv2.circle(
            top_down,
            (goal_coord[1], goal_coord[0]),
            dot_radius,
            (255, 0, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            top_down,
            (agent_coord[1], agent_coord[0]),
            dot_radius,
            (0, 255, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        top_down = self._fit_panel(top_down)

        frame = np.zeros(
            (self.frame_height, self.frame_width, 3), dtype=np.uint8
        )
        frame[:self.panel_height, :self.panel_width] = top_down
        frame[:self.panel_height, self.panel_width:2 * self.panel_width] = fpv

        agent_position = np.asarray(state["agent_position"], dtype=float)
        goal_position = np.asarray(state["goal_position"], dtype=float)
        position_error = float(np.linalg.norm(goal_position - agent_position))
        angle_error = float(state["angle_error"])
        metadata = self._episode_metadata[env_index]
        if metadata is None:
            raise RuntimeError(
                f"No curriculum metadata cached for env {env_index}"
            )
        stage = int(self.env.stage)
        stage_name = {
            0: "warm-up",
            1: "main",
            2: "final",
        }.get(stage, "unknown")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            f"Goal position: x={goal_position[0]:.2f}, z={goal_position[1]:.2f}",
            (16, self.panel_height + 31),
            font,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Position error: {position_error:.2f} m",
            (16, self.panel_height + 65),
            font,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"Angle error: {angle_error:.3f} rad "
                f"({np.degrees(angle_error):.1f} deg)"
            ),
            (self.panel_width + 16, self.panel_height + 48),
            font,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                "Expert controlled: "
                f"{'yes' if metadata['expert_controlled'] else 'no'} | "
                f"Curriculum stage: {stage} ({stage_name})"
            ),
            (self.panel_width + 16, self.panel_height + 76),
            font,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        last_reward = self._last_rewards[env_index]
        reward_text = (
            "N/A" if last_reward is None else f"{last_reward:.4f}"
        )
        cv2.putText(
            frame,
            (
                f"Step reward: {reward_text} | "
                f"Episode return: {self._episode_returns[env_index]:.4f}"
            ),
            (16, self.panel_height + 108),
            font,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Target object: {metadata['target_object']}",
            (16, self.panel_height + 142),
            font,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Episode status: {self._episode_statuses[env_index]}",
            (self.panel_width + 16, self.panel_height + 142),
            font,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _world_to_map_coord(self, position, env_index):
        position = np.asarray(position, dtype=float)
        if position.shape != (2,):
            raise ValueError(
                "Debug positions must contain Habitat (x, z) coordinates"
            )
        top_down = self._base_maps[env_index]
        bounds = self._map_bounds[env_index]
        if top_down is None or bounds is None:
            raise RuntimeError(f"No map metadata cached for env {env_index}")
        lower_bound, upper_bound = bounds
        rows, columns = top_down.shape[:2]
        z_span = abs(upper_bound[2] - lower_bound[2])
        x_span = abs(upper_bound[0] - lower_bound[0])
        if z_span == 0 or x_span == 0:
            raise ValueError("Simulator map bounds have a zero-sized axis")
        row = int((position[1] - lower_bound[2]) / (z_span / rows))
        column = int((position[0] - lower_bound[0]) / (x_span / columns))
        return (
            int(np.clip(row, 0, rows - 1)),
            int(np.clip(column, 0, columns - 1)),
        )

    def _write_frame(self, env_index, observation, state):
        writer = self._writers[env_index]
        if writer is None:
            raise RuntimeError(f"No active debug video for env {env_index}")
        # VideoWriter consumes BGR while all environment images are RGB.
        writer.write(
            cv2.cvtColor(
                self._compose_frame(observation, state, env_index),
                cv2.COLOR_RGB2BGR,
            )
        )

    def _query_state(self, env_index):
        try:
            return self.env.call_at(
                env_index,
                "get_debug_video_state",
                {
                    "include_map": True,
                    "map_resolution": self.map_resolution,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                "DebugVideoWrapper requires vector workers to expose "
                "get_debug_video_state(). Wrap CurriculumVectorEnv directly."
            ) from exc

    def reset(self, **kwargs):
        for env_index in range(self.num_envs):
            self._release_writer(env_index)
        observations = self.env.reset(**kwargs)
        for env_index in range(self.num_envs):
            state = self._query_state(env_index)
            self._open_episode(env_index, state)
            self._write_frame(
                env_index, observations["observation"][env_index], state
            )
        return observations

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        for env_index in range(self.num_envs):
            reward = float(np.asarray(rewards[env_index]).item())
            self._last_rewards[env_index] = reward
            self._episode_returns[env_index] += reward
            if dones[env_index]:
                success = infos[env_index].get("success")
                if success is None:
                    self._episode_statuses[env_index] = (
                        "terminated (success unknown)"
                    )
                elif bool(success):
                    self._episode_statuses[env_index] = (
                        "terminated successfully"
                    )
                else:
                    self._episode_statuses[env_index] = (
                        "terminated unsuccessfully"
                    )
            state = infos[env_index].get(self.DEBUG_STATE_KEY)
            if state is None:
                if dones[env_index]:
                    raise RuntimeError(
                        "Terminal info is missing the debug state needed to "
                        "record the final pre-reset frame"
                    )
                state = self._query_state(env_index)

            frame_observation = (
                infos[env_index]["done_observation"]["observation"]
                if dones[env_index]
                else observations["observation"][env_index]
            )
            self._write_frame(env_index, frame_observation, state)

            if dones[env_index]:
                self._release_writer(env_index)
                reset_state = self._query_state(env_index)
                self._open_episode(env_index, reset_state)
                self._write_frame(
                    env_index,
                    observations["observation"][env_index],
                    reset_state,
                )
        return observations, rewards, dones, infos

    def close(self):
        for env_index in range(self.num_envs):
            self._release_writer(env_index)
        return self.env.close()

    def __del__(self):
        # Do not close the underlying env from a finalizer; only flush files
        # owned by this wrapper.
        writers = getattr(self, "_writers", ())
        for writer in writers:
            if writer is not None:
                writer.release()

    def __getattr__(self, name):
        return getattr(self.env, name)
