import math
import os
import gym
import habitat_sim
import numpy as np

from copy import deepcopy
from collections import deque
from habitat.core.env import RLEnv
from curriculum_habitat.modified_vector_env import ModifiedVectorEnv
from curriculum_habitat.perception.graph_builder import MetricGraphBuilder, NUM_NODES
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from habitat.core.simulator import Observations
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import maps
from omegaconf import DictConfig
from habitat.core.dataset import Dataset

GRAPH_NODE_FIELDS = 6


class ObjRLNav(RLEnv):
    def __init__(self, config: "DictConfig", dataset: Optional[Dataset] = None):
        self.previous_distance_to_goal = None
        self.relevant_observation = None
        self.potential_coef = 0.5
        self.step_penalty = -0.05

        self.observation_memory = []
        self.action_memory = []
        self.debug_video_enabled = False
        self._metric_graph_builders = {}

        # IMPORTANT
        # Due to environment specifics, we don't add it. Episode is done only when agent said so
        # so there is difference between our environment (GIROL) and Habitat.
        # This comment is left here for discussion.
        # 
        # self.collision_penalty = -3

        super().__init__(config, dataset)
        self.observation_space = gym.spaces.Dict({
            "observation": self.habitat_env.observation_space["forward_rgb"],
            "absolute_goal_position": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "angle_to_goal": gym.spaces.Box(-np.pi, np.pi, (), np.float32),
            "knowledge_graph": gym.spaces.Box(
                -np.inf,
                np.inf,
                (NUM_NODES * GRAPH_NODE_FIELDS,),
                np.float32,
            ),
        })
        self.action_space = self.original_action_space = gym.spaces.Discrete(4)
        self.shortest_path_follower = ShortestPathFollower(
            self.habitat_env.sim,
            self.config.task.measurements.success.success_distance,
            return_one_hot=False,
        )

    def _sample_position_and_direction(self, mean_distance, max_angle_error, goal):
        sim, height = self.habitat_env.sim, self.habitat_env.sim.get_agent_state().position[1]
        goal_position = np.asarray(goal.position)
        goal_islands = {sim.pathfinder.get_island(v.agent_state.position) for v in goal.view_points}
        for _ in range(100):
            distance, bearing = max(np.random.normal(mean_distance, 0.1), 1.2), np.random.uniform(-np.pi, np.pi)
            position = np.array([goal_position[0] + distance * np.cos(bearing), height, goal_position[2] + distance * np.sin(bearing)])
            if sim.is_navigable(position) and sim.pathfinder.get_island(position) in goal_islands:
                angle = np.arctan2(goal_position[0] - position[0], position[2] - goal_position[2]) + np.random.uniform(-max_angle_error, max_angle_error)
                self.sampled_position, self.sampled_angle = position[[0, 2]], float((angle + np.pi) % (2 * np.pi) - np.pi)
                return self.sampled_position, self.sampled_angle
        self.sampled_position = self.sampled_angle = None
        print(f"""Failed to sample a valid position and direction after 100 attempts. 
                  Mean distance: {mean_distance}, Max angle error: {max_angle_error}""")
        return None, None

    
    def reset(self, mean_distance = None, max_angle_error = None):
        """Optionally reset at map position [x, z] and absolute yaw angle."""
        obs = super().reset()
        self.shortest_path_follower._follower = None
        self.shortest_path_follower._current_scene = None

        if mean_distance is not None or max_angle_error is not None:
            if (mean_distance is None) != (max_angle_error is None):
                raise ValueError("Both mean_distance and max_angle_error must be provided together.")

            goal = np.random.choice(self.habitat_env.current_episode.goals)
            position, angle = self._sample_position_and_direction(mean_distance, max_angle_error, goal)

            if position is not None and angle is not None:
                state = self.habitat_env.sim.get_agent_state()
                position = [float(position[0]), float(state.position[1]), float(position[1])]
                rotation = [0.0, float(-np.sin(angle / 2)), 0.0, float(np.cos(angle / 2))]
                obs = self.habitat_env.sim.get_observations_at(position, rotation, True)
                if obs is None:
                    raise ValueError("Habitat-Sim could not place the agent at this position")
                obs.update(self.habitat_env.task.sensor_suite.get_observations(
                    observations=obs,
                    episode=self.habitat_env.current_episode,
                    task=self.habitat_env.task,
                ))
                self.habitat_env.task.measurements.reset_measures(
                    episode=self.habitat_env.current_episode,
                    task=self.habitat_env.task,
                    observations=obs,
                )

        viewpoint = self.get_closest_viewpoint()
        self.viewpoint_goal = np.asarray(viewpoint.agent_state.position)

        self.previous_distance_to_goal = self.habitat_env.get_metrics()['distance_to_goal']
        obs = self.calculate_full_observation(obs)

        self.relevant_observation = deepcopy(obs)
        return obs

    def step(self, action):
        self.previous_distance_to_goal = self.habitat_env.get_metrics()['distance_to_goal']
        action_value = action
        if action == -1:
            action_value = self.calculate_next_best_action()
        action_name = self.convert_action(action_value)

        obs, reward, done, info = super().step(action_name)
        obs = self.calculate_full_observation(obs)
        info['executed_action'] = {'value': action_value, 'name': action_name['action']}
        self.relevant_observation = deepcopy(obs)
        if self.debug_video_enabled:
            info["_debug_video_state"] = self.get_debug_video_state(
                include_map=False
            )
        return obs, reward, done, info

    def convert_action(self, action):
        """Convert policy actions 0/1/2/3 to Habitat's explicit action format."""
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
            raise ValueError(f"Expected action 0, 1, 2, or 3, got {action!r}") from None
        return {"action": ("turn_left", "turn_right", "move_forward", "stop")[action]}

    def calculate_full_observation(self, observations: Observations):
        # Recompute this on every observation because the closest goal can
        # change as the agent moves.
        goal = self.get_closest_goal()
        goal_position = np.asarray(goal.position, dtype=np.float32)
        if goal_position.shape != (3,):
            raise ValueError("Goal position must be a 3D Habitat coordinate")

        # Habitat uses Y as the vertical axis, so the map plane is X-Z.
        goal_position_2d = goal_position[[0, 2]]
        obs_dict = {
            "observation": observations['forward_rgb'],
            "absolute_goal_position": goal_position_2d,
            "angle_to_goal": self.calculate_angle_to_goal(goal_position),
            "knowledge_graph": self.calculate_knowledge_graph(
                observations,
                goal=goal,
            )
        }
        return obs_dict

    def _current_scene_key(self):
        """Return the scene hash used by scene_graphs/<stage>/<scene>.json."""
        scene_id = getattr(self.habitat_env.current_episode, "scene_id", None)
        if scene_id is None:
            raise RuntimeError(
                "Current episode is missing scene_id; cannot build knowledge graph"
            )

        scene_name = os.path.basename(os.path.normpath(str(scene_id)))
        for suffix in (".basis.glb", ".semantic.glb", ".glb"):
            if scene_name.endswith(suffix):
                scene_name = scene_name[: -len(suffix)]
                break
        else:
            scene_name = os.path.splitext(scene_name)[0]

        scene_index, separator, scene_hash = scene_name.partition("-")
        if separator and scene_index.isdigit():
            return scene_hash
        return scene_name

    def _metric_graph_builder(self):
        scene = self._current_scene_key()
        if scene not in self._metric_graph_builders:
            self._metric_graph_builders[scene] = MetricGraphBuilder(scene)
        return self._metric_graph_builders[scene]

    def calculate_knowledge_graph(
        self,
        observations: Observations,
        goal=None,
    ):
        if goal is None:
            goal = self.get_closest_goal()

        target_category = getattr(
            self.habitat_env.current_episode,
            "object_category",
            None,
        )
        if target_category is None:
            raise RuntimeError(
                "Current episode is missing object_category; cannot build "
                "knowledge graph"
            )

        knowledge_graph = self._metric_graph_builder().build_flat(
            target_category,
            goal.position,
        )
        expected_shape = (NUM_NODES * GRAPH_NODE_FIELDS,)
        if knowledge_graph.shape != expected_shape:
            raise RuntimeError(
                "MetricGraphBuilder returned graph with shape "
                f"{knowledge_graph.shape}; expected {expected_shape}"
            )
        return knowledge_graph.astype(np.float32, copy=False)

    def get_reward(self, observations: Observations):
        # IMPORTANT 
        # Distance is geodesic rather than euclidean (Divergence with GIROL)
        current_distance_to_goal = self.habitat_env.get_metrics()['distance_to_goal']

        done_reward = 0.
        if self.habitat_env.task.is_stop_called:
            done_reward = 5.0 if self.habitat_env.get_metrics()["success"] else -1.0

        potential_based_reward = self.potential_coef * (self.previous_distance_to_goal - current_distance_to_goal)
        return potential_based_reward + self.step_penalty + done_reward

    def get_reward_range(self):
        return -np.inf, np.inf

    def get_done(self, observations: Observations):
        return self.habitat_env.episode_over
    
    def get_info(self, observations: Observations):
        info = dict(self.habitat_env.get_metrics())
        info['truncated'] = self.habitat_env._past_limit()
        info['terminated'] = self.habitat_env.task.is_stop_called
        return info

    def get_debug_video_state(
        self, include_map=True, map_resolution=512
    ):
        """Return serializable state needed to render a debug-video frame."""
        # Calling this method from DebugVideoWrapper opts the worker into
        # returning exact post-action poses in subsequent step infos.
        self.debug_video_enabled = True
        sim = self.habitat_env.sim
        agent_position_3d = np.asarray(
            sim.get_agent_state().position, dtype=np.float32
        )
        if self.relevant_observation is None:
            goal_position_3d = np.asarray(
                self.get_closest_goal().position, dtype=np.float32
            )
            goal_position = goal_position_3d[[0, 2]]
            angle_error = self.calculate_angle_to_goal(goal_position_3d)
        else:
            goal_position = np.asarray(
                self.relevant_observation["absolute_goal_position"],
                dtype=np.float32,
            )
            angle_error = np.float32(
                self.relevant_observation["angle_to_goal"]
            )

        if include_map:
            top_down_map = maps.get_topdown_map_from_sim(
                sim, map_resolution=int(map_resolution), draw_border=True
            )
            top_down_map = maps.colorize_topdown_map(top_down_map)
        else:
            top_down_map = None
        lower_bound, upper_bound = sim.pathfinder.get_bounds()
        target_object = getattr(
            self.habitat_env.current_episode,
            "object_category",
            None,
        )
        return {
            "top_down_map": top_down_map,
            "map_lower_bound": np.asarray(lower_bound, dtype=np.float32),
            "map_upper_bound": np.asarray(upper_bound, dtype=np.float32),
            "agent_position": agent_position_3d[[0, 2]],
            "goal_position": goal_position,
            "angle_error": angle_error,
            "target_object": (
                target_object if target_object is not None else "unknown"
            ),
        }
    
    def get_closest_viewpoint(self):
        episode = self.habitat_env.current_episode
        robot_position = self.habitat_env.sim.get_agent_state().position

        candidates = []

        for goal in episode.goals:
            for viewpoint in goal.view_points:
                position = viewpoint.agent_state.position
                distance = self.habitat_env.sim.geodesic_distance(
                    robot_position,
                    position
                )
                if np.isfinite(distance):
                    candidates.append((float(distance), goal, viewpoint))
        
        if not candidates:
            raise RuntimeError(
            "No reachable target viewpoints in the current ObjectNav episode"
            )
        
        distance, goal, viewpoint = min(
            candidates,
            key=lambda candidate: candidate[0],
        )

        return viewpoint

    def get_closest_goal(self):
        """Return the goal owning the nearest reachable viewpoint."""
        sim = self.habitat_env.sim
        goals, positions = zip(*(
            (goal, viewpoint.agent_state.position)
            for goal in self.habitat_env.current_episode.goals
            for viewpoint in goal.view_points
        ))
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = sim.get_agent_state().position
        path.requested_ends = np.asarray(positions, dtype=np.float32)
        if not sim.pathfinder.find_path(path):
            raise RuntimeError(
                "No reachable target goals in the current ObjectNav episode"
            )
        return goals[path.closest_end_point_index]
    
    def calculate_angle_to_goal(self, closest_goal_position):
        state = self.habitat_env.sim.get_agent_state()
        direction = np.asarray(closest_goal_position) - state.position
        forward = quaternion_rotate_vector(state.rotation, np.array([0, 0, -1]))
        angle = np.arctan2(direction[0], -direction[2]) - np.arctan2(forward[0], -forward[2])
        return np.float32((angle + np.pi) % (2 * np.pi) - np.pi)
    
    def calculate_next_best_action(self):
        self.shortest_path_follower._build_follower()
        try:
            action = self.shortest_path_follower._follower.next_action_along(
                self.viewpoint_goal
            )
        except habitat_sim.errors.GreedyFollowerError:
            return 3
        return (3, 2, 0, 1)[action]

class CurriculumVectorEnv(ModifiedVectorEnv):
    def __init__(
        self,
        make_env_fn: Callable[..., gym.Env],
        env_fn_args: Sequence[Tuple],
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
        stage_zero_experience: int = 100_000,
        maximal_expert_fraction: float = 0.6,
        min_coef_of_controlled_episodes: float = 1.8,
        radius_stages: int = 3,
        angle_stages: int = 7,
        difficulty_increase_sr: float = 0.85,
        difficulty_decrease_sr: float = 0.7,
        success_episode_threshold: int = 512,
        fail_episode_threshold: int = 15120
    ) -> None:
        super().__init__(
            make_env_fn=make_env_fn,
            env_fn_args=env_fn_args,
            auto_reset_done=False,
            multiprocessing_start_method=multiprocessing_start_method,
            workers_ignore_signals=workers_ignore_signals,
        )
        self.observation_space = gym.vector.utils.batch_space(self.observation_spaces[0], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.action_spaces[0], self.num_envs)
        self.stage = 0  # Phase (0=warm, 1=main, 2=final)
        self.start_mean_radius, self.start_angle_error = 3, 0
        self.max_mean_radius, self.max_angle_error = 6., math.pi
        self.radius_increment = (self.max_mean_radius - self.start_mean_radius) / radius_stages
        self.angle_increment = self.max_angle_error / angle_stages
        
        self.difficulty_increase_sr, self.difficulty_decrease_sr = difficulty_increase_sr, difficulty_decrease_sr
        self.success_episode_threshold, self.fail_episode_threshold = success_episode_threshold, fail_episode_threshold

        self.cur_mean_radius, self.cur_angle_error = self.start_mean_radius, self.start_angle_error  # Mean goal-distance given to spawn, max heading error
        self.cur_success_rate = 0.0  # Policy-only rolling SR (%)
        self.sr_stack_full = False  # Whether SR window is populated
        self.stage = 0
        
        self.success_ep_num = 0  # Consecutive high-SR episodes
        self.failed_ep_num = 0  # Consecutive low-SR episodes

        self.stage_zero_experience_threshold = stage_zero_experience
        self.maximal_expert_fraction = maximal_expert_fraction
        self.minimal_number_of_agent_episodes = int(self.num_envs * min_coef_of_controlled_episodes)
        self.generated_transitions = 0

        max_queue_size = 10 * self.num_envs
        self.success_queues = {'agent': deque(maxlen = max_queue_size),
                               'controller': deque(maxlen = max_queue_size)}
        self.ammount_of_resets = [0 for _ in range(self.num_envs)]
        self.success_rates = {'agent': 0., 'controller': 0.}
        self.expert_envs = set([i for i in range(int(self.num_envs * self.maximal_expert_fraction))])

        # Variables for testing
        self.has_difficulty_changed = False

    def _update_success_queues(self, dones, infos):
        for idx, done in enumerate(dones):
            if done:
                if idx in self.expert_envs:
                    self.success_queues['controller'].append(infos[idx]['success'])
                else:
                    self.success_queues['agent'].append(infos[idx]['success'])

    def _calculate_success_rates(self):
        for key, queue in self.success_queues.items():
            if len(queue) > 0:
                self.success_rates[key] = sum(queue) / len(queue)
            else:
                self.success_rates[key] = 0.0

    def get_logging_stats(self, rewards=None):
        reset_per_env = {'Resets per env{}'.format(i): self.ammount_of_resets[i] for i in range(self.num_envs)}
        return {
            'controller_success_rate': self.success_rates['controller'],
            'policy_success_rate': self.success_rates['agent'],
            'stage': self.stage,
            'current_mean_radius': self.cur_mean_radius,
            'current_angle_error': self.cur_angle_error,
            'fraction_of_controlled_environments': len(self.expert_envs) / self.num_envs,
            **reset_per_env,
        }

    def _update_controller_envs(self, dones):
        if len(self.success_queues['agent']) >= self.minimal_number_of_agent_episodes:
            control_percentage = max(0.10, min(0.60, 0.9 - 0.8 * (self.success_rates['agent'])))
            target_count = max(1, int(self.num_envs * control_percentage))

            for i, done in enumerate(dones):
                if done:
                    if len(self.expert_envs) > target_count and (i in self.expert_envs):
                        self.expert_envs.remove(i)
                    elif len(self.expert_envs) < target_count and (i not in self.expert_envs):
                        self.expert_envs.add(i)

    def change_difficulty_if_needed(self):
        if self.generated_transitions < self.stage_zero_experience_threshold:
            self.stage = 0
            return 
        
        elif (len(self.success_queues['agent']) > self.minimal_number_of_agent_episodes) and self.stage < 2:
            if self.stage != 1:
                self._update_sr_stack()
                self.has_difficulty_changed = True
                self.stage = 1

                return

            if self.success_rates['agent'] > self.difficulty_increase_sr:
                self.success_ep_num += 1
                self.failed_ep_num = 0

            elif self.success_rates['agent'] < self.difficulty_decrease_sr:
                self.failed_ep_num += 1
            
            if self.success_ep_num > self.success_episode_threshold:
                if self.max_mean_radius - self.cur_mean_radius > 1e-3:
                    self.has_difficulty_changed = True
                    self._increase_difficulty()
            
            elif self.failed_ep_num > self.fail_episode_threshold:
                self.has_difficulty_changed = True
                self._decrease_difficulty()
        
        if (self.max_mean_radius - self.cur_mean_radius < 1e-3) and self.stage < 2:
            self.stage = 2
            self.has_difficulty_changed = True
            self._update_sr_stack()

    def check_if_difficulty_has_recently_changed(self):
        if self.has_difficulty_changed:
            self.has_difficulty_changed = False
            return True
        return False

    def _increase_difficulty(self):
        self.success_ep_num, self.failed_ep_num = 0, 0
        self.cur_angle_error += self.angle_increment

        # Когда углы исчерпаны → переходим на радиус
        if self.cur_angle_error > self.max_angle_error:
            self.cur_angle_error = 0
            self.cur_mean_radius = min(8.0, self.cur_mean_radius + self.radius_increment)
        print(f"""[UP ↑] SR={100 * self.success_rates['agent']:.0f}% → r={self.cur_mean_radius:.1f}m, 
              a={self.cur_angle_error:.2f}rad""")
        self._update_sr_stack()

    def _decrease_difficulty(self):
        self.success_ep_num, self.failed_ep_num = 0, 0
        if self.cur_angle_error < 1e-3:
            if self.cur_mean_radius - self.start_mean_radius - 0.5 < 1e-3:
                self.cur_mean_radius = self.start_mean_radius
            else:
                self.cur_mean_radius -= 0.5
            self.cur_mean_radius = max(self.start_mean_radius, self.cur_mean_radius)
        self.cur_angle_error = 0
        print(f"""[DOWN ↓] SR={100 * self.success_rates['agent']:.0f}% → r={self.cur_mean_radius:.1f}m, 
              a={self.cur_angle_error:.2f}rad""")
        self._update_sr_stack()

    def _update_sr_stack(self):
        self._step_update_counter = 0
        self.success_queues['agent'].clear(), self.success_queues['controller'].clear()
        self.success_rates['agent'], self.success_rates['controller'] = 0.0, 0.0

    def substitute_action(self, actions):
        actions = deepcopy(actions)
        for i, action in enumerate(actions):
            if i in self.expert_envs:
                actions[i] = -1

        return actions

    def step(self, actions):
        #if not np.all(np.logical_and(actions >= 0, actions < 4)):
        #    raise ValueError(f"Expected actions to be 0, 1, 2 or 3, got {actions!r}")
        
        actions = self.substitute_action(actions)
        result = super().step(actions)

        self.generated_transitions += self.num_envs
        if (self.generated_transitions // self.num_envs) % 200 == 0:
            print(f"[LOG] env stat: {self.get_logging_stats()}")
        obs = {key: np.stack([result[i][0][key] for i in range(self.num_envs)], axis=0) \
                    for key in result[0][0].keys()}
        rewards = np.stack([result[i][1] for i in range(self.num_envs)], axis=0)
        dones = np.stack([result[i][2] for i in range(self.num_envs)], axis=0)
        infos = [result[i][3] for i in range(self.num_envs)]

        self._update_success_queues(dones, infos)
        self._calculate_success_rates()
        self._update_controller_envs(dones)
        if np.any(dones):
            self.change_difficulty_if_needed()

        for i, done in enumerate(dones):
            if done:
                infos[i]['done_observation'] = {key: deepcopy(obs[key][i]) for key in obs.keys()}
                mean_distance = self.cur_mean_radius if self.stage == 1 else None
                max_angle_error = self.cur_angle_error if self.stage == 1 else None
                reset_result = self.reset_at(i, mean_distance = mean_distance, 
                                                max_angle_error = max_angle_error)[0]
                self.ammount_of_resets[i] += 1
                for key in obs.keys():
                    obs[key][i] = deepcopy(reset_result[key])
            else:
                infos[i]['done_observation'] = np.full(obs['observation'][i].shape, np.nan, dtype=obs['observation'][i].dtype)
        return obs, rewards, dones, infos
    
    def reset(self):
        result = super().reset()
        obs = {key: np.stack([result[i][key] for i in range(self.num_envs)], axis=0) \
                    for key in result[0].keys()}
        return obs


class EvalVectorEnv(ModifiedVectorEnv):
    def __init__(
        self,
        make_env_fn: Callable[..., gym.Env],
        env_fn_args: Sequence[Tuple],
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
    ):
        super().__init__(
            make_env_fn=make_env_fn,
            env_fn_args=env_fn_args,
            auto_reset_done=False,
            multiprocessing_start_method=multiprocessing_start_method,
            workers_ignore_signals=workers_ignore_signals,
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.observation_spaces[0], self.num_envs
        )
        self.action_space = gym.vector.utils.batch_space(
            self.action_spaces[0], self.num_envs
        )

    def step(self, actions):
        result = super().step(actions)
        obs = {
            key: np.stack([result[i][0][key] for i in range(self.num_envs)])
            for key in result[0][0]
        }
        rewards = np.stack([item[1] for item in result])
        dones = np.stack([item[2] for item in result])
        infos = [item[3] for item in result]

        for i, done in enumerate(dones):
            if done:
                infos[i]["done_observation"] = {
                    key: deepcopy(obs[key][i]) for key in obs
                }
                reset_obs = self.reset_at(i)[0]
                for key in obs:
                    obs[key][i] = deepcopy(reset_obs[key])
            else:
                infos[i]["done_observation"] = np.full(
                    obs["observation"][i].shape,
                    np.nan,
                    dtype=obs["observation"][i].dtype,
                )
        return obs, rewards, dones, infos

    def reset(self):
        result = super().reset()
        return {
            key: np.stack([item[key] for item in result])
            for key in result[0]
        }
