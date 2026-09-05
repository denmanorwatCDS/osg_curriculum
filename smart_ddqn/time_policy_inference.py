"""Measure DDQN policy inference over exactly 100 completed episodes.

The script reports two different timing boundaries:

* policy time: time spent in the Q-network forward pass and argmax;
* rollout time: wall time spent collecting the episodes, including Habitat,
  CLIP preprocessing, inter-process communication, and policy inference.

Environment/model construction, checkpoint loading, and the initial reset are
intentionally excluded from both measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

print("[startup] Importing PyTorch...", flush=True)
import torch
print("[startup] PyTorch import complete.", flush=True)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TOTAL_EPISODES = 100
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config_chpt.json"
DEFAULT_TASK_CONFIG_PATH = (
    REPO_ROOT / "configs/objectnav_hm3d_v2_with_semantic.yaml"
)
DEFAULT_STAGE2_DATA_PATH = (
    REPO_ROOT / "configs/homerobot_hm3d_objectnav_debug.yaml"
)
DEFAULT_HELDOUT_DATA_PATH = (
    REPO_ROOT / "configs/homerobot_hm3d_objectnav_val.yaml"
)
DEFAULT_STAGE2_NUM_ENVS = 32
ACTION_NAMES = ("turn_left", "turn_right", "move_forward", "stop")


def _print_stage(message: str) -> None:
    print(f"[stage] {message}", flush=True)


def _print_progress(
    processed: int,
    *,
    policy_calls: int | None = None,
    active_envs: int | None = None,
    latest_spl: float | None = None,
    latest_habitat_spl: float | None = None,
    mean_spl: float | None = None,
    latest_success: bool | None = None,
    optimal_distance: float | None = None,
    traveled_distance: float | None = None,
) -> None:
    details = []
    if policy_calls is not None:
        details.append(f"policy calls: {policy_calls}")
    if active_envs is not None:
        details.append(f"active envs: {active_envs}")
    if latest_spl is not None:
        details.append(f"manual SPL: {latest_spl:.3f}")
    if latest_habitat_spl is not None:
        details.append(f"Habitat SPL: {latest_habitat_spl:.3f}")
    if mean_spl is not None:
        details.append(f"mean SPL: {mean_spl:.3f}")
    if latest_success is not None:
        details.append(f"success: {int(latest_success)}")
    if optimal_distance is not None:
        details.append(f"optimal path: {optimal_distance:.3f}m")
    if traveled_distance is not None:
        details.append(f"traveled path: {traveled_distance:.3f}m")
    suffix = f" | {' | '.join(details)}" if details else ""
    print(
        f"[progress] episodes processed: {processed}/{TOTAL_EPISODES}{suffix}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DDQN policy for exactly 100 completed Habitat episodes "
            "and measure its inference time."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("stage2", "heldout"),
        default="stage2",
        help=(
            "Evaluation setup. 'stage2' matches smart_ddqn/main.py: debug "
            "split, fixated target, 32 workers, and final epsilon. 'heldout' "
            "uses the unfiltered validation split with greedy actions."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="DDQN configuration containing matching trained checkpoints.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help=(
            "Number of parallel Habitat workers. Defaults to 32 for the "
            "stage2 profile and run.num_envs for heldout. Must be 1-100."
        ),
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=DEFAULT_TASK_CONFIG_PATH,
        help="Base Habitat task configuration.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the evaluation profile's dataset configuration.",
    )
    parser.add_argument(
        "--agent-checkpoint",
        type=Path,
        default=None,
        help="Override run.agent_checkpoint from the config.",
    )
    parser.add_argument(
        "--perception-checkpoint",
        type=Path,
        default=None,
        help="Override aux.resume_from from the config.",
    )
    parser.add_argument(
        "--checkpoint-step",
        default="latest",
        help=(
            "Matching DDQN/perception checkpoint step, or 'latest' "
            "(default). Ignored when both checkpoint override paths are given."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=(
            "Epsilon-greedy action probability. Defaults to agent.final_epsilon "
            "for stage2 and 0 for heldout."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Habitat episode step limit (default: 500).",
    )
    parser.add_argument(
        "--spl-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Maximum manual-vs-Habitat SPL difference before an episode is "
            "reported as a mismatch (default: 1e-4)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optionally write the timing result to this JSON file.",
    )
    return parser.parse_args()


def _resolve_checkpoint(value: str | Path | None, name: str) -> str:
    if value is None:
        raise ValueError(
            f"{name} is not configured. Timing a trained policy requires both "
            "the DDQN and perception checkpoints."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return str(path)


def _numbered_checkpoint_steps(directory: Path, prefix: str) -> set[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.pt$")
    steps = set()
    for path in directory.glob(f"{prefix}_*.pt"):
        match = pattern.match(path.name)
        if match:
            steps.add(int(match.group(1)))
    return steps


def _select_checkpoints(
    cfg: dict[str, Any],
    *,
    agent_override: Path | None,
    perception_override: Path | None,
    requested_step: str,
) -> tuple[str, str, int | None]:
    if (agent_override is None) != (perception_override is None):
        raise ValueError(
            "--agent-checkpoint and --perception-checkpoint must be provided "
            "together so mismatched model components cannot be evaluated."
        )

    if agent_override is not None and perception_override is not None:
        agent_path = _resolve_checkpoint(agent_override, "DDQN checkpoint")
        perception_path = _resolve_checkpoint(
            perception_override, "perception checkpoint"
        )
        agent_match = re.search(r"agent_(\d+)\.pt$", agent_path)
        perception_match = re.search(
            r"perception_(\d+)\.pt$", perception_path
        )
        if bool(agent_match) != bool(perception_match):
            raise ValueError(
                "Checkpoint overrides must both have numbered filenames or "
                "both use non-numbered filenames."
            )
        step = None
        if agent_match and perception_match:
            agent_step = int(agent_match.group(1))
            perception_step = int(perception_match.group(1))
            if agent_step != perception_step:
                raise ValueError(
                    "DDQN and perception checkpoint steps differ: "
                    f"{agent_step} != {perception_step}"
                )
            step = agent_step
        return agent_path, perception_path, step

    configured_agent = cfg["run"].get("agent_checkpoint")
    configured_perception = cfg["aux"].get("resume_from")
    if configured_agent is None or configured_perception is None:
        raise ValueError(
            "The config must identify the DDQN and perception checkpoint "
            "directories when explicit overrides are not supplied."
        )
    configured_agent_path = Path(configured_agent).expanduser()
    configured_perception_path = Path(configured_perception).expanduser()
    if not configured_agent_path.is_absolute():
        configured_agent_path = REPO_ROOT / configured_agent_path
    if not configured_perception_path.is_absolute():
        configured_perception_path = REPO_ROOT / configured_perception_path
    agent_directory = configured_agent_path.resolve().parent
    perception_directory = configured_perception_path.resolve().parent
    common_steps = (
        _numbered_checkpoint_steps(agent_directory, "agent")
        & _numbered_checkpoint_steps(perception_directory, "perception")
    )
    if not common_steps:
        raise FileNotFoundError(
            "No matching numbered DDQN/perception checkpoint pairs found in "
            f"{agent_directory} and {perception_directory}"
        )

    if requested_step == "latest":
        step = max(common_steps)
    else:
        try:
            step = int(requested_step)
        except ValueError as error:
            raise ValueError(
                "--checkpoint-step must be a positive integer or 'latest'"
            ) from error
        if step < 1:
            raise ValueError("--checkpoint-step must be positive")
        if step not in common_steps:
            available = f"{min(common_steps)}..{max(common_steps)}"
            raise FileNotFoundError(
                f"No matching checkpoint pair at step {step}. Available "
                f"matching steps span {available}."
            )

    return (
        str((agent_directory / f"agent_{step}.pt").resolve()),
        str((perception_directory / f"perception_{step}.pt").resolve()),
        step,
    )


def _episode_targets(total: int, num_envs: int) -> list[int]:
    quotient, remainder = divmod(total, num_envs)
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(num_envs)
    ]


def _create_eval_env(
    task_config_path: str,
    data_path: str,
    index: int,
    simulator_gpu_device_id: int,
    max_episode_steps: int,
    fixated_object: bool,
    shuffle_episodes: bool,
):
    _print_stage(f"Worker {index}: importing Habitat modules...")
    from habitat.config import read_write
    from habitat.datasets import make_dataset

    from curriculum_habitat.curriculum_wrapper import ObjRLNav
    from utils.habitat_utils import setup_env_config

    _print_stage(f"Worker {index}: building Habitat configuration...")
    config = setup_env_config(
        params_path=data_path,
        default_config_path=task_config_path,
    )
    with read_write(config):
        config.habitat.seed = int(config.habitat.seed) + index
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
            simulator_gpu_device_id
        )
        config.habitat.environment.max_episode_steps = max_episode_steps
        config.habitat.environment.iterator_options.shuffle = shuffle_episodes
        config.habitat.environment.iterator_options.cycle = True
    dataset = None
    if fixated_object:
        dataset = make_dataset(
            config.habitat.dataset.type, config=config.habitat.dataset
        )
        if not dataset.episodes:
            raise RuntimeError(
                "Cannot select a fixated target from an empty dataset"
            )
        target = max(
            dataset.episodes, key=lambda episode: len(episode.goals)
        ).object_category
        dataset = dataset.filter_episodes(
            lambda episode: episode.object_category == target
        )
        _print_stage(
            f"Worker {index}: fixated target={target!r}, "
            f"episodes={len(dataset.episodes)}."
        )
    _print_stage(f"Worker {index}: creating Habitat environment...")
    env = ObjRLNav(config=config, dataset=dataset)
    _print_stage(f"Worker {index}: Habitat environment ready.")
    return env


def _build_eval_env(
    cfg: dict[str, Any],
    num_envs: int,
    max_episode_steps: int,
    task_config_path: Path,
    data_path: Path,
    fixated_object: bool,
    shuffle_episodes: bool,
):
    _print_stage("Importing vector-environment and wrapper modules...")
    from skrl.envs.wrappers.torch import wrap_env

    from curriculum_habitat.curriculum_wrapper import EvalVectorEnv
    from curriculum_habitat.helper_wrappers import CLIPWrapper, ToSKRLWrapper
    _print_stage("Vector-environment and wrapper imports complete.")

    habitat_cfg = cfg["habitat"]
    clip_device = str(
        habitat_cfg.get("clip_device") or cfg["run"]["device"]
    )
    simulator_gpu_device_id = int(
        habitat_cfg.get("simulator_gpu_device_id", 0)
    )

    _print_stage(f"Starting {num_envs} Habitat evaluation workers...")
    env = EvalVectorEnv(
        make_env_fn=_create_eval_env,
        env_fn_args=[
            (
                str(task_config_path),
                str(data_path),
                index,
                simulator_gpu_device_id,
                max_episode_steps,
                fixated_object,
                shuffle_episodes,
            )
            for index in range(num_envs)
        ],
    )
    _print_stage("All Habitat evaluation workers started.")
    _print_stage(f"Loading CLIP model on {clip_device}...")
    env = CLIPWrapper(env, device=clip_device)
    _print_stage("CLIP model loaded.")
    _print_stage("Applying SKRL observation/action wrappers...")
    env = ToSKRLWrapper(env, device=str(cfg["run"]["device"]))
    env = wrap_env(env, wrapper="gymnasium")
    _print_stage("Evaluation environment is fully wrapped and ready.")
    return env


def _collect_episodes(
    env,
    agent,
    targets: list[int],
    *,
    epsilon: float,
    spl_tolerance: float,
) -> dict[str, Any]:
    """Collect the per-worker quotas while timing active policy decisions.

    A vector worker that reaches its quota still has to be stepped until all
    other workers finish. It receives STOP actions, but is excluded from the
    policy batch and all metrics. Thus policy timing and decision counts cover
    exactly the requested 100 episodes.
    """

    _print_stage("Resetting all evaluation environments...")
    if hasattr(env, "_reset_once"):
        env._reset_once = True
    states, _ = env.reset()
    _print_stage("Initial environment reset complete.")

    num_envs = env.num_envs
    completed = [0] * num_envs
    episode_returns = torch.zeros(num_envs, device=env.device)
    episode_steps = [0] * num_envs
    returns: list[float] = []
    steps: list[int] = []
    episode_spls: list[float] = []
    habitat_spls: list[float] = []
    optimal_distances: list[float] = []
    traveled_distances: list[float] = []
    successful_path_efficiencies: list[float] = []
    spl_errors: list[float] = []
    start_distance_errors: list[float] = []
    successes = 0
    policy_calls = 0
    policy_decisions = 0
    action_decisions = 0
    exploration_decisions = 0
    action_counts = [0] * len(ACTION_NAMES)
    cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    cpu_policy_seconds = 0.0

    agent.set_running_mode("eval")
    agent.set_mode("eval")

    if states.device.type == "cuda":
        torch.cuda.synchronize(states.device)
    _print_progress(0, policy_calls=0, active_envs=num_envs)
    rollout_started = time.perf_counter()
    last_heartbeat = rollout_started

    while completed != targets:
        active_indices = [
            index
            for index in range(num_envs)
            if completed[index] < targets[index]
        ]
        active_index_tensor = torch.as_tensor(
            active_indices, device=states.device, dtype=torch.long
        )
        active_states = states.index_select(0, active_index_tensor)

        with torch.inference_mode():
            if epsilon > 0:
                active_actions = agent.q_network.random_act(
                    {"states": active_states}, role="q_network"
                )[0]
                greedy_local_indices = torch.nonzero(
                    torch.rand(
                        len(active_indices), device=states.device
                    ) >= epsilon,
                    as_tuple=False,
                ).reshape(-1)
            else:
                active_actions = torch.empty(
                    (len(active_indices), 1),
                    dtype=torch.long,
                    device=states.device,
                )
                greedy_local_indices = torch.arange(
                    len(active_indices), device=states.device
                )

            greedy_count = int(greedy_local_indices.numel())
            if greedy_count:
                greedy_states = active_states.index_select(
                    0, greedy_local_indices
                )
                if states.device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    q_values = agent.q_network.act(
                        {"states": greedy_states}, role="q_network"
                    )[0]
                    greedy_actions = torch.argmax(
                        q_values, dim=1, keepdim=True
                    )
                    end_event.record()
                    cuda_events.append((start_event, end_event))
                else:
                    policy_started = time.perf_counter()
                    q_values = agent.q_network.act(
                        {"states": greedy_states}, role="q_network"
                    )[0]
                    greedy_actions = torch.argmax(
                        q_values, dim=1, keepdim=True
                    )
                    cpu_policy_seconds += (
                        time.perf_counter() - policy_started
                    )
                active_actions.index_copy_(
                    0, greedy_local_indices, greedy_actions
                )
                policy_calls += 1
                policy_decisions += greedy_count

            exploration_decisions += len(active_indices) - greedy_count

        # Inactive vector slots must still be stepped. STOP is Habitat action 3.
        actions = torch.full(
            (num_envs, 1),
            3,
            dtype=active_actions.dtype,
            device=states.device,
        )
        actions.index_copy_(0, active_index_tensor, active_actions)
        states, rewards, terminated, truncated, infos = env.step(actions)

        action_decisions += len(active_indices)
        done = (terminated | truncated).reshape(-1)

        for index in active_indices:
            action_value = int(infos[index]["executed_action"]["value"])
            if not 0 <= action_value < len(action_counts):
                raise RuntimeError(
                    f"Environment executed invalid action {action_value}"
                )
            action_counts[action_value] += 1
            episode_returns[index] += rewards[index].reshape(-1)[0]
            episode_steps[index] += 1
            if bool(done[index].item()):
                required_spl_keys = (
                    "spl",
                    "manual_spl",
                    "spl_optimal_distance",
                    "spl_traveled_distance",
                    "spl_habitat_start_distance",
                    "spl_start_distance_error",
                    "spl_error",
                )
                missing_spl_keys = [
                    key
                    for key in required_spl_keys
                    if key not in infos[index]
                ]
                if missing_spl_keys:
                    raise RuntimeError(
                        "Environment did not return SPL audit fields: "
                        f"{missing_spl_keys}"
                    )
                habitat_spl = float(infos[index]["spl"])
                episode_spl = float(infos[index]["manual_spl"])
                optimal_distance = float(
                    infos[index]["spl_optimal_distance"]
                )
                traveled_distance = float(
                    infos[index]["spl_traveled_distance"]
                )
                habitat_start_distance = float(
                    infos[index]["spl_habitat_start_distance"]
                )
                start_distance_error = float(
                    infos[index]["spl_start_distance_error"]
                )
                spl_error = float(infos[index]["spl_error"])
                numeric_spl_values = {
                    "manual SPL": episode_spl,
                    "Habitat SPL": habitat_spl,
                    "optimal distance": optimal_distance,
                    "traveled distance": traveled_distance,
                    "Habitat start distance": habitat_start_distance,
                    "start distance error": start_distance_error,
                    "SPL error": spl_error,
                }
                non_finite = {
                    name: value
                    for name, value in numeric_spl_values.items()
                    if not math.isfinite(value)
                }
                if non_finite:
                    raise RuntimeError(
                        f"Non-finite SPL audit values: {non_finite}"
                    )
                if (
                    not 0.0 <= episode_spl <= 1.0 + 1e-6
                    or not 0.0 <= habitat_spl <= 1.0 + 1e-6
                ):
                    raise RuntimeError(
                        "SPL outside the expected [0, 1] range: "
                        f"manual={episode_spl}, Habitat={habitat_spl}"
                    )
                success = bool(infos[index].get("success", False))
                returns.append(float(episode_returns[index].item()))
                steps.append(episode_steps[index])
                episode_spls.append(episode_spl)
                habitat_spls.append(habitat_spl)
                optimal_distances.append(optimal_distance)
                traveled_distances.append(traveled_distance)
                spl_errors.append(spl_error)
                start_distance_errors.append(start_distance_error)
                if success:
                    successful_path_efficiencies.append(episode_spl)
                successes += int(success)
                completed[index] += 1
                episode_returns[index] = 0
                episode_steps[index] = 0
                _print_progress(
                    len(returns),
                    policy_calls=policy_calls,
                    active_envs=sum(
                        completed[i] < targets[i] for i in range(num_envs)
                    ),
                    latest_spl=episode_spl,
                    latest_habitat_spl=habitat_spl,
                    mean_spl=sum(episode_spls) / len(episode_spls),
                    latest_success=success,
                    optimal_distance=optimal_distance,
                    traveled_distance=traveled_distance,
                )
                if abs(spl_error) > spl_tolerance:
                    print(
                        "[SPL mismatch] "
                        f"episode={len(returns)} manual={episode_spl:.8f} "
                        f"Habitat={habitat_spl:.8f} "
                        f"error={spl_error:+.8f}",
                        flush=True,
                    )
                last_heartbeat = time.perf_counter()

        now = time.perf_counter()
        if now - last_heartbeat >= 10.0:
            _print_progress(
                len(returns),
                policy_calls=policy_calls,
                active_envs=sum(
                    completed[i] < targets[i] for i in range(num_envs)
                ),
                mean_spl=(
                    sum(episode_spls) / len(episode_spls)
                    if episode_spls
                    else None
                ),
            )
            last_heartbeat = now

    if states.device.type == "cuda":
        torch.cuda.synchronize(states.device)
    rollout_seconds = time.perf_counter() - rollout_started
    _print_stage("All 100 evaluation episodes completed.")

    if cuda_events:
        policy_seconds = sum(
            start.elapsed_time(end) for start, end in cuda_events
        ) / 1_000.0
    else:
        policy_seconds = cpu_policy_seconds

    if len(returns) != TOTAL_EPISODES:
        raise RuntimeError(
            f"Expected {TOTAL_EPISODES} completed episodes, got {len(returns)}"
        )
    if action_decisions != sum(steps):
        raise RuntimeError(
            "Action decision count does not match completed-episode steps: "
            f"{action_decisions} != {sum(steps)}"
        )
    if policy_decisions + exploration_decisions != action_decisions:
        raise RuntimeError(
            "Greedy and exploration decisions do not sum to all actions: "
            f"{policy_decisions} + {exploration_decisions} != "
            f"{action_decisions}"
        )
    if sum(action_counts) != action_decisions:
        raise RuntimeError(
            "Executed action counts do not sum to all actions: "
            f"{sum(action_counts)} != {action_decisions}"
        )
    if len(episode_spls) != len(returns):
        raise RuntimeError(
            "SPL count does not match completed episodes: "
            f"{len(episode_spls)} != {len(returns)}"
        )
    if len(habitat_spls) != len(returns):
        raise RuntimeError(
            "Habitat SPL count does not match completed episodes: "
            f"{len(habitat_spls)} != {len(returns)}"
        )

    spl_mismatch_episodes = sum(
        abs(error) > spl_tolerance for error in spl_errors
    )

    return {
        "episodes": len(returns),
        "num_envs": num_envs,
        "episode_targets": targets,
        "epsilon": epsilon,
        "policy_calls": policy_calls,
        "policy_decisions": policy_decisions,
        "action_decisions": action_decisions,
        "exploration_decisions": exploration_decisions,
        "action_counts": {
            name: action_counts[index]
            for index, name in enumerate(ACTION_NAMES)
        },
        "policy_seconds": policy_seconds,
        "policy_ms_per_call": 1_000.0 * policy_seconds / policy_calls,
        "policy_ms_per_decision": 1_000.0 * policy_seconds / policy_decisions,
        "policy_seconds_per_episode": policy_seconds / len(returns),
        "rollout_seconds": rollout_seconds,
        "rollout_seconds_per_episode": rollout_seconds / len(returns),
        "episodes_per_rollout_second": len(returns) / rollout_seconds,
        "mean_episode_steps": sum(steps) / len(steps),
        "success_rate": successes / len(returns),
        "spl": sum(episode_spls) / len(episode_spls),
        "manual_spl": sum(episode_spls) / len(episode_spls),
        "habitat_spl": sum(habitat_spls) / len(habitat_spls),
        "spl_tolerance": spl_tolerance,
        "spl_mismatch_episodes": spl_mismatch_episodes,
        "spl_max_abs_error": max(map(abs, spl_errors)),
        "spl_start_distance_max_abs_error": max(
            map(abs, start_distance_errors)
        ),
        "mean_optimal_path_length": (
            sum(optimal_distances) / len(optimal_distances)
        ),
        "mean_traveled_path_length": (
            sum(traveled_distances) / len(traveled_distances)
        ),
        "mean_successful_path_efficiency": (
            sum(successful_path_efficiencies)
            / len(successful_path_efficiencies)
            if successful_path_efficiencies
            else 0.0
        ),
        "min_successful_path_efficiency": (
            min(successful_path_efficiencies)
            if successful_path_efficiencies
            else 0.0
        ),
        "mean_reward": sum(returns) / len(returns),
    }


def _print_result(result: dict[str, Any]) -> None:
    print("\nPolicy inference timing (exactly 100 completed episodes)")
    print(f"  Evaluation profile:          {result['profile']}")
    print(f"  Checkpoint step:             {result['checkpoint_step']}")
    print(f"  Fixated target filtering:    {result['fixated_object']}")
    print(f"  Parallel environments:       {result['num_envs']}")
    print(f"  Epsilon:                     {result['epsilon']:.3f}")
    print(f"  Total action decisions:      {result['action_decisions']}")
    print(f"  Policy decisions:            {result['policy_decisions']}")
    print(f"  Exploration decisions:       {result['exploration_decisions']}")
    print(f"  Executed action counts:      {result['action_counts']}")
    print(f"  Batched policy calls:        {result['policy_calls']}")
    print(f"  Policy time:                 {result['policy_seconds']:.3f} s")
    print(
        "  Policy time / call:          "
        f"{result['policy_ms_per_call']:.3f} ms"
    )
    print(
        "  Policy time / decision:      "
        f"{result['policy_ms_per_decision']:.3f} ms"
    )
    print(
        "  Policy time / episode:       "
        f"{result['policy_seconds_per_episode']:.3f} s"
    )
    print(f"  End-to-end rollout time:     {result['rollout_seconds']:.3f} s")
    print(
        "  Rollout time / episode:      "
        f"{result['rollout_seconds_per_episode']:.3f} s"
    )
    print(
        "  Rollout throughput:          "
        f"{result['episodes_per_rollout_second']:.3f} episodes/s"
    )
    print(f"  Mean episode length:         {result['mean_episode_steps']:.1f}")
    print(f"  Success rate:                {result['success_rate']:.3f}")
    print(f"  Manual SPL:                  {result['manual_spl']:.6f}")
    print(f"  Habitat SPL:                 {result['habitat_spl']:.6f}")
    print(
        "  SPL mismatch episodes:       "
        f"{result['spl_mismatch_episodes']}"
    )
    print(
        "  Maximum SPL difference:      "
        f"{result['spl_max_abs_error']:.8f}"
    )
    print(
        "  Start-distance difference:   "
        f"{result['spl_start_distance_max_abs_error']:.8f} m"
    )
    print(
        "  Mean optimal path length:    "
        f"{result['mean_optimal_path_length']:.3f} m"
    )
    print(
        "  Mean traveled path length:   "
        f"{result['mean_traveled_path_length']:.3f} m"
    )
    print(
        "  Successful path efficiency:  "
        f"{result['mean_successful_path_efficiency']:.6f} mean, "
        f"{result['min_successful_path_efficiency']:.6f} minimum"
    )
    print(f"  Mean reward:                 {result['mean_reward']:.3f}")


def main() -> None:
    _print_stage("Parsing command-line arguments...")
    args = parse_args()
    if args.num_envs is not None and not 1 <= args.num_envs <= TOTAL_EPISODES:
        raise ValueError(
            f"--num-envs must be between 1 and {TOTAL_EPISODES}"
        )
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive")
    if args.spl_tolerance < 0:
        raise ValueError("--spl-tolerance must be non-negative")
    if args.epsilon is not None and not 0.0 <= args.epsilon < 1.0:
        raise ValueError("--epsilon must be in [0, 1)")

    if args.data_path is None:
        data_path = (
            DEFAULT_STAGE2_DATA_PATH
            if args.profile == "stage2"
            else DEFAULT_HELDOUT_DATA_PATH
        )
    else:
        data_path = args.data_path
    task_config_path = args.task_config.expanduser().resolve()
    data_path = data_path.expanduser().resolve()
    if not task_config_path.is_file():
        raise FileNotFoundError(
            f"Habitat task config does not exist: {task_config_path}"
        )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Evaluation split config does not exist: {data_path}"
        )
    _print_stage("Arguments and input paths validated.")

    _print_stage("Preparing project imports...")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from config_utils import load_config
    from skrl.utils import set_seed
    from trainer import build_agent
    _print_stage("Project imports complete.")

    _print_stage(f"Loading configuration from {args.config}...")
    cfg = load_config(args.config)
    set_seed(int(cfg["run"]["seed"]))
    for value in reversed(cfg["run"].get("python_paths", [])):
        path = str((REPO_ROOT / value).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    _print_stage("Configuration loaded and random seed set.")

    if args.num_envs is not None:
        num_envs = args.num_envs
    elif args.profile == "stage2":
        num_envs = DEFAULT_STAGE2_NUM_ENVS
    else:
        num_envs = int(cfg["run"]["num_envs"])
    if not 1 <= num_envs <= TOTAL_EPISODES:
        raise ValueError(
            f"run.num_envs must be between 1 and {TOTAL_EPISODES}; got "
            f"{num_envs}"
        )

    epsilon = (
        args.epsilon
        if args.epsilon is not None
        else (
            float(cfg["agent"]["final_epsilon"])
            if args.profile == "stage2"
            else 0.0
        )
    )
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"Resolved epsilon must be in [0, 1); got {epsilon}")

    (
        cfg["run"]["agent_checkpoint"],
        cfg["aux"]["resume_from"],
        checkpoint_step,
    ) = _select_checkpoints(
        cfg,
        agent_override=args.agent_checkpoint,
        perception_override=args.perception_checkpoint,
        requested_step=args.checkpoint_step,
    )
    cfg["run"]["eval"] = True
    cfg["run"]["num_envs"] = num_envs
    fixated_object = args.profile == "stage2"
    shuffle_episodes = args.profile == "stage2"
    _print_stage(
        "Evaluation profile resolved: "
        f"profile={args.profile}, num_envs={num_envs}, "
        f"fixated_object={fixated_object}, "
        f"shuffle_episodes={shuffle_episodes}, epsilon={epsilon:.3f}, "
        f"checkpoint_step={checkpoint_step or 'custom'}."
    )
    if args.profile == "stage2":
        _print_stage(
            "Stage2 comparison uses policy actions only; curriculum "
            "expert/controller action substitution is intentionally disabled."
        )
        _print_stage(
            f"Episode safety cap: {args.max_episode_steps} steps. "
            "This prevents a non-stopping policy from hanging the timing run; "
            "override it explicitly if a longer horizon is required."
        )
    _print_stage(
        f"DDQN checkpoint: {cfg['run']['agent_checkpoint']}"
    )
    _print_stage(
        f"Perception checkpoint: {cfg['aux']['resume_from']}"
    )

    _print_stage("Building evaluation environment...")
    env = _build_eval_env(
        cfg,
        num_envs,
        args.max_episode_steps,
        task_config_path,
        data_path,
        fixated_object,
        shuffle_episodes,
    )
    _print_stage("Evaluation environment build complete.")
    agent = aux_trainer = perception = None
    try:
        _print_stage("Building policy and loading checkpoints...")
        agent, aux_trainer, perception = build_agent(env, cfg)
        _print_stage("Policy and checkpoints loaded.")
        _print_stage("Starting the 100-episode timed evaluation...")
        result = _collect_episodes(
            env,
            agent,
            _episode_targets(TOTAL_EPISODES, num_envs),
            epsilon=epsilon,
            spl_tolerance=args.spl_tolerance,
        )
        result["profile"] = args.profile
        result["fixated_object"] = fixated_object
        result["shuffle_episodes"] = shuffle_episodes
        result["checkpoint_step"] = checkpoint_step
        result["config"] = str(Path(args.config).expanduser().resolve())
        result["agent_checkpoint"] = cfg["run"]["agent_checkpoint"]
        result["perception_checkpoint"] = cfg["aux"]["resume_from"]
        result["task_config"] = str(task_config_path)
        result["data_path"] = str(data_path)
        result["max_episode_steps"] = args.max_episode_steps
        _print_stage("Computing and printing timing summary...")
        _print_result(result)
        _print_stage("Timing summary printed.")

        if args.json_out is not None:
            _print_stage(f"Writing JSON result to {args.json_out}...")
            output_path = args.json_out.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
            print(f"\nWrote JSON result to {output_path}", flush=True)
            _print_stage("JSON result written.")
    finally:
        _print_stage("Closing evaluation environment...")
        env.close()
        _print_stage("Evaluation environment closed.")
        del perception, aux_trainer, agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _print_stage("Cleanup complete.")


if __name__ == "__main__":
    main()
