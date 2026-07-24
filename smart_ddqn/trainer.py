from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from skrl.agents.torch.dqn.ddqn import DDQN, DDQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.sequential import SequentialTrainer
from skrl.utils import set_seed

from aux_trainer import ReplayOrientationAuxTrainer
from habitat_factory import build_habitat_env
from models import DDQNQNetwork
from perception import AuxPerceptionModules

from utils.habitat_utils import setup_env_config
from habitat.config import read_write
from skrl.envs.wrappers.torch import wrap_env
from curriculum_habitat.helper_wrappers import (
    CLIPWrapper,
    ToSKRLWrapper,
)
from curriculum_habitat.curriculum_wrapper import (
    CurriculumVectorEnv,
    ObjRLNav,
)

DEFAULT_TASK_CONFIG_PATH = "configs/objectnav_hm3d_v2_with_semantic.yaml"
DEFAULT_DATA_PATH = "configs/homerobot_hm3d_objectnav_train.yaml"
EVAL_DATA_PATH = "configs/homerobot_hm3d_objectnav_val.yaml"

NUM_OF_PARALLEL_ENVS = 5
EVAL_ROUNDS = 5

NUM_OF_STEPS = 100_000
EVAL_INTERVAL = 500

def create_homerobot_env(
    task_config_path=DEFAULT_TASK_CONFIG_PATH,
    data_path=DEFAULT_DATA_PATH,
    index=0,
):
    config = setup_env_config(
        params_path=data_path,
        default_config_path=task_config_path,
    )
    with read_write(config):
        config.habitat.seed = int(config.habitat.seed) + index
    env = ObjRLNav(config=config)
    return env

def make_env_vectorised(create_env_fn, task_config_path, data_path, num_envs):
    vec_env = CurriculumVectorEnv(
            make_env_fn=create_env_fn,
            env_fn_args=[
                (task_config_path, data_path, index)
                for index in range(num_envs)
            ],
        )
    vec_env = CLIPWrapper(vec_env, device="cuda")
    vec_env = ToSKRLWrapper(vec_env, device="cuda")
    vec_env = wrap_env(vec_env, wrapper='gymnasium')
    return vec_env

def _parameter_ids(module: torch.nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def assert_parameter_separation(
    *,
    online_model: DDQNQNetwork,
    target_model: DDQNQNetwork,
    perception: AuxPerceptionModules,
) -> None:
    """Fail immediately if perception leaked into either DDQN model."""
    perception_ids = _parameter_ids(perception)
    online_ids = _parameter_ids(online_model)
    target_ids = _parameter_ids(target_model)

    online_overlap = perception_ids & online_ids
    target_overlap = perception_ids & target_ids
    online_target_overlap = online_ids & target_ids

    if online_overlap or target_overlap:
        raise RuntimeError(
            "Perception parameters were registered inside a DDQN model. "
            f"online_overlap={len(online_overlap)}, "
            f"target_overlap={len(target_overlap)}"
        )
    if online_target_overlap:
        raise RuntimeError(
            "Online and target DDQN networks unexpectedly share parameters: "
            f"overlap={len(online_target_overlap)}"
        )

    print(
        "[SEPARATION] "
        f"ddqn_online_params={sum(p.numel() for p in online_model.parameters())} "
        f"ddqn_target_params={sum(p.numel() for p in target_model.parameters())} "
        f"perception_params={sum(p.numel() for p in perception.parameters())} "
        "overlap=0",
        flush=True,
    )


def build_agent(env, cfg: dict[str, Any]):
    # Perception is constructed independently and is shared only as a detached
    # feature provider by online and target Q-networks.
    perception = AuxPerceptionModules(cfg).to(env.device)
    perception.eval()

    online_model = DDQNQNetwork(
        env.observation_space,
        env.action_space,
        env.device,
        cfg,
        perception=perception,
    )
    target_model = DDQNQNetwork(
        env.observation_space,
        env.action_space,
        env.device,
        cfg,
        perception=perception,
    )
    assert_parameter_separation(
        online_model=online_model,
        target_model=target_model,
        perception=perception,
    )

    models = {
        "q_network": online_model,
        "target_q_network": target_model,
    }

    memory = RandomMemory(
        memory_size=int(cfg["agent"]["memory_size"]),
        num_envs=env.num_envs,
        device=env.device,
    )

    agent_cfg = deepcopy(DDQN_DEFAULT_CONFIG)
    agent_cfg["gradient_steps"] = int(cfg["agent"]["gradient_steps"])
    agent_cfg["batch_size"] = int(cfg["agent"]["batch_size"])
    agent_cfg["discount_factor"] = float(cfg["agent"]["gamma"])
    agent_cfg["polyak"] = float(cfg["agent"]["polyak"])
    agent_cfg["learning_rate"] = float(cfg["agent"]["learning_rate"])
    agent_cfg["random_timesteps"] = int(cfg["agent"]["random_timesteps"])
    agent_cfg["learning_starts"] = int(cfg["agent"]["learning_starts"])
    agent_cfg["update_interval"] = int(cfg["agent"]["update_interval"])
    agent_cfg["target_update_interval"] = int(
        cfg["agent"]["target_update_interval"]
    )
    agent_cfg["exploration"]["initial_epsilon"] = float(
        cfg["agent"]["initial_epsilon"]
    )
    agent_cfg["exploration"]["final_epsilon"] = float(
        cfg["agent"]["final_epsilon"]
    )
    agent_cfg["exploration"]["timesteps"] = int(
        cfg["agent"]["exploration_timesteps"]
    )

    log_dir = Path(cfg["run"]["log_dir"])
    agent_cfg["experiment"]["directory"] = str(log_dir.parent)
    agent_cfg["experiment"]["experiment_name"] = str(
        cfg["run"].get("experiment_name", log_dir.name)
    )
    agent_cfg["experiment"]["write_interval"] = int(
        cfg["run"]["write_interval"]
    )
    agent_cfg["experiment"]["checkpoint_interval"] = int(
        cfg["run"]["checkpoint_interval"]
    )

    if bool(cfg["run"].get("eval", False)):
        agent_cfg["random_timesteps"] = 0
        agent_cfg["exploration"]["initial_epsilon"] = 0.0
        agent_cfg["exploration"]["final_epsilon"] = 0.0
        agent_cfg["exploration"]["timesteps"] = 0

    agent = DDQN(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=agent_cfg,
    )

    # The DDQN checkpoint contains only DDQN parameters by design.
    agent_checkpoint = cfg["run"].get("agent_checkpoint")
    if agent_checkpoint:
        agent.load(str(agent_checkpoint))

    # The separate perception checkpoint is loaded by the aux trainer, even in
    # evaluation mode. This is required for orientation_source='pred'.
    aux_trainer = ReplayOrientationAuxTrainer(
        agent=agent,
        perception=perception,
        observation_space=env.observation_space,
        device=env.device,
        cfg=cfg,
    )
    if not bool(cfg["run"].get("eval", False)) and aux_trainer.enabled:
        aux_trainer.attach()

    if (
        str(cfg["model"].get("orientation_source", "pred")) == "pred"
        and not cfg["aux"].get("resume_from")
    ):
        print(
            "[WARNING] orientation_source='pred' but aux.resume_from is null. "
            "The orientation module starts from random weights unless this is "
            "a fresh joint training run.",
            flush=True,
        )

    return agent, aux_trainer, perception


def run_training(cfg: dict[str, Any]) -> None:
    set_seed(int(cfg["run"]["seed"]))

    # OLD
    # raw_env = build_habitat_env(cfg)
    # Explicit wrapper selection prevents skrl from misidentifying the custom adapter.
    # env = wrap_env(raw_env, wrapper="gymnasium")
    # NEW
    env = make_env_vectorised(
        create_homerobot_env,
        DEFAULT_TASK_CONFIG_PATH,
        DEFAULT_DATA_PATH,
        NUM_OF_PARALLEL_ENVS,
    )

    agent = None
    aux_trainer = None
    perception = None
    try:
        agent, aux_trainer, perception = build_agent(env, cfg)
        trainer = SequentialTrainer(
            env=env,
            agents=agent,
            cfg={
                "timesteps": int(cfg["run"]["timesteps"]),
                "headless": True,
                "close_environment_at_exit": False,
                # DDQN evaluation must consume the action returned by agent.act().
                "stochastic_evaluation": True,
            },
        )

        if bool(cfg["run"].get("eval", False)):
            trainer.eval()
        else:
            trainer.train()
    finally:
        env.close()
        del perception, aux_trainer, agent
        torch.cuda.empty_cache()
