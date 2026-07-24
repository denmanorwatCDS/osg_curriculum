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
from models import DDQNPerceptionQNetwork


def build_agent(env, cfg: dict[str, Any]):
    online_model = DDQNPerceptionQNetwork(
        env.observation_space,
        env.action_space,
        env.device,
        cfg,
    )
    target_model = DDQNPerceptionQNetwork(
        env.observation_space,
        env.action_space,
        env.device,
        cfg,
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
    agent_cfg["target_update_interval"] = int(cfg["agent"]["target_update_interval"])
    agent_cfg["exploration"]["initial_epsilon"] = float(cfg["agent"]["initial_epsilon"])
    agent_cfg["exploration"]["final_epsilon"] = float(cfg["agent"]["final_epsilon"])
    agent_cfg["exploration"]["timesteps"] = int(cfg["agent"]["exploration_timesteps"])

    log_dir = Path(cfg["run"]["log_dir"])
    agent_cfg["experiment"]["directory"] = str(log_dir.parent)
    agent_cfg["experiment"]["experiment_name"] = str(
        cfg["run"].get("experiment_name", log_dir.name)
    )
    agent_cfg["experiment"]["write_interval"] = int(cfg["run"]["write_interval"])
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

    checkpoint = cfg["run"].get("agent_checkpoint")
    if checkpoint:
        agent.load(str(checkpoint))

    aux_trainer = ReplayOrientationAuxTrainer(
        agent=agent,
        online_model=online_model,
        observation_space=env.observation_space,
        device=env.device,
        cfg=cfg,
    )
    if not bool(cfg["run"].get("eval", False)):
        aux_trainer.attach()

    return agent, aux_trainer


def run_training(cfg: dict[str, Any]) -> None:
    set_seed(int(cfg["run"]["seed"]))

    raw_env = build_habitat_env(cfg)
    # Explicit wrapper selection prevents skrl from misidentifying the custom adapter.
    env = wrap_env(raw_env, wrapper="gymnasium")

    agent = None
    aux_trainer = None
    try:
        agent, aux_trainer = build_agent(env, cfg)
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
        del aux_trainer, agent
        torch.cuda.empty_cache()
