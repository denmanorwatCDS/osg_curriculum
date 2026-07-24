from typing import List, Optional, Sequence, Union

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch.sequential import SequentialTrainer


class EvalSequentialTrainer(SequentialTrainer):
    """Sequential trainer with deterministic DDQN evaluation.

    An integer ``eval_episodes`` is distributed between vector slots; a
    sequence specifies each slot's exact quota. Habitat evaluation environments
    must use ``cycle=True`` so slots that finish their quota earlier can
    continue stepping while the remaining slots finish.
    """

    def __init__(
        self,
        env: Wrapper,
        eval_env: Wrapper,
        agents: Union[Agent, List[Agent]],
        eval_episodes: Union[int, Sequence[int]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ):
        if isinstance(eval_episodes, int):
            if eval_episodes < 1:
                raise ValueError("eval_episodes must be positive")
        elif (
            len(eval_episodes) != eval_env.num_envs
            or any(episodes < 1 for episodes in eval_episodes)
        ):
            raise ValueError(
                "eval_episodes must contain one positive value per environment"
            )
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        super().__init__(env, agents, agents_scope, cfg)

    def eval(self):
        if self.num_simultaneous_agents != 1:
            raise NotImplementedError("Evaluation supports one agent")

        if hasattr(self.eval_env, "_reset_once"):
            self.eval_env._reset_once = True
        states, _ = self.eval_env.reset()
        episode_returns = torch.zeros(
            self.eval_env.num_envs, device=self.eval_env.device
        )
        if isinstance(self.eval_episodes, int):
            episodes_per_env, remainder = divmod(
                self.eval_episodes, self.eval_env.num_envs
            )
            targets = [
                episodes_per_env + (index < remainder)
                for index in range(self.eval_env.num_envs)
            ]
        else:
            targets = list(self.eval_episodes)
        completed = [0] * self.eval_env.num_envs
        returns, successes = [], 0.0
        self.agents.set_running_mode("eval")
        self.agents.set_mode("eval")

        try:
            while completed != targets:
                with torch.no_grad():
                    q_values = self.agents.q_network.act(
                        {"states": states}, role="q_network"
                    )[0]
                    actions = torch.argmax(q_values, dim=1, keepdim=True)
                    states, rewards, terminated, truncated, infos = (
                        self.eval_env.step(actions)
                    )

                episode_returns += rewards.reshape(-1)
                done_indices = torch.nonzero(
                    (terminated | truncated).reshape(-1), as_tuple=False
                ).reshape(-1)
                for index in done_indices.tolist():
                    if completed[index] < targets[index]:
                        returns.append(episode_returns[index].item())
                        successes += float(infos[index].get("success", False))
                        completed[index] += 1
                    episode_returns[index] = 0
        finally:
            self.agents.set_mode("train")
            self.agents.set_running_mode("train")

        return {
            "success_rate": successes / len(returns),
            "mean_reward": sum(returns) / len(returns),
        }
