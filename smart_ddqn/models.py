from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from perception import SimplePerception, make_mlp


class DDQNPerceptionQNetwork(DeterministicMixin, Model):
    """The only network implementation used for both online and target DDQN roles."""

    def __init__(self, observation_space, action_space, device, cfg: dict[str, Any]):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(f"DDQN requires Discrete action space, got {type(action_space)}")

        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.perception = SimplePerception(cfg)
        goal_dim = int(cfg["observation"]["dims"]["goal"])
        goal_hidden = int(cfg["model"]["goal_hidden"])
        self.goal_encoder = make_mlp(goal_dim, [goal_hidden], goal_hidden)

        q_input_dim = self.perception.policy_output_dim + goal_hidden
        hidden_dims = [int(value) for value in cfg["model"]["q_hidden"]]
        self.q_head = make_mlp(q_input_dim, hidden_dims, self.num_actions)

    def compute(self, inputs, role):
        observation = unflatten_tensorized_space(
            self.observation_space,
            inputs["states"],
        )

        # teacher_orientation is intentionally ignored here. It is present only
        # so that replay memory can serve as the supervised aux dataset.
        perception_feature = self.perception.policy_features(
            observation["img"],
            observation["graph"],
        )
        goal_feature = self.goal_encoder(observation["goal"].float())
        q_values = self.q_head(torch.cat((perception_feature, goal_feature), dim=-1))
        return q_values, {}
