from __future__ import annotations

import weakref
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from perception import AuxPerceptionModules, make_mlp


class DDQNQNetwork(DeterministicMixin, Model):
    """One Q-network implementation used for online and target DDQN roles.

    The network owns only RL parameters:

    - a DDQN-specific image encoder;
    - a goal encoder;
    - a target-object embedding encoder;
    - the Q-value head.

    Graph embedding and orientation prediction are supplied by an external
    ``AuxPerceptionModules`` object through a weak reference. A weak reference is
    intentional: PyTorch therefore does not register perception as a child
    module, and the DDQN optimizer/checkpoint cannot include its parameters.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        cfg: dict[str, Any],
        *,
        perception: AuxPerceptionModules,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(
                f"DDQN requires Discrete action space, got {type(action_space)}"
            )

        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Do not assign the nn.Module itself to this Q-network. A weak reference
        # prevents nn.Module.__setattr__ from registering perception parameters.
        self._perception_ref = weakref.ref(perception)

        dims = cfg["observation"]["dims"]
        model = cfg["model"]

        dqn_img_hidden = int(model["dqn_img_hidden"])
        goal_hidden = int(model["goal_hidden"])
        emb_obj_hidden = int(model["emb_obj_hidden"])

        self.img_encoder = make_mlp(
            int(dims["img"]),
            [dqn_img_hidden],
            dqn_img_hidden,
        )
        self.goal_encoder = make_mlp(
            int(dims["goal"]),
            [goal_hidden],
            goal_hidden,
        )
        self.emb_obj_encoder = make_mlp(
            int(dims["emb_obj"]),
            [emb_obj_hidden],
            emb_obj_hidden,
        )

        q_input_dim = (
            dqn_img_hidden
            + goal_hidden
            + emb_obj_hidden
            + int(perception.graph_output_dim)
            + 2  # orientation represented as [sin(yaw), cos(yaw)]
        )
        hidden_dims = [int(value) for value in model["q_hidden"]]
        self.q_head = make_mlp(q_input_dim, hidden_dims, self.num_actions)

    def _perception(self) -> AuxPerceptionModules:
        perception = self._perception_ref()
        if perception is None:
            raise RuntimeError(
                "External perception modules were destroyed while DDQN is active"
            )
        return perception

    def compute(self, inputs, role):
        observation = unflatten_tensorized_space(
            self.observation_space,
            inputs["states"],
        )
        img = observation["observation"]
        goal = observation["absolute_goal_position"]
        graph = observation["knowledge_graph"]
        emb_obj = observation["goal_description"]
        teacher_orientation = observation.get("angle_to_goal")

        graph_embedding, orientation_feature = (
            self._perception().detached_policy_features(
                img=img,
                goal=goal,
                emb_obj=emb_obj,
                graph=graph,
                teacher_yaw=teacher_orientation,
            )
        )

        # These encoders and q_head are the only trainable DDQN path.
        img_feature = self.img_encoder(img.float())
        goal_feature = self.goal_encoder(goal.float())
        emb_obj_feature = self.emb_obj_encoder(emb_obj.float())

        q_input = torch.cat(
            (
                img_feature,
                goal_feature,
                emb_obj_feature,
                graph_embedding,
                orientation_feature,
            ),
            dim=-1,
        )
        return self.q_head(q_input), {}
