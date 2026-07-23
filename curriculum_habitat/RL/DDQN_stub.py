import torch

from skrl.agents.torch.dqn import DDQN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space["observation"].shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_actions),
        )

    def compute(self, inputs, role):
        observation = unflatten_tensorized_space(
            self.observation_space, inputs["states"]
        )["observation"]
        return self.net(observation), {}


def create_ddqn(env):
    models = {
        name: QNetwork(env.observation_space, env.action_space, env.device)
        for name in ("q_network", "target_q_network")
    }
    return DDQN(
        models=models,
        memory=RandomMemory(10_000, env.num_envs, env.device),
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg={"random_timesteps": 1_000, "learning_starts": 1_000},
    )
