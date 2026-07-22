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

    def _encode(self, obs):
        images = [obs["observation"][i, ..., :3] for i in range(obs["observation"].shape[0])]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        obs["observation"] = torch.nn.functional.normalize(embeddings).cpu().numpy()
        return obs

    def reset(self): 
        return self._encode(self.env.reset())
    
    def step(self, actions):
        obs, *rest = self.env.step(actions)
        return (self._encode(obs), *rest)
    
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
