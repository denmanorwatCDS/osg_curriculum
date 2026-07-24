from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from skrl.utils.spaces.torch import unflatten_tensorized_space

from models import DDQNPerceptionQNetwork


class ReplayOrientationAuxTrainer:
    """Train online perception from GT teacher values already stored in replay memory."""

    def __init__(
        self,
        *,
        agent,
        online_model: DDQNPerceptionQNetwork,
        observation_space,
        device,
        cfg: dict[str, Any],
    ):
        self.agent = agent
        self.model = online_model
        self.perception = online_model.perception
        self.observation_space = observation_space
        self.device = device

        aux = cfg["aux"]
        self.enabled = bool(aux.get("enabled", True))
        self.batch_size = int(aux["batch_size"])
        self.updates_per_step = int(aux.get("updates_per_step", 1))
        self.start_after_samples = int(aux.get("start_after_samples", self.batch_size))
        self.train_encoders = bool(aux.get("train_encoders", True))
        self.grad_norm_clip = float(aux.get("grad_norm_clip", 1.0))
        self.log_interval = int(aux.get("log_interval", 500))
        self.save_interval = int(aux.get("save_interval", 10000))

        parameters = (
            self.perception.parameters()
            if self.train_encoders
            else self.perception.orientation_head.parameters()
        )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(aux["learning_rate"]),
            weight_decay=1e-4,
        )

        checkpoint_dir = aux.get("checkpoint_dir")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric_sum: dict[str, float] = {}
        self.metric_count = 0

        resume_from = aux.get("resume_from")
        if resume_from:
            self.load(resume_from)

    def attach(self) -> None:
        original_post_interaction = self.agent.post_interaction

        def post_interaction_with_aux(timestep: int, timesteps: int):
            original_post_interaction(timestep, timesteps)
            self.step(timestep)

        self.agent.post_interaction = post_interaction_with_aux

    def step(self, timestep: int) -> None:
        if not self.enabled or not self._enough_samples():
            return

        self.model.train()
        for _ in range(self.updates_per_step):
            states = self._sample_states()
            loss, metrics = self.perception.orientation_loss(
                states["img"],
                states["graph"],
                states["teacher_orientation"],
                train_encoders=self.train_encoders,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.perception.parameters(),
                self.grad_norm_clip,
            )
            self.optimizer.step()

            metrics = dict(metrics)
            metrics["grad_norm"] = float(grad_norm)
            self._accumulate(metrics)

        if self.log_interval > 0 and timestep % self.log_interval == 0:
            self._print_metrics(timestep)
        if self.save_interval > 0 and timestep % self.save_interval == 0:
            self.save(timestep)

    def _enough_samples(self) -> bool:
        memory = self.agent.memory
        if getattr(memory, "filled", False):
            return True
        # In the skrl version used by the supplied pipeline, memory_index counts
        # vector time slots. Multiplication converts it to approximate transitions.
        slots = int(getattr(memory, "memory_index", 0))
        num_envs = int(getattr(memory, "num_envs", 1))
        return slots * num_envs >= max(self.start_after_samples, self.batch_size)

    def _sample_states(self) -> dict[str, torch.Tensor]:
        # This follows the memory.sample layout used by the supplied aux trainer.
        sample_result = self.agent.memory.sample(
            names=["states"],
            batch_size=self.batch_size,
        )
        samples = sample_result[0] if isinstance(sample_result, tuple) else sample_result
        raw_states = samples[0]
        return unflatten_tensorized_space(self.observation_space, raw_states)

    def _accumulate(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.metric_sum[key] = self.metric_sum.get(key, 0.0) + float(value)
        self.metric_count += 1

    def _print_metrics(self, timestep: int) -> None:
        if self.metric_count == 0:
            return
        averaged = {
            key: value / self.metric_count
            for key, value in sorted(self.metric_sum.items())
        }
        line = " | ".join(f"{key}={value:.4f}" for key, value in averaged.items())
        print(f"[AUX {timestep}] {line}", flush=True)
        self.metric_sum.clear()
        self.metric_count = 0

    def save(self, timestep: int) -> None:
        if self.checkpoint_dir is None:
            return
        payload = {
            "timestep": int(timestep),
            "perception": self.perception.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        numbered = self.checkpoint_dir / f"aux_{int(timestep)}.pt"
        latest = self.checkpoint_dir / "aux_latest.pt"
        torch.save(payload, numbered)
        torch.save(payload, latest)
        print(f"[AUX] saved {numbered}", flush=True)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.perception.load_state_dict(payload["perception"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        print(f"[AUX] loaded {path}", flush=True)
