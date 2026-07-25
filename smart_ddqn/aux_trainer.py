from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from skrl.utils.spaces.torch import unflatten_tensorized_space

from perception import AuxPerceptionModules


class ReplayOrientationAuxTrainer:
    """Train graph embedding and orientation modules outside DDQN.

    Teacher targets are sampled from ``states`` already stored in the skrl
    replay buffer. DDQN parameters are never passed to either auxiliary
    optimizer.
    """

    def __init__(
        self,
        *,
        agent,
        perception: AuxPerceptionModules,
        observation_space,
        device,
        cfg: dict[str, Any],
    ):
        self.agent = agent
        self.perception = perception
        self.observation_space = observation_space
        self.device = device

        aux = cfg["aux"]
        self.enabled = bool(aux.get("enabled", True))
        self.batch_size = int(aux["batch_size"])
        self.updates_per_step = int(aux.get("updates_per_step", 1))
        self.start_after_samples = int(
            aux.get("start_after_samples", self.batch_size)
        )
        self.grad_norm_clip = float(aux.get("grad_norm_clip", 1.0))
        self.log_interval = int(aux.get("log_interval", 500))
        self.save_interval = int(aux.get("save_interval", 10000))

        weight_decay = float(aux.get("weight_decay", 1e-4))
        self.graph_optimizer = torch.optim.AdamW(
            self.perception.graph_encoder.parameters(),
            lr=float(aux["learning_rate_graph"]),
            weight_decay=weight_decay,
        )
        self.orientation_optimizer = torch.optim.AdamW(
            self.perception.orientation_module.parameters(),
            lr=float(aux["learning_rate_orientation"]),
            weight_decay=weight_decay,
        )

        checkpoint_dir = aux.get("checkpoint_dir")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric_sum: dict[str, float] = {}
        self.metric_count = 0

        resume_from = aux.get("resume_from")
        if resume_from:
            self.load(
                resume_from,
                load_optimizers=not bool(cfg["run"].get("eval", False)),
            )

        self.perception.eval()

    def attach(self) -> None:
        """Run one auxiliary phase after the normal DDQN post-interaction."""
        original_post_interaction = self.agent.post_interaction

        def post_interaction_with_aux(timestep: int, timesteps: int):
            original_post_interaction(timestep, timesteps)
            self.step(timestep)

        self.agent.post_interaction = post_interaction_with_aux

    def step(self, timestep: int) -> None:
        if not self.enabled or not self._enough_samples():
            return

        self.perception.train()
        for _ in range(self.updates_per_step):
            states = self._sample_states()
            loss, metrics = self.perception.orientation_loss(
                img=states["observation"],
                goal=states["absolute_goal_position"],
                emb_obj=states["goal_description"],
                graph=states["knowledge_graph"],
                teacher_yaw=states["angle_to_goal"],
            )

            self.graph_optimizer.zero_grad(set_to_none=True)
            self.orientation_optimizer.zero_grad(set_to_none=True)
            loss.backward()

            graph_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.perception.graph_encoder.parameters(),
                self.grad_norm_clip,
            )
            orientation_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.perception.orientation_module.parameters(),
                self.grad_norm_clip,
            )

            self.graph_optimizer.step()
            self.orientation_optimizer.step()

            metrics = dict(metrics)
            metrics["graph_grad_norm"] = float(graph_grad_norm)
            metrics["orientation_grad_norm"] = float(orientation_grad_norm)
            self._accumulate(metrics)

        # Policy inference must use deterministic perception behavior.
        self.perception.eval()

        if self.log_interval > 0 and timestep % self.log_interval == 0:
            self._print_metrics(timestep)
        if self.save_interval > 0 and timestep % self.save_interval == 0:
            self.save(timestep)

    def _enough_samples(self) -> bool:
        memory = self.agent.memory
        if getattr(memory, "filled", False):
            return True

        # In the skrl version used by the supplied pipeline, memory_index counts
        # vector slots. Multiplication approximates the number of transitions.
        slots = int(getattr(memory, "memory_index", 0))
        num_envs = int(getattr(memory, "num_envs", 1))
        return slots * num_envs >= max(
            self.start_after_samples,
            self.batch_size,
        )

    def _sample_states(self) -> dict[str, torch.Tensor]:
        sample_result = self.agent.memory.sample(
            names=["states"],
            batch_size=self.batch_size,
        )
        samples = (
            sample_result[0]
            if isinstance(sample_result, tuple)
            else sample_result
        )
        raw_states = samples[0][0]
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
        line = " | ".join(
            f"{key}={value:.4f}" for key, value in averaged.items()
        )
        print(f"[AUX {timestep}] {line}", flush=True)
        self.metric_sum.clear()
        self.metric_count = 0

    def save(self, timestep: int) -> None:
        if self.checkpoint_dir is None:
            return

        payload = {
            "timestep": int(timestep),
            "graph_encoder": self.perception.graph_encoder.state_dict(),
            "orientation_module": self.perception.orientation_module.state_dict(),
            "graph_optimizer": self.graph_optimizer.state_dict(),
            "orientation_optimizer": self.orientation_optimizer.state_dict(),
        }
        numbered = self.checkpoint_dir / f"perception_{int(timestep)}.pt"
        latest = self.checkpoint_dir / "perception_latest.pt"
        torch.save(payload, numbered)
        torch.save(payload, latest)
        print(f"[AUX] saved {numbered}", flush=True)

    def load(self, path: str, *, load_optimizers: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)

        if "graph_encoder" not in payload or "orientation_module" not in payload:
            raise KeyError(
                "Perception checkpoint must contain 'graph_encoder' and "
                "'orientation_module'. Old combined checkpoints are not "
                "compatible with the separated architecture."
            )

        self.perception.graph_encoder.load_state_dict(payload["graph_encoder"])
        self.perception.orientation_module.load_state_dict(
            payload["orientation_module"]
        )

        if load_optimizers:
            if "graph_optimizer" in payload:
                self.graph_optimizer.load_state_dict(payload["graph_optimizer"])
            if "orientation_optimizer" in payload:
                self.orientation_optimizer.load_state_dict(
                    payload["orientation_optimizer"]
                )

        self.perception.eval()
        print(f"[AUX] loaded {path}", flush=True)
