from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    """Build a small feed-forward network with explicit dimensions."""
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
        previous = int(hidden)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


class GraphEmbeddingModule(nn.Module):
    """Standalone graph embedding module.

    It is deliberately NOT a child of either DDQN Q-network. Its parameters are
    updated only by the auxiliary optimizer in ``ReplayOrientationAuxTrainer``.

    # TODO: Replace this flat MLP with the real graph encoder when the exact
    # Habitat graph transport format is fixed.
    """

    def __init__(self, graph_dim: int, embedding_dim: int):
        super().__init__()
        self.input_dim = int(graph_dim)
        self.output_dim = int(embedding_dim)
        self.net = make_mlp(
            self.input_dim,
            [self.output_dim],
            self.output_dim,
        )

    def forward(self, graph: torch.Tensor) -> torch.Tensor:
        return self.net(graph.float())


class OrientationModule(nn.Module):
    """Standalone orientation predictor.

    The module consumes the current image representation from Habitat and the
    graph embedding produced by ``GraphEmbeddingModule``. It predicts one of
    ``num_bins`` yaw bins and exposes a continuous circular feature [sin, cos].

    This module is also NOT a child of a DDQN Q-network. It is trained only by
    the auxiliary supervised loss.
    """

    def __init__(
        self,
        *,
        img_dim: int,
        graph_embedding_dim: int,
        img_hidden: int,
        hidden_dim: int,
        num_bins: int,
    ):
        super().__init__()
        self.img_dim = int(img_dim)
        self.graph_embedding_dim = int(graph_embedding_dim)
        self.img_hidden = int(img_hidden)
        self.hidden_dim = int(hidden_dim)
        self.num_bins = int(num_bins)

        self.img_encoder = make_mlp(
            self.img_dim,
            [self.img_hidden],
            self.img_hidden,
        )
        self.head = make_mlp(
            self.img_hidden + self.graph_embedding_dim,
            [self.hidden_dim],
            self.num_bins,
        )

        bin_size = 2.0 * math.pi / self.num_bins
        centers = (
            torch.arange(self.num_bins, dtype=torch.float32) * bin_size
            - math.pi
            + bin_size / 2.0
        )
        self.register_buffer("bin_centers", centers)
        self.register_buffer("bin_sin", torch.sin(centers))
        self.register_buffer("bin_cos", torch.cos(centers))

    def logits(self, img: torch.Tensor, graph_embedding: torch.Tensor) -> torch.Tensor:
        img_feature = self.img_encoder(img.float())
        return self.head(torch.cat((img_feature, graph_embedding.float()), dim=-1))

    def circular_feature_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=-1)
        predicted_sin = probabilities @ self.bin_sin
        predicted_cos = probabilities @ self.bin_cos
        return torch.stack((predicted_sin, predicted_cos), dim=-1)

    def predict_feature(
        self,
        img: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.circular_feature_from_logits(
            self.logits(img, graph_embedding)
        )

    def supervised_loss(
        self,
        img: torch.Tensor,
        graph_embedding: torch.Tensor,
        teacher_yaw: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.logits(img, graph_embedding)
        teacher_yaw = teacher_yaw.float().reshape(-1)
        wrapped = torch.atan2(torch.sin(teacher_yaw), torch.cos(teacher_yaw))

        bin_size = 2.0 * math.pi / self.num_bins
        labels = torch.floor((wrapped + math.pi) / bin_size).long() % self.num_bins

        loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
        with torch.no_grad():
            predicted_bins = logits.argmax(dim=-1)
            predicted_yaw = self.bin_centers[predicted_bins]
            error = torch.atan2(
                torch.sin(wrapped - predicted_yaw),
                torch.cos(wrapped - predicted_yaw),
            ).abs()
            metrics = {
                "loss": float(loss.detach().item()),
                "accuracy": float(
                    (predicted_bins == labels).float().mean().item()
                ),
                "mean_error_deg": float(
                    (error.mean() * 180.0 / math.pi).item()
                ),
            }
        return loss, metrics


class AuxPerceptionModules(nn.Module):
    """Container owned by the auxiliary learning path, not by DDQN.

    Separation contract:

    - ``graph_encoder`` and ``orientation_module`` are registered here;
    - neither DDQN online nor target network registers this container;
    - policy access always runs under ``torch.no_grad()`` and returns detached
      tensors;
    - only ``ReplayOrientationAuxTrainer`` owns optimizers for these parameters.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        dims = cfg["observation"]["dims"]
        model = cfg["model"]

        self.orientation_source = str(model.get("orientation_source", "pred"))
        if self.orientation_source not in {"gt", "pred"}:
            raise ValueError(
                "model.orientation_source must be 'gt' or 'pred', got "
                f"{self.orientation_source!r}"
            )

        self.graph_encoder = GraphEmbeddingModule(
            graph_dim=int(dims["graph"]),
            embedding_dim=int(model["graph_embedding_dim"]),
        )
        self.orientation_module = OrientationModule(
            img_dim=int(dims["img"]),
            graph_embedding_dim=int(model["graph_embedding_dim"]),
            img_hidden=int(model["orientation_img_hidden"]),
            hidden_dim=int(model["orientation_hidden"]),
            num_bins=int(model["orientation_bins"]),
        )

    @property
    def graph_output_dim(self) -> int:
        return int(self.graph_encoder.output_dim)

    @staticmethod
    def _gt_circular_feature(teacher_yaw: torch.Tensor) -> torch.Tensor:
        yaw = teacher_yaw.float().reshape(-1)
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)

    def detached_policy_features(
        self,
        *,
        img: torch.Tensor,
        graph: torch.Tensor,
        teacher_yaw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return graph embedding and orientation with no DDQN gradient path."""
        with torch.no_grad():
            graph_embedding = self.graph_encoder(graph)

            if self.orientation_source == "gt":
                if teacher_yaw is None:
                    raise RuntimeError(
                        "model.orientation_source='gt' requires "
                        "observation['teacher_orientation']"
                    )
                orientation_feature = self._gt_circular_feature(teacher_yaw)
            else:
                orientation_feature = self.orientation_module.predict_feature(
                    img,
                    graph_embedding,
                )

        # Explicit detach is retained even inside no_grad as an executable
        # statement of the separation contract.
        return graph_embedding.detach(), orientation_feature.detach()

    def orientation_loss(
        self,
        *,
        img: torch.Tensor,
        graph: torch.Tensor,
        teacher_yaw: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Aux loss; gradients flow through graph and orientation modules only."""
        graph_embedding = self.graph_encoder(graph)
        return self.orientation_module.supervised_loss(
            img,
            graph_embedding,
            teacher_yaw,
        )
