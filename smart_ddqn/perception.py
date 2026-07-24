from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
        previous = int(hidden)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


class SimplePerception(nn.Module):
    """Image + graph encoder with an auxiliary orientation classifier.

    The policy receives the predicted orientation as a circular pair [sin, cos].
    The teacher yaw is never passed into policy features.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        dims = cfg["observation"]["dims"]
        model = cfg["model"]

        self.img_dim = int(dims["img"])
        self.graph_dim = int(dims["graph"])
        self.img_hidden = int(model["img_hidden"])
        self.graph_hidden = int(model["graph_hidden"])
        self.num_bins = int(model["orientation_bins"])

        self.img_encoder = make_mlp(self.img_dim, [self.img_hidden], self.img_hidden)
        self.graph_encoder = make_mlp(
            self.graph_dim, [self.graph_hidden], self.graph_hidden
        )
        self.orientation_head = make_mlp(
            self.img_hidden + self.graph_hidden,
            [self.img_hidden],
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

    @property
    def policy_output_dim(self) -> int:
        return self.img_hidden + self.graph_hidden + 2

    def encode(self, img: torch.Tensor, graph: torch.Tensor):
        img_feature = self.img_encoder(img.float())
        graph_feature = self.graph_encoder(graph.float())
        return img_feature, graph_feature

    def orientation_logits(
        self,
        img: torch.Tensor,
        graph: torch.Tensor,
        *,
        detach_encoders: bool = False,
    ) -> torch.Tensor:
        img_feature, graph_feature = self.encode(img, graph)
        if detach_encoders:
            img_feature = img_feature.detach()
            graph_feature = graph_feature.detach()
        return self.orientation_head(torch.cat((img_feature, graph_feature), dim=-1))

    def policy_features(self, img: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        img_feature, graph_feature = self.encode(img, graph)
        logits = self.orientation_head(torch.cat((img_feature, graph_feature), dim=-1))
        probabilities = F.softmax(logits, dim=-1)

        # Circular representation avoids the discontinuity between -pi and +pi.
        predicted_sin = probabilities @ self.bin_sin
        predicted_cos = probabilities @ self.bin_cos
        orientation_feature = torch.stack((predicted_sin, predicted_cos), dim=-1)
        return torch.cat((img_feature, graph_feature, orientation_feature), dim=-1)

    def orientation_loss(
        self,
        img: torch.Tensor,
        graph: torch.Tensor,
        teacher_yaw: torch.Tensor,
        *,
        train_encoders: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.orientation_logits(
            img,
            graph,
            detach_encoders=not train_encoders,
        )
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
                "accuracy": float((predicted_bins == labels).float().mean().item()),
                "mean_error_deg": float((error.mean() * 180.0 / math.pi).item()),
            }
        return loss, metrics
