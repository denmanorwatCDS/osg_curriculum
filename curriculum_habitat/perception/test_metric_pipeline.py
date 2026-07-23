"""End-to-end check of the metric scene-graph pipeline on real scenes.

semantics + selection + vocab  ->  MetricGraphBuilder  ->  graph_flat[28,6]
                               ->  MetricGraphEncoder (real CLIP table)  ->  [B,128]

Verifies: shapes, batching across scenes, that different goals give different embeddings,
and that gradients flow back to the encoder. Run:
    python -m curriculum_habitat.perception.test_metric_pipeline
"""

import json
from pathlib import Path

import numpy as np
import torch

from curriculum_habitat.perception.graph_builder import MetricGraphBuilder, SEM_DIR
from curriculum_habitat.perception.graph_encoder import MetricGraphEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB_DIR = REPO_ROOT / "scene_graphs" / "vocab"


def a_goal_instance(scene, category="chair"):
    objects = json.loads((SEM_DIR / f"{scene}.json").read_text())["objects"]
    return next(o for o in objects if o["category"].lower() == category)


def main():
    scenes = ["DoSbsoo4EAg", "GsQBY83r3hb", "H8rQCnvBgo6", "YHmAkqgwe2p"]
    vocab = json.loads((VOCAB_DIR / "categories.json").read_text())
    clip_table = torch.load(VOCAB_DIR / "clip_text_embeddings.pt")

    # Build one metric graph per scene (goal = a chair instance in that scene).
    graphs = []
    for scene in scenes:
        chair = a_goal_instance(scene, "chair")
        g = MetricGraphBuilder(scene, vocab=vocab).build(
            "chair", chair["position"], drop_instance_id=chair["instance_id"])
        graphs.append(g)
    batch = torch.from_numpy(np.stack(graphs)).float()  # [B,28,6]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = MetricGraphEncoder(clip_text_embeddings=clip_table, include_node_metric=True).to(device)
    emb = enc(batch.to(device))
    print(f"batch graph_flat {tuple(batch.shape)} -> embeddings {tuple(emb.shape)} on {device}")
    print("active nodes/scene:", [int(g[:, 1].sum()) for g in graphs])
    print("embedding norms:", [round(x, 3) for x in emb.norm(dim=-1).tolist()])

    # Same scene, two different goal categories -> embeddings should differ.
    scene = "DoSbsoo4EAg"
    b = MetricGraphBuilder(scene, vocab=vocab)
    chair = a_goal_instance(scene, "chair")
    bed = a_goal_instance(scene, "bed")
    g_chair = torch.from_numpy(b.build("chair", chair["position"])).float()[None].to(device)
    g_bed = torch.from_numpy(b.build("bed", bed["position"])).float()[None].to(device)
    delta = (enc(g_chair) - enc(g_bed)).norm().item()
    print(f"same scene, chair-goal vs bed-goal embedding distance: {delta:.3f} (should be > 0)")

    # Gradient flow.
    loss = enc(batch.to(device)).square().mean()
    loss.backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters()
             if p.requires_grad]
    print(f"backprop OK: {all(grads)} ({sum(grads)}/{len(grads)} tensors have finite grads)")
    print("trainable params:", sum(p.numel() for p in enc.parameters() if p.requires_grad))


if __name__ == "__main__":
    main()
