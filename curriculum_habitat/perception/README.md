# perception — metric scene graph + encoder (GIROL/HM3D)

Pre-computed metric scene graphs for the 10 HM3D scenes and an end-to-end-trainable
GATv2 encoder that turns them into a **128-d embedding** for the DDQN + orientation head.

## Data contract (what the encoder consumes)

```
graph_flat : [B, 28*6]   (or [B, 28, 6])
per node   : [object_id, active, is_goal, x, y, z]
```
- `object_id` indexes the CLIP vocabulary (`scene_graphs/vocab/categories.json`).
- `active` — node/padding mask.
- `is_goal` — marks the goal node (set per episode; encoder is agnostic to the policy).
- `x, y, z` — **floor plane = x,y; height = z** (GIROL convention). The builder converts
  from Habitat world `(X, Y-up, Z)` via `(x,y,z) = (X, Z, Y)`.

Node layout from `MetricGraphBuilder`: node 0 = injected episode goal, nodes 1..22 = the
scene's static selected context objects, nodes 23..27 = padding.

## Pipeline (offline, already run — outputs in `scene_graphs/`)

| stage | command | output |
|---|---|---|
| 1. extract | `python -m curriculum_habitat.perception.extract_semantics` | `semantics/<scene>.json` |
| 2. stats | `python -m curriculum_habitat.perception.room_stats` | `stats/<scene>.json` |
| 3. select | `python -m curriculum_habitat.perception.select_nodes` | `selection/<scene>.json` (22 ids, hand-editable) |
| — vocab | `python -m curriculum_habitat.perception.build_vocab` | `vocab/categories.json`, `clip_text_embeddings.pt` |

Stage 1 is pure trimesh+numpy (no habitat-sim/GPU). Positions were validated against 206
ground-truth episode goals: **median error 0.11 m, max 1.09 m, 0 missing**.

## Encoder API (what teammates integrate)

```python
import torch, json
from curriculum_habitat.perception.graph_encoder import MetricGraphEncoder

clip = torch.load("scene_graphs/vocab/clip_text_embeddings.pt")   # [V, 512], frozen
enc  = MetricGraphEncoder(clip_text_embeddings=clip, include_node_metric=True)
emb  = enc(graph_flat)          # [B, 28*6] or [B,28,6]  ->  [B, 128]
```
- `include_node_metric=True` → **metric** graph (#1). `False` → **flat non-metric** (#3);
  the only difference is whether the raw-xyz node block is fed in (edges keep distance in
  both). Same class, one flag — the tight metric-vs-non-metric control from `GRAPHS.md`.
- Edges: `edge_dim=10` = direction(6) + distance-bucket(4); goal-star topology, 2 parallel
  edges/object (x-axis + y-axis) + self-loops.
- CLIP table is a frozen buffer; everything else trains e2e (~231k params).

## Building a graph at episode time

```python
from curriculum_habitat.perception.graph_builder import MetricGraphBuilder
builder = MetricGraphBuilder(scene_hash)            # cache one per scene
graph_flat = builder.build_flat(target_category, goal_world_xyz)   # [28*6]
```

Wiring into `ObjRLNav.calculate_knowledge_graph` (replaces the current stub): keep a
`MetricGraphBuilder` for the episode's scene, and per step call `build` with the episode
target category and `get_closest_goal().position`. Then swap the `knowledge_graph`
observation-space box from `(0,)` to `(28*6,)`.

## Self-tests
```
python -m curriculum_habitat.perception.graph_encoder          # random-input unit check
python -m curriculum_habitat.perception.test_metric_pipeline   # real scenes, fwd+bwd
```
