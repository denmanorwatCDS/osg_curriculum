"""Visual correctness check for the metric scene graph.

Renders, per scene, a top-down (X-Z floor plane) figure that overlays:
    - the scene footprint (semantic-mesh vertices) COLOURED BY ROOM (region),
    - room-id labels at each region centroid,
    - the 22 selected context nodes (white circles, labelled),
    - the injected goal node (star),
    - the goal-star edges, coloured by distance bucket,
plus an edge table listing, for each node, the direction (left/right, front/behind) and
distance bucket EXACTLY as the encoder computes them - so the picture and the encoded
edge features can be checked against each other.

Run:  python -m curriculum_habitat.perception.visualize_metric_graph            # all scenes
      python -m curriculum_habitat.perception.visualize_metric_graph DoSbsoo4EAg chair
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from curriculum_habitat.perception.extract_semantics import (
    COLOR_REJECT, _nearest, asset_to_world, load_palette,
)
from curriculum_habitat.perception.graph_builder import MetricGraphBuilder, SEM_DIR
from curriculum_habitat.perception.graph_encoder import ALIGN_THRESHOLD, DIST_BOUNDS

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = REPO_ROOT / "scene_graphs" / "vocab" / "categories.json"
TRAIN_DIR = REPO_ROOT / "data/scene_datasets/hm3d_v0.2/train"
OUT_DIR = REPO_ROOT / "scene_graphs" / "viz"
BUCKET_COLORS = ["#2ca02c", "#ffbf00", "#ff7f0e", "#d62728"]  # <5, <10, <15, >=15 m
BUCKET_LABELS = ["<5m", "<10m", "<15m", ">=15m"]


def bucket(distance):
    return int(np.searchsorted(DIST_BOUNDS, distance, side="right"))


def axis_relation(delta, axis):
    """Mirror of MetricGraphEncoder._axis_edge_attr for one floor axis (delta = node-goal)."""
    if abs(delta) < ALIGN_THRESHOLD:
        return "aligned"
    if axis == "x":
        return "right" if delta > 0 else "left"
    return "front" if delta > 0 else "behind"


def mesh_footprint_by_region(scene):
    """Per-vertex top-down (X, Z) points of the semantic mesh + each vertex's region id.

    Uses the same texture->instance->region assignment as extract_semantics, so the
    footprint is coloured by the exact rooms the graph uses. Returns (None, None) if the
    mesh is unavailable."""
    glbs = list(TRAIN_DIR.glob(f"*{scene}*/{scene}.semantic.glb"))
    txts = list(TRAIN_DIR.glob(f"*{scene}*/{scene}.semantic.txt"))
    if not glbs or not txts:
        return None, None
    import trimesh
    _, palette, _, regions, _ = load_palette(txts[0])
    regions = np.array(regions)
    meshes = trimesh.load(str(glbs[0]), process=False)
    meshes = meshes.dump() if hasattr(meshes, "dump") else [meshes]

    xz, reg = [], []
    for mesh in meshes:
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        texture = getattr(material, "baseColorTexture", None) if material else None
        if uv is None or texture is None:
            continue
        img = np.asarray(texture.convert("RGB"))
        h, w = img.shape[:2]
        uv = np.asarray(uv)
        px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        py = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)
        idx, dist = _nearest(img[py, px], palette)
        keep = dist <= COLOR_REJECT
        world = asset_to_world(np.asarray(mesh.vertices)[keep])
        xz.append(world[:, [0, 2]])
        reg.append(regions[idx[keep]])
    if not xz:
        return None, None
    xz, reg = np.vstack(xz), np.concatenate(reg)
    if len(xz) > 90000:
        sel = np.random.choice(len(xz), 90000, replace=False)
        xz, reg = xz[sel], reg[sel]
    return xz, reg


def region_colors(regions):
    regions = sorted(regions)
    if len(regions) <= 10:
        cmap, order = plt.get_cmap("tab10"), list(range(10))
    else:  # interleave even/odd so adjacent room ids don't get similar tab20 shades.
        cmap, order = plt.get_cmap("tab20"), list(range(0, 20, 2)) + list(range(1, 20, 2))
    return {r: cmap(order[i % len(order)]) for i, r in enumerate(regions)}


def visualize(scene, goal_category="chair"):
    semantics = json.loads((SEM_DIR / f"{scene}.json").read_text())
    vocab = json.loads(VOCAB_PATH.read_text())
    id2cat = {v: k for k, v in vocab.items()}
    selection = json.loads((REPO_ROOT / f"scene_graphs/selection/{scene}.json").read_text())

    goal_obj = next((o for o in semantics["objects"] if o["category"].lower() == goal_category),
                    max(semantics["objects"], key=lambda o: o["bbox_diag"]))
    goal_category = goal_obj["category"]
    graph = MetricGraphBuilder(scene, vocab=vocab).build(
        goal_category, goal_obj["position"], drop_instance_id=goal_obj["instance_id"])
    goal_xy = graph[0, 3:5]
    nodes = graph[1:][graph[1:, 1] > 0]

    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(19, 9), gridspec_kw={"width_ratios": [2.4, 1]})

    # footprint coloured by room + room-id labels at region centroids.
    fp_xz, fp_reg = mesh_footprint_by_region(scene)
    all_regions = {o["region_id"] for o in semantics["objects"]}
    if fp_reg is not None:
        all_regions |= set(fp_reg.tolist())
    rcolors = region_colors(all_regions)
    if fp_xz is not None:
        ax.scatter(fp_xz[:, 0], fp_xz[:, 1], s=8, c=[rcolors[r] for r in fp_reg],
                   alpha=0.25, linewidths=0, zorder=0)
        for r in sorted(set(fp_reg.tolist())):
            mask = fp_reg == r
            if mask.sum() < 200:
                continue
            ax.text(fp_xz[mask, 0].mean(), fp_xz[mask, 1].mean(), f"R{r}",
                    fontsize=11, fontweight="bold", ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=rcolors[r], alpha=0.75))

    # object positions as small dark dots (rooms already shown by the footprint).
    for o in semantics["objects"]:
        x, _, z = o["position"]
        ax.scatter(x, z, s=6, color="0.25", alpha=0.30, linewidths=0, zorder=1)

    # goal-star edges + selected nodes + edge table.
    rows = []
    for node in nodes:
        nx, ny = node[3], node[4]
        dx, dy = nx - goal_xy[0], ny - goal_xy[1]
        dist = float(np.hypot(dx, dy))
        ax.plot([goal_xy[0], nx], [goal_xy[1], ny], color=BUCKET_COLORS[bucket(dist)],
                alpha=0.7, lw=1.4, zorder=2)
        cat = id2cat.get(int(node[0]), "?")
        ax.scatter(nx, ny, s=70, color="white", edgecolors="black", linewidths=1.2, zorder=4)
        ax.annotate(cat, (nx, ny), fontsize=7, xytext=(3, 3), textcoords="offset points", zorder=5)
        rows.append([cat[:16], f"{dist:.1f}",
                     f"{axis_relation(dx,'x')}/{BUCKET_LABELS[bucket(abs(dx))]}",
                     f"{axis_relation(dy,'y')}/{BUCKET_LABELS[bucket(abs(dy))]}"])

    ax.scatter(*goal_xy, marker="*", s=520, color="red", edgecolors="black",
               linewidths=1.2, zorder=6)

    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    ax.annotate("", xy=(x0 + 1.5, y0 + 0.4), xytext=(x0 + 0.4, y0 + 0.4),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(x0 + 1.6, y0 + 0.4, "+x (right)", fontsize=8, va="center")
    ax.annotate("", xy=(x0 + 0.4, y0 + 1.5), xytext=(x0 + 0.4, y0 + 0.4),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(x0 + 0.4, y0 + 1.65, "+y (front)", fontsize=8, ha="center")

    ax.set_aspect("equal")
    ax.set_xlabel("world X (m)"); ax.set_ylabel("world Z (m)")
    ax.set_title(f"{scene} - metric graph  |  {len(nodes)} context + goal  "
                 f"|  {semantics['num_objects']} objects, {len(semantics['regions'])} rooms")
    handles = [plt.Line2D([], [], color=c, lw=3, label=l) for c, l in zip(BUCKET_COLORS, BUCKET_LABELS)]
    handles.append(plt.Line2D([], [], marker="*", color="red", ls="", label=f"goal ({goal_category})"))
    ax.legend(handles=handles, title="edge distance / goal", loc="upper right", fontsize=8)

    ax_t.axis("off")
    table = ax_t.table(cellText=rows, colLabels=["node", "dist(m)", "x: dir/bucket", "y: dir/bucket"],
                       loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 1.25)
    ax_t.set_title("goal-star edge features (as encoded)", fontsize=10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{scene}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    if len(sys.argv) >= 2:
        print("wrote", visualize(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else "chair"))
        return
    import yaml
    scenes = yaml.safe_load(open(REPO_ROOT / "configs/homerobot_hm3d_objectnav.yaml"))["habitat_data"]["content_scenes"]
    for scene in scenes:
        print("wrote", visualize(scene))


if __name__ == "__main__":
    main()
