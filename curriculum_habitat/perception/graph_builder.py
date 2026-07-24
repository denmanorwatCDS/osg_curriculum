"""Stage 4 - build the metric `graph_flat[28, 6]` for a scene + episode.

Per node: [object_id, active, is_goal, x, y, z].

Layout produced here:
    node 0        = the episode GOAL (injected each reset; is_goal=1, exact position).
    nodes 1..22   = the scene's static selected context objects (active=1).
    nodes 23..27  = padding (active=0).

Coordinate convention: the extracted semantics store Habitat world coords (X, Y-up, Z).
The graph uses the GIROL convention x,y = floor plane, z = height, i.e.
    (x, y, z)_graph = (X, Z, Y)_habitat
so the encoder's x/y-axis edges live in the floor plane. is_goal is set per episode, so
the encoder stays agnostic to how goals are chosen.

Usage (offline / test):
    b = MetricGraphBuilder("DoSbsoo4EAg")
    graph_flat = b.build(target_category="chair", goal_world_xyz=[x, y, z])

Wiring into ObjRLNav.calculate_knowledge_graph: pass the current episode's target
category and the goal position (e.g. from get_closest_goal().position).
"""

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SEM_DIR = REPO_ROOT / "scene_graphs" / "semantics"
SEL_DIR = REPO_ROOT / "scene_graphs" / "selection"
VOCAB_PATH = REPO_ROOT / "scene_graphs" / "vocab" / "categories.json"

NUM_NODES = 31  # 1 injected goal + up to 30 static context objects
GOAL_SLOT = 0
UNKNOWN_ID = 0  # fallback object_id for categories missing from the vocab


def habitat_to_graph(xyz):
    """Habitat world (X, Y-up, Z) -> graph (x_floor, y_floor, z_up) = (X, Z, Y)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.array([xyz[0], xyz[2], xyz[1]])


class MetricGraphBuilder:
    def __init__(self, scene, vocab=None, num_nodes=NUM_NODES):
        self.scene = scene
        self.num_nodes = num_nodes
        self.vocab = vocab or json.loads(VOCAB_PATH.read_text())

        semantics = json.loads((SEM_DIR / f"{scene}.json").read_text())
        by_id = {o["instance_id"]: o for o in semantics["objects"]}
        selection = json.loads((SEL_DIR / f"{scene}.json").read_text())

        # Static context nodes: (object_id, is_goal=0, graph_xyz).
        self.context = []
        for instance_id in selection["node_instance_ids"][: num_nodes - 1]:
            obj = by_id[instance_id]
            self.context.append((
                self._object_id(obj["category"]),
                habitat_to_graph(obj["position"]),
                instance_id,
            ))

    def _object_id(self, category):
        return self.vocab.get(category.strip().lower(), UNKNOWN_ID)

    def build(self, target_category, goal_world_xyz, drop_instance_id=None):
        """Return graph_flat [num_nodes, 6] for this episode's goal."""
        graph = np.zeros((self.num_nodes, 6), dtype=np.float32)

        # node 0: the goal.
        graph[GOAL_SLOT, 0] = self._object_id(target_category)
        graph[GOAL_SLOT, 1] = 1.0  # active
        graph[GOAL_SLOT, 2] = 1.0  # is_goal
        graph[GOAL_SLOT, 3:6] = habitat_to_graph(goal_world_xyz)

        # nodes 1..: static context (optionally drop the instance that is the goal).
        slot = 1
        for object_id, graph_xyz, instance_id in self.context:
            if slot >= self.num_nodes:
                break
            if drop_instance_id is not None and instance_id == drop_instance_id:
                continue
            graph[slot, 0] = object_id
            graph[slot, 1] = 1.0
            graph[slot, 2] = 0.0
            graph[slot, 3:6] = graph_xyz
            slot += 1
        return graph

    def build_flat(self, target_category, goal_world_xyz, drop_instance_id=None):
        """graph_flat as a flat [num_nodes*6] vector (transport form)."""
        return self.build(target_category, goal_world_xyz, drop_instance_id).reshape(-1)


if __name__ == "__main__":
    # Offline demo: build a graph using a real goal-category instance as the goal.
    scene = "DoSbsoo4EAg"
    semantics = json.loads((SEM_DIR / f"{scene}.json").read_text())
    chair = next(o for o in semantics["objects"] if o["category"].lower() == "chair")
    builder = MetricGraphBuilder(scene)
    g = builder.build("chair", chair["position"], drop_instance_id=chair["instance_id"])
    active = int(g[:, 1].sum())
    print(f"scene {scene}: graph_flat {g.shape}, active nodes {active}, goal id {int(g[0,0])}")
    print("goal node (obj_id, active, is_goal, x, y, z):", np.round(g[0], 2).tolist())
    print("first 3 context nodes:\n", np.round(g[1:4], 2))
