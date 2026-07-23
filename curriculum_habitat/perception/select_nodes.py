"""Stage 3 - pick the object instances that become graph nodes (22 per scene).

The metric graph tensor is [28, 6]; per episode the builder injects the goal as one node,
so each scene contributes 22 *static context* objects here. This auto-selector produces a
sensible default that you can hand-edit: it guarantees >=1 representative of every goal
category present, then fills up with the most salient non-clutter furniture, spread evenly
across rooms.

Output: scene_graphs/selection/<scene>.json
    {"scene", "node_instance_ids": [22 ids], "nodes": [{id, category, region_id, bbox_diag}]}

Run: python -m curriculum_habitat.perception.select_nodes
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEM_DIR = REPO_ROOT / "scene_graphs" / "semantics"
OUT_DIR = REPO_ROOT / "scene_graphs" / "selection"

NUM_CONTEXT = 22
MIN_BBOX_DIAG = 0.25  # drop tiny specks (knobs, outlets, switches)

# Raw annotation label -> ObjectNav goal category (so we ensure a landmark per goal type).
GOAL_LABELS = {
    "chair": "chair", "armchair": "chair", "office chair": "chair", "rocking chair": "chair",
    "dining chair": "chair", "bar chair": "chair", "lounge chair": "chair", "stool": "chair",
    "bed": "bed",
    "plant": "plant", "potted plant": "plant", "decorative plant": "plant", "flowerpot": "plant",
    "flower": "plant", "vase": "plant",
    "toilet": "toilet",
    "tv": "tv_monitor", "television": "tv_monitor", "monitor": "tv_monitor", "tv monitor": "tv_monitor",
    "sofa": "sofa", "couch": "sofa", "sofa seat": "sofa", "sofa chair": "sofa",
}

STRUCTURE = {
    "wall", "ceiling", "floor", "unknown", "misc", "objects", "object", "door", "door frame",
    "doorframe", "window", "window frame", "windowsill", "window shutter", "vent", "ceiling vent",
    "air vent", "outlet", "light switch", "switch", "sensor", "rail", "railing", "handrail",
    "beam", "column", "pipe", "wire", "wires", "cable", "baseboard", "trim", "molding",
    "curtain rail", "curtain rod", "curtain bar", "shower wall", "bath wall", "step", "stairs",
    "staircase", "threshold", "knob", "door knob", "wall board", "smoke detector",
    "motion detector", "fire detector", "fire sprinkler", "fire alarm", "banister",
    "bedroom ceiling", "parapet",
}


def is_structure(cat):
    return cat.lower() in STRUCTURE


def select_scene(record):
    objects = [o for o in record["objects"] if o["bbox_diag"] >= MIN_BBOX_DIAG]
    chosen, chosen_ids = [], set()

    # 1) one representative (largest) per goal category present.
    per_goal = defaultdict(list)
    for o in objects:
        goal = GOAL_LABELS.get(o["category"].lower())
        if goal:
            per_goal[goal].append(o)
    for goal, objs in per_goal.items():
        best = max(objs, key=lambda o: o["bbox_diag"])
        chosen.append(best)
        chosen_ids.add(best["instance_id"])

    # 2) fill with salient non-clutter furniture, round-robin across rooms for spread.
    by_room = defaultdict(list)
    for o in objects:
        if o["instance_id"] in chosen_ids or is_structure(o["category"]):
            continue
        by_room[o["region_id"]].append(o)
    for room in by_room.values():
        room.sort(key=lambda o: o["bbox_diag"], reverse=True)
    rooms = sorted(by_room, key=lambda r: -len(by_room[r]))
    cursor = {r: 0 for r in rooms}
    while len(chosen) < NUM_CONTEXT and any(cursor[r] < len(by_room[r]) for r in rooms):
        for r in rooms:
            if cursor[r] < len(by_room[r]):
                o = by_room[r][cursor[r]]
                cursor[r] += 1
                if o["instance_id"] not in chosen_ids:
                    chosen.append(o)
                    chosen_ids.add(o["instance_id"])
            if len(chosen) >= NUM_CONTEXT:
                break

    chosen = chosen[:NUM_CONTEXT]
    return {
        "scene": record["scene"],
        "num_nodes": len(chosen),
        "node_instance_ids": [o["instance_id"] for o in chosen],
        "nodes": [{"instance_id": o["instance_id"], "category": o["category"],
                   "region_id": o["region_id"], "bbox_diag": o["bbox_diag"]} for o in chosen],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(SEM_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        selection = select_scene(record)
        (OUT_DIR / f"{record['scene']}.json").write_text(json.dumps(selection, indent=2))
        cats = ", ".join(sorted({n["category"] for n in selection["nodes"]}))
        print(f"{record['scene']:14} {selection['num_nodes']:2} nodes | {cats}")
    print(f"\nWrote {OUT_DIR.relative_to(REPO_ROOT)}/<scene>.json  (edit these by hand as needed).")


if __name__ == "__main__":
    main()
