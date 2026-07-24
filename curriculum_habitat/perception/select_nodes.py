"""Stage 3 - pick the object instances that become graph nodes (30 per scene).

The metric graph tensor is [31, 6]; per episode the builder injects the goal as node 0,
so each scene contributes up to 30 *static context* objects here. This auto-selector keeps
only real furniture / appliances / fixtures (bed, table, chair, sofa, cabinet, toilet,
sink, tv, ...) and drops architectural surfaces and clutter (wall, ceiling, floor, carpet,
rug, curtain, pillow, towel, picture, mirror, decorations, small tableware, ...). It
guarantees >=1 representative of every goal category present, then fills with the most
salient furniture, spread evenly across rooms. Hand-edit the output as needed.

Output: scene_graphs/selection/<scene>.json
    {"scene", "node_instance_ids": [<=30 ids], "nodes": [{id, category, region_id, bbox_diag}]}

Run: python -m curriculum_habitat.perception.select_nodes
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEM_DIR = REPO_ROOT / "scene_graphs" / "semantics"
OUT_DIR = REPO_ROOT / "scene_graphs" / "selection"

NUM_CONTEXT = 30
MIN_BBOX_DIAG = 0.30   # drop specks (knobs, outlets, cutlery)
MAX_BBOX_DIAG = 5.00   # drop architectural leftovers / polluted instances spanning rooms

# Raw annotation label -> ObjectNav goal category (ensure a landmark per goal type).
GOAL_LABELS = {
    "chair": "chair", "armchair": "chair", "office chair": "chair", "rocking chair": "chair",
    "dining chair": "chair", "bar chair": "chair", "lounge chair": "chair",
    "bed": "bed",
    "plant": "plant", "potted plant": "plant", "decorative plant": "plant", "flowerpot": "plant",
    "toilet": "toilet",
    "tv": "tv_monitor", "television": "tv_monitor", "monitor": "tv_monitor", "tv monitor": "tv_monitor",
    "sofa": "sofa", "couch": "sofa", "sofa seat": "sofa", "sofa chair": "sofa",
}

# Substring tokens that mark non-furniture (no common furniture label contains these;
# note "robe" is avoided so it doesn't strike "wardrobe").
EXCLUDE_SUBSTR = (
    "wall", "ceiling", "floor", "curtain", "towel", "pillow", "cushion", "carpet", "rug",
    "blind", "vent", "outlet", "sensor", "detector", "sprinkler", "picture", "painting",
    "decorat", "mirror", "window", "shower", "railing", "handrail", "knob", "door",
    "faucet", "hanger", "stack of", "parapet", "soap", "cosmetic", "bag",
)
# Exact non-furniture labels (clutter / textiles / tableware / wall & ceiling mounted).
EXCLUDE_EXACT = {
    "unknown", "misc", "object", "objects", "mat", "doormat", "rail", "beam", "column",
    "pipe", "wire", "wires", "cable", "baseboard", "trim", "molding", "step", "stairs",
    "staircase", "threshold", "banister", "book", "books", "candle", "clock", "alarm",
    "alarm clock", "frame", "poster", "box", "basket", "bag", "bottle", "cup", "plate",
    "plates", "jar", "can", "cans", "tray", "cutlery", "knife", "kettle", "bowl", "pot",
    "pan", "soap", "liquid soap", "shampoo", "tissue", "tissue box", "toilet paper",
    "paper", "papers", "newspaper", "remote", "remote control", "phone", "telephone",
    "watch", "backpack", "clothes", "stack of clothes", "hanger", "hangers", "shoe",
    "shoes", "toy", "toys", "blanket", "duvet", "robe", "jacket", "vase", "flower",
    "figurine", "figure", "sculpture", "decoration", "switch", "socket", "fire extinguisher",
    "ceiling fan", "wall lamp", "ceiling lamp", "table cloth", "tablecloth", "cosmetic",
    "cosmetics", "weight", "case", "container", "candlestick", "light switch",
    "fuse panel", "folder", "toilet brush", "bathrobe", "ironing board", "barrel",
    "boiler", "plunger", "trashcan", "trash can", "bin", "waste bin", "dustbin",
    "garbage bin", "rubbish bin", "recycling bin", "fire alarm", "smoke detector",
    "tap", "bucket", "water tap",
}


def is_furniture(category):
    cat = category.strip().lower()
    if cat in EXCLUDE_EXACT:
        return False
    return not any(token in cat for token in EXCLUDE_SUBSTR)


def select_scene(record):
    pool = [o for o in record["objects"] if MIN_BBOX_DIAG <= o["bbox_diag"] <= MAX_BBOX_DIAG]
    chosen, chosen_ids = [], set()

    # 1) one representative (largest) per goal category present.
    per_goal = defaultdict(list)
    for o in pool:
        goal = GOAL_LABELS.get(o["category"].lower())
        if goal:
            per_goal[goal].append(o)
    for objs in per_goal.values():
        best = max(objs, key=lambda o: o["bbox_diag"])
        chosen.append(best)
        chosen_ids.add(best["instance_id"])

    # 2) fill with salient furniture, round-robin across rooms for spread.
    by_room = defaultdict(list)
    for o in pool:
        if o["instance_id"] in chosen_ids or not is_furniture(o["category"]):
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
        cats = ", ".join(sorted(n["category"] for n in selection["nodes"]))
        print(f"{record['scene']:14} {selection['num_nodes']:2} nodes | {cats}")
    print(f"\nWrote {OUT_DIR.relative_to(REPO_ROOT)}/<scene>.json  (edit these by hand as needed).")


if __name__ == "__main__":
    main()
