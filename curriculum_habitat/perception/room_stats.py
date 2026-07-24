#!/usr/bin/env python3
"""Stage 2 - per-scene / per-room object statistics from the extracted semantics.

Reads scene_graphs/semantics/<scene>.json (Stage 1 output) and produces, for each scene:
    - per-room category counts (how many tables / chairs / ... in each region)
    - a scene-level category total
    - a suggested object shortlist (goal categories + salient furniture, clutter dropped)

Writes scene_graphs/stats/<scene>.json and prints a readable table so you can decide
which instances become graph nodes (Stage 3).

Run: python -m curriculum_habitat.perception.room_stats
"""

import collections
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEM_DIR = REPO_ROOT / "scene_graphs" / "semantics"
OUT_DIR = REPO_ROOT / "scene_graphs" / "stats"

# The 6 HM3D ObjectNav goal categories (annotation-label spellings included).
GOAL_CATEGORIES = {"chair", "bed", "plant", "potted plant", "toilet", "tv", "tv_monitor",
                   "monitor", "sofa", "couch"}

# Architectural / non-navigational clutter excluded from the *suggested* shortlist
# (still present in the full counts).
STRUCTURE = {
    "wall", "ceiling", "floor", "unknown", "misc", "objects", "object", "door",
    "door frame", "doorframe", "window", "window frame", "windowsill", "window shutter",
    "vent", "ceiling vent", "air vent", "outlet", "light switch", "switch", "sensor",
    "rail", "railing", "handrail", "beam", "column", "pipe", "wire", "wires", "cable",
    "baseboard", "trim", "molding", "curtain rail", "shower wall", "step", "stairs",
    "staircase", "threshold", "ceiling lamp", "wall lamp", "ceiling light",
}


def load_scene(path):
    return json.loads(Path(path).read_text())


def build_stats(record):
    rooms = collections.defaultdict(collections.Counter)
    scene_totals = collections.Counter()
    for obj in record["objects"]:
        rooms[obj["region_id"]][obj["category"]] += 1
        scene_totals[obj["category"]] += 1
    return {
        "scene": record["scene"],
        "num_objects": record["num_objects"],
        "num_rooms": len(rooms),
        "scene_totals": dict(scene_totals.most_common()),
        "rooms": {r: dict(c.most_common()) for r, c in sorted(rooms.items())},
    }


def suggest_shortlist(record, per_room_cap=6):
    """A first-pass node suggestion: every goal-category instance + the largest
    non-clutter objects per room. You edit this by hand in Stage 3."""
    keep = []
    by_room = collections.defaultdict(list)
    for obj in record["objects"]:
        by_room[obj["region_id"]].append(obj)
    for objs in by_room.values():
        goals = [o for o in objs if o["category"].lower() in GOAL_CATEGORIES]
        furniture = sorted(
            (o for o in objs if o["category"].lower() not in STRUCTURE
             and o["category"].lower() not in GOAL_CATEGORIES),
            key=lambda o: o["bbox_diag"], reverse=True,
        )
        chosen = goals + furniture[:per_room_cap]
        keep.extend(o["instance_id"] for o in chosen)
    return sorted(set(keep))


def print_scene(stats, record):
    print(f"\n{'='*70}\n{stats['scene']}   "
          f"{stats['num_objects']} objects across {stats['num_rooms']} rooms")
    goals = {k: v for k, v in stats["scene_totals"].items()
             if k.lower() in GOAL_CATEGORIES}
    print("  goal-category objects present:",
          ", ".join(f"{k}x{v}" for k, v in goals.items()) or "(none)")
    for room, counts in stats["rooms"].items():
        # Show meaningful furniture first, hide pure structure in the tail.
        furniture = {k: v for k, v in counts.items() if k.lower() not in STRUCTURE}
        line = ", ".join(f"{k}x{v}" for k, v in list(furniture.items())[:12])
        n = sum(counts.values())
        print(f"  room {room:>3} ({n:>3} objs): {line}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(SEM_DIR.glob("*.json"))
    if not paths:
        raise SystemExit(f"No semantics found in {SEM_DIR}. Run extract_semantics first.")
    for path in paths:
        record = load_scene(path)
        stats = build_stats(record)
        stats["suggested_node_instance_ids"] = suggest_shortlist(record)
        (OUT_DIR / f"{record['scene']}.json").write_text(json.dumps(stats, indent=2))
        print_scene(stats, record)
    print(f"\nWrote {OUT_DIR.relative_to(REPO_ROOT)}/<scene>.json (with suggested shortlist).")


if __name__ == "__main__":
    main()
