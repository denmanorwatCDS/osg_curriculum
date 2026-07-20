from __future__ import annotations

import gzip
import json
import random
from copy import deepcopy
from collections import defaultdict, Counter
from pathlib import Path

SRC_ROOT = Path.home() / "work/datasets/objectnav_hm3d_v2/train"
DST_ROOT = Path("data/scene_datasets/hm3d/val")

SCENES = [
    "oPj9qMxrDEa",
    "RaYrxWt5pR1",
    "741Fdj7NLF9",
    "DoSbsoo4EAg",
    "H8rQCnvBgo6",
    "nGhNxKrgBPb",
    "QVAA6zecMHu",
    "GsQBY83r3hb",
    "DNWbUAJYsPy",
    "YHmAkqgwe2p",
]

N_PER_SCENE = 10
SEED = 17


def read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json_gz(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def sample_diverse_episodes(episodes, n: int, scene_seed: int):
    """
    Goal-diverse sampler:
    1. Group episodes by object_category.
    2. Shuffle episodes inside each category.
    3. Round-robin categories until n episodes are selected.
    """
    rng = random.Random(scene_seed)

    by_cat = defaultdict(list)
    for ep in episodes:
        cat = ep.get("object_category")
        if cat is None:
            continue
        by_cat[cat].append(ep)

    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    cats = sorted(by_cat.keys())
    rng.shuffle(cats)

    selected = []
    round_idx = 0

    while len(selected) < n:
        added_this_round = False

        for cat in cats:
            if len(selected) >= n:
                break
            if round_idx < len(by_cat[cat]):
                selected.append(deepcopy(by_cat[cat][round_idx]))
                added_this_round = True

        if not added_this_round:
            break

        round_idx += 1

    if len(selected) < n:
        raise RuntimeError(f"Could only sample {len(selected)} episodes, needed {n}")

    return selected


DST_ROOT.mkdir(parents=True, exist_ok=True)
(DST_ROOT / "content").mkdir(parents=True, exist_ok=True)

total = 0

print("Creating diverse OSG HM3D eval split")
print("Episodes per scene:", N_PER_SCENE)
print("Seed:", SEED)
print()

for scene_idx, sid in enumerate(SCENES):
    src = SRC_ROOT / "content" / f"{sid}.json.gz"
    if not src.exists():
        raise FileNotFoundError(src)

    data = read_json_gz(src)
    episodes = data.get("episodes", [])

    full_counts = Counter(ep.get("object_category") for ep in episodes)
    selected = sample_diverse_episodes(
        episodes,
        n=N_PER_SCENE,
        scene_seed=SEED + scene_idx,
    )

    selected_counts = Counter(ep.get("object_category") for ep in selected)

    out_data = deepcopy(data)
    out_data["episodes"] = selected

    out = DST_ROOT / "content" / f"{sid}.json.gz"
    write_json_gz(out, out_data)

    total += len(selected)

    print(sid)
    print("  total source episodes:", len(episodes))
    print("  available categories:", dict(sorted(full_counts.items())))
    print("  selected categories:", dict(sorted(selected_counts.items())))
    print("  selected episode ids:", [ep.get("episode_id") for ep in selected])
    print()

meta = read_json_gz(SRC_ROOT / "train.json.gz")
meta["episodes"] = []
if "content_scenes" in meta:
    meta["content_scenes"] = SCENES

write_json_gz(DST_ROOT / "val.json.gz", meta)

print("DONE")
print("Total selected episodes:", total)
print("Output:", DST_ROOT)
