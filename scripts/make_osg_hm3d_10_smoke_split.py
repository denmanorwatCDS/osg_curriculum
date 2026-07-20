from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

SRC_ROOT = Path.home() / "work/datasets/objectnav_hm3d_v2/train"
DST_ROOT = Path("data/scene_datasets/hm3d/val")

# The 10 selected HM3D train scenes.
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

def read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def write_json_gz(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

DST_ROOT.mkdir(parents=True, exist_ok=True)
(DST_ROOT / "content").mkdir(parents=True, exist_ok=True)

total = 0

for sid in SCENES:
    src = SRC_ROOT / "content" / f"{sid}.json.gz"
    assert src.exists(), f"Missing source content file: {src}"

    data = read_json_gz(src)
    eps = data.get("episodes", [])
    assert eps, f"No episodes in {src}"

    # For the current navigate_homerobot.py, make one episode per scene,
    # and force episode_id='1', because the script filters for episode_id == '1'.
    ep = deepcopy(eps[0])
    ep["episode_id"] = "1"

    data["episodes"] = [ep]

    out = DST_ROOT / "content" / f"{sid}.json.gz"
    write_json_gz(out, data)

    print(f"{sid}: wrote 1 episode")
    print("  scene_id:", ep.get("scene_id"))
    print("  object_category:", ep.get("object_category"))
    total += 1

# Metadata file.
src_meta = SRC_ROOT / "train.json.gz"
meta = read_json_gz(src_meta)
meta["episodes"] = []

# Helpful if this field exists in this dataset version.
if "content_scenes" in meta:
    meta["content_scenes"] = SCENES

write_json_gz(DST_ROOT / "val.json.gz", meta)

print()
print("DONE")
print("Total episodes:", total)
print("Wrote:", DST_ROOT / "val.json.gz")
print("Wrote content dir:", DST_ROOT / "content")