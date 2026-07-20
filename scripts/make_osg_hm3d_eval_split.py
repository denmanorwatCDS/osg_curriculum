from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path
import os

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

N_PER_SCENE = int(os.environ.get("OSG_N_PER_SCENE", "20"))
START = int(os.environ.get("OSG_START_EP", "1000"))

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
    data = read_json_gz(src)

    eps = deepcopy(data.get("episodes", [])[START:START + N_PER_SCENE])
    if len(eps) != N_PER_SCENE:
        raise RuntimeError(f"{sid}: expected {N_PER_SCENE}, got {len(eps)}")

    data["episodes"] = eps
    out = DST_ROOT / "content" / f"{sid}.json.gz"
    write_json_gz(out, data)

    print(f"{sid}: wrote {len(eps)} episodes")
    total += len(eps)

meta = read_json_gz(SRC_ROOT / "train.json.gz")
meta["episodes"] = []
if "content_scenes" in meta:
    meta["content_scenes"] = SCENES

write_json_gz(DST_ROOT / "val.json.gz", meta)

print("DONE")
print("Total episodes:", total)
print("Output:", DST_ROOT)
