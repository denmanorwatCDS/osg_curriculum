#!/usr/bin/env python3
"""Stage 1 - extract per-scene HM3D semantic info into JSON.

For every scene used by the curriculum config we dump, per annotated object instance:
    - category (raw HM3D label)
    - region_id (which room, HM3D's own region index)
    - 3D position (world-frame centroid) and axis-aligned bbox
    - source hex color + vertex count (diagnostics)

Data sources (no habitat-sim / GPU needed):
    - `<scene>.semantic.txt`  : instance_id, hex_color, "category", region_id
    - `<scene>.semantic.glb`  : texture-encoded instance segmentation (per-face color
                                == the instance's hex color). We sample the texture at
                                each vertex, match it to the palette, group vertices per
                                instance and compute a robust centroid + bbox.

Coordinate frame:
    The .glb asset is Z-up (dataset config: up=[0,0,1], front=[0,1,0]); Habitat world is
    Y-up. We convert asset -> world with  world = (x, z, -y). This was validated against
    the episodes' ground-truth goal positions (toilet goal matched to < 0.1 m).

Output: scene_graphs/semantics/<scene>.json
Run:    python -m curriculum_habitat.perception.extract_semantics
"""

import csv
import gzip
import io
import json
import os
from pathlib import Path

import numpy as np
import trimesh
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "homerobot_hm3d_objectnav.yaml"
OUT_DIR = REPO_ROOT / "scene_graphs" / "semantics"

# Vertices whose sampled texture color is farther than this (RGB euclidean) from every
# palette color are treated as seam/anti-aliasing noise and dropped.
COLOR_REJECT = 12.0
# Per instance, drop the farthest-from-median fraction of points before measuring the
# bbox, so a few stray mis-sampled texels cannot inflate an object's size.
ROBUST_TRIM_PCT = 95.0


def asset_to_world(points):
    """(N,3) asset-frame (Z-up) -> Habitat world-frame (Y-up):  world = (x, z, -y)."""
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])


def load_palette(txt_path):
    """Parse `<scene>.semantic.txt` -> per-instance (id, rgb, category, region_id)."""
    ids, rgb, cats, regions, hexes = [], [], [], [], []
    with open(txt_path) as fh:
        next(fh)  # "HM3D Semantic Annotations" header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = next(csv.reader(io.StringIO(line)))
            if len(parts) < 4:
                continue
            hx = parts[1].strip()
            ids.append(int(parts[0]))
            rgb.append([int(hx[i:i + 2], 16) for i in (0, 2, 4)])
            hexes.append(hx.upper())
            cats.append(parts[2].strip())
            regions.append(parts[3].strip())
    return ids, np.asarray(rgb, dtype=np.int64), cats, regions, hexes


def _nearest(colors, palette):
    """Assign each RGB color to the nearest palette index; return (idx, distance)."""
    try:
        from scipy.spatial import cKDTree
        dist, idx = cKDTree(palette).query(colors)
        return idx, dist
    except Exception:
        # Vectorized fallback (K is small ~ few hundred).
        d = np.linalg.norm(
            colors[:, None, :].astype(np.float64) - palette[None, :, :], axis=2
        )
        return d.argmin(1), d.min(1)


def sample_instance_points(glb_path, palette):
    """Return world-frame vertex arrays grouped by palette index -> {k: (M,3)}."""
    scene = trimesh.load(str(glb_path), process=False)
    meshes = scene.dump() if hasattr(scene, "dump") else [scene]  # bakes node transforms

    all_pts, all_idx = [], []
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
        py = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)  # GLB V axis is flipped
        colors = img[py, px]
        idx, dist = _nearest(colors, palette)
        keep = dist <= COLOR_REJECT
        if not keep.any():
            continue
        all_pts.append(asset_to_world(np.asarray(mesh.vertices)[keep]))
        all_idx.append(idx[keep])

    if not all_pts:
        return {}
    pts = np.vstack(all_pts)
    idx = np.concatenate(all_idx)
    return {int(k): pts[idx == k] for k in np.unique(idx)}


def robust_bbox(points):
    """Centroid + axis-aligned bbox after trimming stray points (world frame)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) > 8:
        median = np.median(points, axis=0)
        radial = np.linalg.norm(points - median, axis=1)
        points = points[radial <= np.percentile(radial, ROBUST_TRIM_PCT)]
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return points.mean(axis=0), (lo + hi) / 2.0, hi - lo


def resolve_scene_paths(data_config_path):
    """From the curriculum data config, yield (scene_hash, glb_rel, txt, glb) per scene.

    Scene directories are resolved from each content episode file's `scene_id`, i.e. the
    exact asset Habitat loads, so the extracted graph matches what training sees.
    """
    with open(data_config_path) as fh:
        params = yaml.safe_load(fh)
    metadata_path = REPO_ROOT / params["habitat_data"]["metadata_path"]
    scenes_dir = REPO_ROOT / params["habitat_data"]["scenes_dir"]
    content_dir = metadata_path.parent / "content"

    for scene in params["habitat_data"]["content_scenes"]:
        with gzip.open(content_dir / f"{scene}.json.gz", "rt") as fh:
            episodes = json.load(fh)["episodes"]
        scene_id = episodes[0]["scene_id"]  # e.g. hm3d_v0.2/train/00324-DoSbsoo4EAg/DoSbsoo4EAg.basis.glb
        glb = scenes_dir / scene_id
        scene_dir = glb.parent
        txt = scene_dir / f"{scene}.semantic.txt"
        sem_glb = scene_dir / f"{scene}.semantic.glb"
        if not txt.exists() or not sem_glb.exists():
            raise FileNotFoundError(f"missing semantic files for {scene} in {scene_dir}")
        yield scene, scene_id, txt, sem_glb


def goal_ground_truth(data_config_path):
    """{scene: [(object_id, world_pos), ...]} from the episode goal annotations.

    Goals carry the semantic `object_id`, so QA can compare each extracted instance to
    its own annotated position (not a same-category neighbour)."""
    with open(data_config_path) as fh:
        params = yaml.safe_load(fh)
    content_dir = (REPO_ROOT / params["habitat_data"]["metadata_path"]).parent / "content"
    out = {}
    for scene in params["habitat_data"]["content_scenes"]:
        with gzip.open(content_dir / f"{scene}.json.gz", "rt") as fh:
            data = json.load(fh)
        goals = []
        for entries in data.get("goals_by_category", {}).values():
            for entry in entries:
                if entry.get("object_id") is not None:
                    goals.append((entry["object_id"], np.asarray(entry["position"], float)))
        out[scene] = goals
    return out


def extract_scene(scene, scene_id, txt, sem_glb):
    ids, palette, cats, regions, hexes = load_palette(txt)
    groups = sample_instance_points(sem_glb, palette)

    objects = []
    region_counts = {}
    for k, instance_id in enumerate(ids):
        pts = groups.get(k)
        if pts is None or len(pts) == 0:
            continue  # annotated but carries no geometry in the mesh
        centroid, aabb_center, aabb_sizes = robust_bbox(pts)
        region = regions[k]
        region_counts[region] = region_counts.get(region, 0) + 1
        objects.append({
            "instance_id": instance_id,
            "category": cats[k],
            "region_id": region,
            "hex_color": hexes[k],
            "position": [round(float(v), 4) for v in centroid],
            "aabb": {
                "center": [round(float(v), 4) for v in aabb_center],
                "sizes": [round(float(v), 4) for v in aabb_sizes],
            },
            "num_vertices": int(len(pts)),
            "bbox_diag": round(float(np.linalg.norm(aabb_sizes)), 3),
        })

    return {
        "scene": scene,
        "scene_glb": scene_id,
        "up_axis": "y",
        "frame_note": "world = (asset_x, asset_z, -asset_y); Habitat Y-up world frame",
        "num_objects": len(objects),
        "regions": {r: {"num_objects": n} for r, n in sorted(region_counts.items())},
        "objects": objects,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = goal_ground_truth(DEFAULT_DATA_CONFIG)

    print(f"{'scene':14} {'objs':>5} {'rooms':>6}  goal-localization QA (per-instance vs episode goals)")
    for scene, scene_id, txt, sem_glb in resolve_scene_paths(DEFAULT_DATA_CONFIG):
        record = extract_scene(scene, scene_id, txt, sem_glb)
        (OUT_DIR / f"{scene}.json").write_text(json.dumps(record, indent=2))

        # QA: compare each extracted instance to its OWN annotated goal position (by id).
        by_id = {obj["instance_id"]: np.asarray(obj["position"]) for obj in record["objects"]}
        errs, missing = [], 0
        for object_id, gpos in gt.get(scene, []):
            pos = by_id.get(object_id)
            if pos is None:
                missing += 1
            else:
                errs.append(float(np.linalg.norm(pos - gpos)))
        qa = (f"median {np.median(errs):.2f}m max {max(errs):.2f}m "
              f"({len(errs)} goals, {missing} missing)") if errs else "no goal ids"
        print(f"{scene:14} {record['num_objects']:>5} {len(record['regions']):>6}  {qa}")

    print(f"\nWrote {OUT_DIR.relative_to(REPO_ROOT)}/<scene>.json for all scenes.")


if __name__ == "__main__":
    main()
