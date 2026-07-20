# OSG ObjectNav eval — 10-scene HM3D replication guide

This reproduces the OSG (OpenSearch) object-goal-navigation evaluation on a fixed
**10-scene / 100-episode HM3D subset** (goal-diverse: ~5–6 object categories per
scene). You do **not** need the full HM3D dataset — a ~360 MB data pack is provided.

---

## 1. Get the code + submodules

```bash
git clone git@github.com:mrizo-maruf/osg_eval.git
cd osg_eval
git submodule update --init Grounded-Segment-Anything
cd Grounded-Segment-Anything && git submodule update --init Tag2Text && cd ..
# home_robot submodule (needed by the FMM controller)
git submodule update --init home-robot
```

## 2. Build the conda environment

Follow the install steps in the main [README.md](README.md) (Simulation section):
Python 3.9, PyTorch, `Grounded-Segment-Anything` (GroundingDINO + Tag2Text/RAM),
LAVIS (BLIP-2), `habitat-sim==0.2.5`, `home_robot` + `home_robot_sim`, `scikit-fmm`.

Two environments are useful depending on how you run perception:

| Env (name it what you like) | Purpose | Notes |
|---|---|---|
| `nav` (CUDA 11.8) | GT semantics, or GroundingDINO on **CPU** | GroundingDINO can't run on GPU here for Blackwell/sm_120 GPUs (NVRTC can't JIT sm_120) |
| `nav_cu128` (CUDA 12.8, torch 2.8) | GroundingDINO on **GPU** | Needed for a new (sm_120) GPU; on older GPUs plain `nav` runs GDINO on GPU fine |

> If your GPU is **not** RTX 40xx/50xx-new, a single CUDA-11.8 `nav` env runs
> everything on GPU and you can ignore the cu128 env.

## 3. Model checkpoints (not in git, ~6.3 GB)

```bash
mkdir -p checkpoints && cd checkpoints
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
BLIP-2 weights download automatically from HuggingFace on the first run.

> RAM + GroundingDINO are loaded at startup **even in GT mode**, so both files are required.

## 4. OpenAI key

The pipeline uses `gpt-3.5-turbo`. Create `configs/openai_api_key.yaml`:
```yaml
api_key: sk-YOUR_OWN_KEY
```
Budget note: the pipeline makes **~120 GPT calls per episode** (~12k for the full 100).

## 5. Data pack — the 10 scenes (no full HM3D needed)

You'll receive `osg_10scene_data.zip` (~360 MB, sent separately — it contains
Matterport HM3D meshes, so you must have accepted the **HM3D license**). Unpack it
into `data/scene_datasets/` so it lands at habitat's default discovery path
(this is required for HM3D **semantics** to load):

```bash
mkdir -p data/scene_datasets logs
cd data/scene_datasets && unzip /path/to/osg_10scene_data.zip && cd ../..
# result:
#   data/scene_datasets/hm3d/val/{val.json.gz, content/<10 scenes>.json.gz}   (episodes)
#   data/scene_datasets/hm3d_v0.2/{*.scene_dataset_config.json, train/00XXX-<scene>/...}
```

Verify it loads (should print `episodes: 100 | semantic objs: <N>`):
```bash
conda activate nav
python - <<'PY'
import habitat
from utils.habitat_utils import setup_env_config, ObjNavEnv
cfg = setup_env_config("configs/homerobot_hm3d_objectnav.yaml",
                       "configs/objectnav_hm3d_v2_with_semantic.yaml")
env = ObjNavEnv(habitat.Env(config=cfg), cfg); env.reset()
print("episodes:", env.env.number_of_episodes,
      "| semantic objs:", len(env.env.sim.semantic_annotations().objects))
env.close()
PY
```

> Do **not** regenerate the split — `scripts/make_osg_hm3d_diverse_100_split.py`
> needs the full HM3D train episodes. The provided split *is* the fixed eval set.

---

## 6. Run the evaluation

The runner is `navigate_homerobot.py`, driven by env vars:

| Var | Default | Meaning |
|---|---|---|
| `OSG_TEST_EPISODES` | 10 | number of episodes to run (use **100** for the full set) |
| `OSG_SKIP_EPISODES` | 0 | resume: skip N already-completed episodes |
| `OSG_GT_SEM` | 1 | `1` = ground-truth semantics; `0` = **GroundingDINO** perception |
| `OSG_GDINO_DEVICE` | cuda | set `cpu` to run RAM+GroundingDINO on CPU (sm_120 fallback) |
| `OSG_NO_CV2_VIS` | 0 | set `1` to disable the OpenCV window (**headless, no DISPLAY needed**) |
| `OSG_LATENCY_LOG` | logs/latency.jsonl | per-episode latency output |
| `OSG_PROFILE` | 1 | set `0` to disable latency profiling |

Always create `logs/` first (`mkdir -p logs`). Recommended flags: `MAGNUM_LOG=quiet GLOG_minloglevel=2`.

### Mode A — GT semantics (oracle perception; fastest)
```bash
conda activate nav
mkdir -p logs
OSG_GT_SEM=1 OSG_NO_CV2_VIS=1 OSG_TEST_EPISODES=100 \
OSG_LATENCY_LOG=logs/latency_gt.jsonl MAGNUM_LOG=quiet GLOG_minloglevel=2 \
python -u navigate_homerobot.py > logs/osg_gt.log 2>&1
```

### Mode B — GroundingDINO on GPU (real perception)
```bash
conda activate nav_cu128        # env where GroundingDINO's custom CUDA op loads
python scripts/check_gdino_device.py     # sanity: should say "running on GPU (cuda)"
mkdir -p logs
OSG_GT_SEM=0 OSG_NO_CV2_VIS=1 OSG_TEST_EPISODES=100 \
OSG_LATENCY_LOG=logs/latency_gdino_gpu.jsonl MAGNUM_LOG=quiet GLOG_minloglevel=2 \
python -u navigate_homerobot.py > logs/osg_gdino_gpu.log 2>&1
```
(`scripts/run_gdino_gpu_eval.sh` does this too, but it hardcodes a conda path/env
name — edit those two lines or just use the command above.)

### Mode C — GroundingDINO on CPU (real perception, no cu128 env; slow)
```bash
conda activate nav
OSG_GT_SEM=0 OSG_GDINO_DEVICE=cpu OSG_NO_CV2_VIS=1 OSG_TEST_EPISODES=100 \
OSG_LATENCY_LOG=logs/latency_gdino_cpu.jsonl MAGNUM_LOG=quiet GLOG_minloglevel=2 \
python -u navigate_homerobot.py > logs/osg_gdino_cpu.log 2>&1
```

For a long/overnight run, wrap it so it survives logout:
```bash
nohup setsid bash -c '<the command above>' > logs/run.log 2>&1 &
```

**Resume** after a stop (per-episode crashes don't kill the run; a machine reboot would):
```bash
DONE=$(grep -c EPISODE_TIME_SEC logs/osg_gdino_gpu.log)
OSG_SKIP_EPISODES=$DONE OSG_TEST_EPISODES=$((100-DONE)) ...<same other vars>... \
python -u navigate_homerobot.py >> logs/osg_gdino_gpu.log 2>&1
```

---

## 7. Aggregate the results

```bash
python scripts/aggregate_osg_eval.py \
  --logs logs/osg_gdino_gpu.log \
  --latency logs/latency_gdino_gpu.jsonl \
  --csv logs/summary.csv
```
Pass multiple `--logs` files if you resumed. Output:

- **Navigation** (crashes excluded): overall + per-scene + per-goal `success` /
  `spl` / `soft_spl` / `distance_to_goal`. Success uses a **1.0 m** threshold
  (`success_distance` in `configs/objectnav_hm3d_v2_with_semantic.yaml`).
- **Latency**: `episode_total`, and per emitted action —
  `obs_to_action_reasoning` (steps that ran BLIP/GPT/planner, slow) vs
  `obs_to_action_control` (pure FMM control steps, fast) vs `obs_to_action_no_viz`
  (blended, minus the debug render) — plus per-submodule means (`llm_api`, `vqa`,
  `detect`, `mapper_update`, `controller_step`, …).

---

## Notes / gotchas

- **GT vs GroundingDINO are different, non-comparable settings.** GT = perfect
  perception (isolates planning/control); GroundingDINO = real perception error
  included (harder). Report them separately, and keep `success_distance` fixed at
  **1.0 m** across any comparison.
- **sm_120 GPUs:** GroundingDINO's custom CUDA op won't JIT on a CUDA-11.8 stack
  (`nvrtc: invalid --gpu-architecture`). Use the cu128 env (Mode B), or run the
  detector on CPU (Mode C). `python scripts/check_gdino_device.py` tells you which.
- **Episodes are covered scene-by-scene** — a partial run only touches the first
  scene(s); run all 100 to span the 10 scenes.
- Per-episode crashes are caught (`[EPISODE ERROR]` in the log) and excluded from
  metrics; the run continues.
