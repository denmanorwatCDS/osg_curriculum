# OSG-Eval Installation Guide
> System: Ubuntu 22.04, NVIDIA RTX 4060 Laptop (Optimus), CUDA 12.8 driver, Python 3.9

---

## Prerequisites

- Miniconda installed
- CUDA 12.8 driver (`nvidia-smi` works)
- X display available (e.g. `DISPLAY=:1`)

---

## Step 1 — Clone repo and init submodules

```bash
git clone <repo_url> osg_eval && cd osg_eval
git submodule update --init Grounded-Segment-Anything
cd Grounded-Segment-Anything && git submodule update --init Tag2Text && cd ..
git submodule update --init home-robot
```

---

## Step 2 — Create conda environment

```bash
conda create -n nav python=3.9 -y
conda activate nav
```

---

## Step 3 — Install PyTorch cu118

> **Do NOT use `conda install pytorch pytorch-cuda=11.8`** — conda solver fails on
> CUDA 12.8-only systems (can't find `cuda-cudart >=11.8,<12.0`). Use pip instead.

```bash
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 4 — Install LAVIS (BLIP-2) and openai

> Pre-pin spacy to 3.7.6 first: latest spacy requires `thinc>=8.3.12` which needs
> Python ≥ 3.10, breaking the Python 3.9 env.

```bash
pip install "spacy==3.7.6"
pip install salesforce-lavis openai
```

---

## Step 5 — Install GroundingDINO (no CUDA extension)

> The CUDA extension can't compile: cu118 PyTorch + CUDA 12.8 nvcc mismatch.
> `CUDA_VISIBLE_DEVICES=""` makes `torch.cuda.is_available()` return False,
> which forces setup.py into the CPU-only build path.
> Note: glvnd's `_find_cuda_home()` falls back to `/usr/local/cuda` even when
> `CUDA_HOME=""`, so PATH-stripping alone is insufficient — `CUDA_VISIBLE_DEVICES`
> is the reliable switch.

```bash
CUDA_VISIBLE_DEVICES="" pip install -e Grounded-Segment-Anything/GroundingDINO \
    --no-build-isolation
```

---

## Step 6 — Install Tag2Text and fix transformers

```bash
pip install -r Grounded-Segment-Anything/Tag2Text/requirements.txt
pip install transformers==4.26.1
```

---

## Step 7 — Install torch-cluster and torch-geometric

> Same mismatch issue as GroundingDINO — use `CUDA_VISIBLE_DEVICES=""`.
> Use pip instead of `conda install pytorch-cluster -c pyg` (conda solver fails).

```bash
CUDA_VISIBLE_DEVICES="" pip install torch-cluster torch-geometric --no-build-isolation
```

---

## Step 8 — Install habitat-sim (WITHOUT headless)

> The `headless` EGL build crashes on Optimus laptops with driver 570.x:
> `GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue`
> Root cause: EGL surfaceless context fails on Optimus (Intel+NVIDIA).
> Fix: use the non-headless build (GLX/X11) with the real display.

```bash
conda install -n nav habitat-sim=0.2.5 withbullet -c conda-forge -c aihabitat -y
```

---

## Step 9 — Install home_robot

```bash
pip install -e home-robot/src/home_robot
```

---

## Step 10 — Fetch habitat-lab submodule and install

```bash
cd home-robot
git submodule update --init --recursive src/third_party/habitat-lab
cd ..
pip install -e home-robot/src/third_party/habitat-lab/habitat-lab
pip install -e home-robot/src/third_party/habitat-lab/habitat-baselines
```

---

## Step 11 — Install home_robot_sim

```bash
pip install -e home-robot/src/home_robot_sim
```

---

## Step 12 — Install pytorch3d (CPU-only)

> `home_robot` mapping code (`voxel.py`, `bboxes_3d.py`, `bboxes_3d_plotly.py`) imports
> pytorch3d at **top level** — missing it causes `ModuleNotFoundError` before any eval
> code runs.
> `FORCE_CUDA=0` alone is **not enough** — pytorch3d's `setup.py` raises the CUDA
> version mismatch error *before* that flag is evaluated. `CUDA_VISIBLE_DEVICES=""`
> makes `torch.cuda.is_available()` return False first, which bypasses the version
> check entirely (same pattern as Steps 5 & 7).

```bash
CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install "git+https://github.com/facebookresearch/pytorch3d.git" \
    --no-build-isolation
```

---

## Step 13 — Install remaining deps and fix numpy

> GroundingDINO upgrades numpy to 2.x, which breaks thinc/spacy.

```bash
pip install scikit-fmm sophuspy
# sophuspy installs as 'sophuspy' but the code does 'import sophus' — create a shim:
echo "from sophuspy import *" > "$(python -c 'import site; print(site.getsitepackages()[0])')/sophus.py"
pip install "numpy<2.0.0"
```

---

## Step 14 — Download model checkpoints

```bash
mkdir -p checkpoints logs data/scene_datasets
wget -P checkpoints \
    https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget -P checkpoints \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# BLIP-2 weights download automatically from HuggingFace on first run.
```

---

## Step 15 — OpenAI API key

```bash
cat > configs/openai_api_key.yaml << 'EOF'
api_key: sk-YOUR_KEY_HERE
EOF
```

---

## Step 16 — Data pack (10 HM3D scenes)

Obtain `osg_10scene_data.zip` (~360 MB) from the repo authors (requires HM3D license).
The zip must contain both episode JSONs **and** the `hm3d_v0.2/` mesh files.

```bash
cd data/scene_datasets
unzip /path/to/osg_10scene_data.zip
# If zip extracts into a subdirectory, move contents up:
# mv data/scene_datasets/hm3d  .
# mv data/scene_datasets/hm3d_v0.2  .
# rm -rf data/   # only the extra nesting wrapper
cd ../..
```

Expected structure:
```
data/scene_datasets/
├── hm3d/val/{val.json.gz, content/<10 scenes>.json.gz}
└── hm3d_v0.2/{hm3d_annotated_basis.scene_dataset_config.json, train/00XXX-<scene>/}
```

---

## Step 17 — Persistent environment variables

Add to `~/.bashrc`:

```bash
export DISPLAY=:1
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset CUDA_VISIBLE_DEVICES   # must NOT be set; blocks EGL/GL init
```

---

## Step 18 — Verify data loads

```bash
cd /path/to/osg_eval
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

Expected output: `episodes: 100 | semantic objs: <N>`

---

## Step 19 — Run evaluation (GT semantics mode)

```bash
mkdir -p logs
OSG_GT_SEM=1 OSG_NO_CV2_VIS=1 OSG_TEST_EPISODES=10 \
OSG_LATENCY_LOG=logs/latency_gt.jsonl \
MAGNUM_LOG=quiet GLOG_minloglevel=2 \
python -u navigate_homerobot.py > logs/osg_gt.log 2>&1
```

## Step 20 - Install skrl
```
pip install skrl==1.4.3
```

## Step 21 - Downgrade protobuf
```
python -m pip install --force-reinstall "protobuf==3.20.1"
```

---

## Known issues / workarounds summary

| Problem | Root cause | Fix |
|---|---|---|
| `thinc<8.4.0,>=8.3.12` install fails | Python 3.9 + latest spacy | `pip install "spacy==3.7.6"` first |
| `No module named 'torch'` during build | pip build isolation | `--no-build-isolation` |
| CUDA extension compiles when it shouldn't | `torch/utils/cpp_extension.py` `_find_cuda_home()` falls back to `/usr/local/cuda` even when `CUDA_HOME` is unset or `nvcc` is removed from PATH | `CUDA_VISIBLE_DEVICES=""` during install — forces `torch.cuda.is_available()` → False |
| CUDA version mismatch (12.8 vs 11.8) | cu118 torch + CUDA 12.8 nvcc are ABI-incompatible | `CUDA_VISIBLE_DEVICES=""` or `FORCE_CUDA=0` during install |
| `LibMambaUnsatisfiableError` pytorch-cluster | conda solver can't find `cuda-cudart >=11.8,<12.0` on CUDA 12.x systems | Use pip instead of conda |
| `GL::Context` crash (EGL) | Optimus + headless EGL + driver 570 | Reinstall habitat-sim without `headless`; use PRIME offload env vars |
| `utils` module not found | Wrong working directory | Run from repo root (`/mnt/ssd/osg_eval`) |
| `val.json.gz` not found | Zip extracted to wrong subdir | Move `hm3d/` and `hm3d_v0.2/` up one level |
| `No module named 'pytorch3d'` at runtime | Required at top-level import in `voxel.py`, `bboxes_3d.py`, `bboxes_3d_plotly.py` | Step 12: `CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install ...` (`FORCE_CUDA=0` alone fails — version check fires before the flag is read) |
| `No module named 'sophus'` or `has no attribute 'SE3'` | `home_robot` requires `sophuspy`, but the code does `import sophus`; PyPI `sophus` (23.0.1) is an empty stub | `pip uninstall sophus -y && pip install sophuspy`, then create shim: `echo "from sophuspy import *" > <site-packages>/sophus.py` (see Step 13) |
| `AssertionError: No Stage Attributes` | `hm3d_v0.2/` present but empty — mesh `.glb` files missing from zip | Obtain complete data pack from repo authors |
| `numpy` 2.x breaks thinc/spacy after GroundingDINO install | GroundingDINO `setup.py` unpins numpy | `pip install "numpy<2.0.0"` after GroundingDINO (Step 13) |
| `Not loading LLAVA` log line | Informational warning from `model_interfaces.py` — LLAVA is optional | Not an error; safe to ignore |
| `transformers` version conflict after Tag2Text install | Tag2Text requires older API | `pip install transformers==4.26.1` (Step 6) |
