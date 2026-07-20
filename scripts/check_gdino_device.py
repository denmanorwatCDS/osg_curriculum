#!/usr/bin/env python3
"""
Check whether GroundingDINO (and RAM) run on GPU or CPU in the current env.

Run from the repo root:
    conda activate nav_cu128
    python scripts/check_gdino_device.py                 # default device (GPU if available)
    OSG_GDINO_DEVICE=cpu python scripts/check_gdino_device.py   # force CPU for comparison

It reports: torch/CUDA versions, whether GroundingDINO's custom CUDA op (_C) loaded,
the actual device of the RAM/GroundingDINO model weights, and it runs ONE real
detection to prove the forward pass works on that device (this is what crashed with
'nvrtc: invalid --gpu-architecture' on the old cu118 / sm_120 stack).
"""

import os
import sys

# Allow running as `python scripts/check_gdino_device.py` from the repo root:
# put the repo root (parent of scripts/) on sys.path so `model_interfaces` imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from PIL import Image
import torch


def main():
    print("=== Environment ===")
    print("torch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}  (compute sm_{cap[0]}{cap[1]})")
    print("OSG_GDINO_DEVICE:", os.environ.get("OSG_GDINO_DEVICE", "(unset -> default cuda)"))

    print("\n=== GroundingDINO custom CUDA op (_C) ===")
    try:
        from groundingdino import _C  # noqa: F401
        c_loaded = True
        print("_C: LOADED  -> native custom CUDA kernels available")
    except Exception as e:
        c_loaded = False
        print("_C: NOT loaded ->", repr(e))
        print("    (deformable attention will use the pure-PyTorch fallback)")

    print("\n=== Loading VLM_GroundingDino (RAM + GroundingDINO) ===")
    from model_interfaces import VLM_GroundingDino
    t0 = time.time()
    gdino = VLM_GroundingDino()
    print(f"loaded in {time.time() - t0:.1f}s")

    gdino_dev = next(gdino.gdino_model.parameters()).device
    ram_dev = next(gdino.ram_model.parameters()).device
    print("GroundingDINO weights device:", gdino_dev)
    print("RAM weights device:          ", ram_dev)
    print("VLM_GroundingDino.device:    ", gdino.device)

    print("\n=== Running one real detection ===")
    img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
    on_cuda = gdino_dev.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    try:
        boxes, labels, crops = gdino.detect_all_objects(img)
        if on_cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak_mib = (torch.cuda.max_memory_allocated() / 1024**2) if on_cuda else 0.0
        print(f"detection OK: {len(labels)} objects in {dt:.2f}s")
        if on_cuda:
            print(f"peak CUDA memory during detect: {peak_mib:.0f} MiB")

        print("\n=== VERDICT ===")
        if on_cuda:
            kind = "native _C custom op" if c_loaded else "pure-PyTorch fallback (on GPU)"
            print(f"GroundingDINO is running on GPU (cuda)  via {kind}")
            print(f"per-detection time: {dt:.2f}s")
        else:
            print("GroundingDINO is running on CPU")
            print(f"per-detection time: {dt:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n=== VERDICT: detection FAILED ===")
        print(repr(e))
        print("If this is the nvrtc/sm_120 error, this env still can't run GDINO on GPU.")


if __name__ == "__main__":
    main()
