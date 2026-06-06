# diff-gaussian-rasterization — Pre-built Wheel for PyTorch 2.8.0 + CUDA 12.8 (Blackwell)

## Why does this exist?

In May 2026, Hugging Face migrated ZeroGPU Spaces from NVIDIA H200 to the 
**NVIDIA RTX Pro 6000 Blackwell** (sm_120, CUDA 12.8). This broke hundreds 
of Spaces that rely on `diff-gaussian-rasterization`, a CUDA extension used 
in 3D Gaussian Splatting models.

The problem is twofold:
1. ZeroGPU Spaces use `python:3.10` as base image — **no nvcc available at build time**
2. Existing pre-built wheels (e.g. from `dylanebert/LGM-mini`) were built against 
   old PyTorch versions and are **ABI-incompatible** with PyTorch 2.8.0

This repository provides a freshly compiled wheel for:
- Python 3.10
- PyTorch 2.8.0
- CUDA 12.8 (cu128)
- Linux x86_64
- CUDA architecture: **sm_120 (Blackwell only)**

## ✅ Tested & Verified

**Tested on Hugging Face ZeroGPU with FaceLift** — ✅ Works perfectly!
- Multi-view consistent 3D reconstruction
- `sparse_mv_attention` optimized for Blackwell
- All 6 views render correctly: frontal, 45° left, 90° profile left, back, 90° profile right, 45° right

## How to use on Hugging Face ZeroGPU Spaces

In your `app.py`:

```python
import subprocess
import sys

try:
    import diff_gaussian_rasterization
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "https://huggingface.co/spaces/painter3000/YOUR_SPACE/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl"
    ])
    import diff_gaussian_rasterization
```

## How the wheel was built

Built via GitHub Actions using:
- Base image: `nvidia/cuda:12.8.0-devel-ubuntu22.04`
- Python 3.10 explicitly installed
- PyTorch 2.8.0 from `https://download.pytorch.org/whl/cu128`
- `TORCH_CUDA_ARCH_LIST="12.0"` (Blackwell only)
- `FORCE_CUDA=1`
- `--no-build-isolation` to use the pre-installed torch

> ⚠️ **Why only sm_120?**
> Building for multiple CUDA architectures simultaneously (sm_70, sm_75, sm_80, sm_86, sm_90, sm_120) 
> causes **Out of Memory (OOM)** errors on GitHub Actions runners (Exit code 137). 
> Since ZeroGPU Spaces use RTX Pro 6000 Blackwell (sm_120), this wheel is optimized 
> for that specific architecture to avoid compilation failures.

## Background

This issue affects all Spaces using CUDA extensions that require compilation,
including `diff-gaussian-rasterization`, `simple-knn`, and similar packages
from the 3D Gaussian Splatting ecosystem.

**Greetings to the ~50-200 people worldwide who are dealing with exactly 
this problem right now. The solution exists now!** 👋😄

## Related issues
- https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960
