# Diagnosing `Segmentation fault (core dumped)` in WSL

The observed failure happened while importing `torch` and `cupy` in one Python process. `cuda-gcc --version` reports the Ubuntu compiler, not NVIDIA GPU or CUDA driver health.

Run these **in order** and record their output:

```bash
nvidia-smi
which python
python -V
python -m pip show torch cupy-cuda12x
python scripts/check_environment.py
```

If `nvidia-smi` fails in WSL, fix Windows GPU driver / WSL GPU passthrough first. If `torch` alone crashes, troubleshoot its installation before CuPy. If CuPy alone crashes, troubleshoot its wheel/driver/runtime. If each works alone but one combined import order crashes, investigate native library interactions and package conflicts. Do not assume the Ubuntu `cuda-gcc` 12.4 version proves the NVIDIA CUDA runtime is 12.4.

For a fresh environment, remove the failing environment only if it has no other needed packages, recreate with `conda env create -f environment.yml`, then install a **specific compatible CUDA-enabled PyTorch build** from https://pytorch.org/get-started/locally/ . Avoid `pip install torch` as a method for pinning an experimental CUDA build. Do not install the full CUDA toolkit or a Linux NVIDIA driver just to resolve an uncharacterised Python import crash.

After both packages import, run `python -c "import torch; print(torch.cuda.is_available())"` and `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`, then the combined preflight script.
