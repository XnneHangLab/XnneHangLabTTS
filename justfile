# 同步 git 子模块
sync-submodule:
  git submodule sync --recursive
  git submodule update --init --recursive
  git submodule absorbgitdirs

# ── PyTorch 版本切换 ──────────────────────────────────────────────────────────
[unix]
_use variant:
    cp torch-variants/{{variant}}.toml uv.toml
    rm -rf .venv uv.lock

[windows]
_use variant:
    Copy-Item -Force torch-variants/{{variant}}.toml uv.toml
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
    if (Test-Path uv.lock) { Remove-Item -Force uv.lock }

# 无独显 / 纯测试
use-cpu: (_use "cpu")
    uv sync

# GTX 10xx ~ RTX 20/30 系，驱动 CUDA ≤ 11.8
use-cu118: (_use "cu118")
    uv sync

# RTX 20/30/40 系新驱动，CUDA 12.4
use-cu124: (_use "cu124")
    uv sync

# RTX 50 系 Blackwell，CUDA ≥ 12.8（默认）
use-cu128: (_use "cu128")
    uv sync

# 验证 torch 是否装好，顺手看 CUDA 是否可用
torch-check:
  uv run --no-sync python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
