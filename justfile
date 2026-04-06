# 同步 git 子模块
sync-submodule:
  git submodule sync --recursive
  git submodule update --init --recursive
  git submodule absorbgitdirs

# ── torch 安装 ──────────────────────────────────────────────────────────────
# 不需要 GPU，或者只是测试
install-cpu:
  uv sync --group cpu

# CUDA 11.8 —— 适合旧驱动 / GTX 10xx 以上
# 先确认驱动版本: nvidia-smi 右上角显示 "CUDA Version: 11.x"
install-gpu-cu118:
  uv sync --group gpu

# CUDA 12.4 —— 推荐，适合 RTX 20/30/40 系 + 新驱动 (≥ 530)
# 先确认驱动版本: nvidia-smi 右上角显示 "CUDA Version: 12.x"
install-gpu-cu124:
  uv sync --group gpu-cu124

# 验证 torch 是否装好，顺手看 CUDA 是否可用
check-torch:
  uv run --no-sync python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
