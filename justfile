# 同步 git 子模块
sync-submodule:
  git submodule sync --recursive
  git submodule update --init --recursive
  git submodule absorbgitdirs

# ── PyTorch 版本切换 ──────────────────────────────────────────────────────────
_use variant:
    sed -i '/# SWITCH:/ s|whl/[^"]*"|whl/{{variant}}"|' pyproject.toml
    rm -rf .venv || (echo "Error: 无法删除 .venv，请关闭占用该目录的程序（VS Code、Python 进程、终端等）后重试" && exit 1)
    rm -f uv.lock

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

# ── 资源下载 ──────────────────────────────────────────────────────────────────
# 下载 Genie 基础资源（speaker_encoder / chinese-hubert-base / G2P 等，约 1 GB）
download-genie-base:
  uv run --no-sync xnnehanglab-tts download genie-base

# 下载 GSV-Lite 推理依赖（chinese-hubert-base / roberta / g2p / sv，约 2 GB）
download-gsv-lite:
  uv run --no-sync xnnehanglab-tts download gsv-lite
