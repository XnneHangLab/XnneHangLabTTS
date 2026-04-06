# 同步 git 子模块
sync-submodule:
  git submodule sync --recursive
  git submodule update --init --recursive
  git submodule absorbgitdirs

# 验证 torch 是否装好，顺手看 CUDA 是否可用
torch-check:
  uv run --no-sync python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
