import os

from .models import DownloadTargetSpec, ManagedPath
from .paths import RuntimePaths

GENIE_BASE_REPO_ID = os.getenv("XH_GENIE_BASE_REPO_ID", "XnneHangLab/GenieData")
GENIE_BASE_REQUIRED_PATHS = [
    "speaker_encoder.onnx",
    "chinese-hubert-base",
    "G2P/EnglishG2P",
    "G2P/ChineseG2P",
]


def build_managed_paths(paths: RuntimePaths) -> list[ManagedPath]:
    return [
        ManagedPath(key="workspace", label="根目录", path=str(paths.workspace_root)),
        ManagedPath(key="models", label="模型目录", path=str(paths.models_root)),
        ManagedPath(key="genieBase", label="Genie 基础资源", path=str(paths.genie_base_root)),
        ManagedPath(
            key="modelscopeCache",
            label="ModelScope 缓存",
            path=str(paths.modelscope_cache_root),
        ),
        ManagedPath(key="downloadLogs", label="下载日志", path=str(paths.download_logs_root)),
    ]


def get_download_target(target_id: str, paths: RuntimePaths) -> DownloadTargetSpec:
    if target_id != "genie-base":
        raise KeyError(f"unsupported target: {target_id}")

    return DownloadTargetSpec(
        target_id="genie-base",
        label="GenieData 基础资源",
        provider="modelscope",
        repo_id=GENIE_BASE_REPO_ID,
        allow_file_pattern=["GenieData/*"],
        local_dir=paths.genie_base_root,
        cache_dir=paths.modelscope_cache_root,
        resource_root=paths.genie_data_dir,
        required_paths=GENIE_BASE_REQUIRED_PATHS,
    )
