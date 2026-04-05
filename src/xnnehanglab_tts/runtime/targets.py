import os

from .models import DownloadStep, DownloadTargetSpec, ManagedPath
from .paths import RuntimePaths

GENIE_BASE_REPO_ID = os.getenv("XH_GENIE_BASE_REPO_ID", "XnneHangLab/GenieData")
GENIE_BASE_REQUIRED_PATHS = [
    "speaker_encoder.onnx",
    "chinese-hubert-base/chinese-hubert-base.onnx",
    "G2P/EnglishG2P/cmudict.rep",
    "G2P/ChineseG2P/opencpop-strict.txt",
]
GENIE_BASE_REQUIRED_FILE_PATHS = list(GENIE_BASE_REQUIRED_PATHS)
GENIE_BASE_REQUIRED_DIR_PATHS: list[str] = []

GSV_LITE_REQUIRED_DIR_PATHS = [
    "chinese-hubert-base",
    "chinese-roberta-wwm-ext-large",
    "g2p",
    "sv",
]
ROBERTA_FILE_PATTERN = [
    "pytorch_model.bin",
    "added_tokens.json",
    "config.json",
    "configuration.json",
    "README.md",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
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
    if target_id == "genie-base":
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
            required_file_paths=GENIE_BASE_REQUIRED_FILE_PATHS,
            required_dir_paths=GENIE_BASE_REQUIRED_DIR_PATHS,
        )

    if target_id == "gsv-lite":
        root = paths.gsv_lite_root
        return DownloadTargetSpec(
            target_id="gsv-lite",
            label="GSV-Lite 数据包",
            provider="modelscope",
            repo_id="pengzhendong/chinese-hubert-base",  # primary (unused when steps present)
            allow_file_pattern=[],
            local_dir=root,
            cache_dir=paths.modelscope_cache_root,
            resource_root=root,
            required_paths=GSV_LITE_REQUIRED_DIR_PATHS,
            required_dir_paths=GSV_LITE_REQUIRED_DIR_PATHS,
            download_steps=[
                DownloadStep(
                    repo_id="pengzhendong/chinese-hubert-base",
                    local_dir=root / "chinese-hubert-base",
                ),
                DownloadStep(
                    repo_id="dienstag/chinese-roberta-wwm-ext-large",
                    local_dir=root / "chinese-roberta-wwm-ext-large",
                    allow_file_pattern=ROBERTA_FILE_PATTERN,
                ),
                DownloadStep(
                    repo_id="xnnehang/gsv-v2proplus-g2p-resource",
                    local_dir=root / "g2p",
                ),
                DownloadStep(
                    repo_id="xnnehang/gsv-v2proplus-sv-resource",
                    local_dir=root / "sv",
                ),
            ],
        )

    raise KeyError(f"unsupported target: {target_id}")
