from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    workspace_root: Path
    models_root: Path
    genie_base_root: Path
    genie_data_dir: Path
    genie_characters_root: Path
    gsv_lite_root: Path
    cache_root: Path
    modelscope_cache_root: Path
    logs_root: Path
    download_logs_root: Path


def resolve_runtime_paths(
    workspace_root: Path,
    models_root: Path,
    cache_root: Path,
    logs_root: Path,
) -> RuntimePaths:
    genie_base_root = models_root / "GenieData"
    return RuntimePaths(
        workspace_root=workspace_root,
        models_root=models_root,
        genie_base_root=genie_base_root,
        genie_data_dir=genie_base_root,
        genie_characters_root=models_root / "genie" / "characters",
        gsv_lite_root=models_root / "GSVLiteData",
        cache_root=cache_root,
        modelscope_cache_root=cache_root / "modelscope",
        logs_root=logs_root,
        download_logs_root=logs_root / "downloads",
    )
