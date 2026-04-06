from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    workspace_root: Path
    models_root: Path
    genie_base_root: Path
    genie_data_dir: Path
    genie_characters_root: Path
    genie_tts_root: Path
    genie_tts_luming_v2_pro_plus_root: Path
    gsv_lite_root: Path
    qwen_tts_0_6b_root: Path
    qwen_tts_1_7b_root: Path
    cache_root: Path
    logs_root: Path
    download_logs_root: Path


def resolve_runtime_paths(
    workspace_root: Path,
    models_root: Path,
    cache_root: Path,
    logs_root: Path,
) -> RuntimePaths:
    genie_base_root = models_root / "GenieData"
    genie_tts_root = models_root / "genie-tts"
    return RuntimePaths(
        workspace_root=workspace_root,
        models_root=models_root,
        genie_base_root=genie_base_root,
        genie_data_dir=genie_base_root,
        genie_characters_root=models_root / "genie" / "characters",
        genie_tts_root=genie_tts_root,
        genie_tts_luming_v2_pro_plus_root=genie_tts_root / "luming-v2-pro-plus",
        gsv_lite_root=models_root / "GSVLiteData",
        qwen_tts_0_6b_root=models_root / "Qwen3-TTS-0.6B",
        qwen_tts_1_7b_root=models_root / "Qwen3-TTS-1.7B",
        cache_root=cache_root,
        logs_root=logs_root,
        download_logs_root=logs_root / "downloads",
    )
