from dataclasses import dataclass
import os
from pathlib import Path
import tomllib

from .paths import RuntimePaths, resolve_runtime_paths


@dataclass(frozen=True)
class RuntimeConfig:
    workspace_root: Path
    models_root: Path
    cache_root: Path
    logs_root: Path
    default_backend: str
    runtime_driver: str
    python_path: str


def default_runtime_config_path() -> Path:
    return Path.cwd() / "config" / "runtime.toml"


def _resolve_child(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def load_runtime_config(config_path: Path | None = None) -> tuple[RuntimeConfig, RuntimePaths]:
    config_file = (config_path or default_runtime_config_path()).resolve()
    repo_root = config_file.parent.parent.resolve()
    raw = tomllib.loads(config_file.read_text(encoding="utf-8"))

    workspace_override = os.getenv("XH_VOICE_WORKSPACE_ROOT")
    workspace_root = (
        Path(workspace_override).resolve()
        if workspace_override
        else _resolve_child(repo_root, raw["workspace_root"])
    )
    models_root = _resolve_child(workspace_root, raw["models_root"])
    cache_root = _resolve_child(workspace_root, raw["cache_root"])
    logs_root = _resolve_child(workspace_root, raw["logs_root"])

    config = RuntimeConfig(
        workspace_root=workspace_root,
        models_root=models_root,
        cache_root=cache_root,
        logs_root=logs_root,
        default_backend=raw["default_backend"],
        runtime_driver=raw["runtime_driver"],
        python_path=raw["python_path"],
    )
    paths = resolve_runtime_paths(
        workspace_root=workspace_root,
        models_root=models_root,
        cache_root=cache_root,
        logs_root=logs_root,
    )
    return config, paths


def ensure_managed_dirs(paths: RuntimePaths) -> None:
    for directory in (
        paths.models_root,
        paths.genie_base_root,
        paths.genie_characters_root,
        paths.gsv_lite_root,
        paths.qwen_tts_0_6b_root,
        paths.qwen_tts_1_7b_root,
        paths.download_logs_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
