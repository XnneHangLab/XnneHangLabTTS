from pathlib import Path

from xnnehanglab_tts.runtime.config import load_runtime_config
from xnnehanglab_tts.runtime.targets import get_download_target
from xnnehanglab_tts.runtime.verify import verify_target


def _write_runtime_config(repo_root: Path) -> Path:
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "runtime.toml"
    config_path.write_text(
        "\n".join(
            [
                'workspace_root = "."',
                'models_root = "models"',
                'cache_root = "models/cache"',
                'logs_root = "logs"',
                'default_backend = "genie-tts"',
                'runtime_driver = "uv"',
                'python_path = ""',
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_verify_target_returns_missing_when_resource_root_does_not_exist(tmp_path: Path):
    _, paths = load_runtime_config(_write_runtime_config(tmp_path))
    target = get_download_target("genie-base", paths)

    result = verify_target(target)

    assert result.status == "missing"
    assert result.missing_paths == target.required_paths


def test_verify_target_returns_partial_when_some_required_paths_are_missing(tmp_path: Path):
    _, paths = load_runtime_config(_write_runtime_config(tmp_path))
    target = get_download_target("genie-base", paths)
    target.resource_root.mkdir(parents=True, exist_ok=True)
    (target.resource_root / "speaker_encoder.onnx").write_text("ok", encoding="utf-8")

    result = verify_target(target)

    assert result.status == "partial"
    assert "chinese-hubert-base" in result.missing_paths


def test_verify_target_returns_ready_when_all_required_paths_exist(tmp_path: Path):
    _, paths = load_runtime_config(_write_runtime_config(tmp_path))
    target = get_download_target("genie-base", paths)

    for relative_path in target.required_paths:
        candidate = target.resource_root / relative_path
        if "." in relative_path.split("/")[-1]:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text("ok", encoding="utf-8")
        else:
            candidate.mkdir(parents=True, exist_ok=True)

    result = verify_target(target)

    assert result.status == "ready"
    assert result.missing_paths == []
