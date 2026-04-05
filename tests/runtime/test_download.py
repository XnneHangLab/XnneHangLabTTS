from pathlib import Path

from xnnehanglab_tts.runtime.config import load_runtime_config
from xnnehanglab_tts.runtime.download import download_target_bundle
from xnnehanglab_tts.runtime.targets import get_download_target


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


def test_download_target_bundle_uses_modelscope_and_emits_stage_events(tmp_path: Path):
    _, paths = load_runtime_config(_write_runtime_config(tmp_path))
    target = get_download_target("genie-base", paths)
    events = []
    calls = []

    def fake_emit(payload):
        events.append(payload)

    def fake_snapshot_download(model_id, cache_dir, local_dir, allow_file_pattern):
        calls.append(
            {
                "model_id": model_id,
                "cache_dir": cache_dir,
                "local_dir": local_dir,
                "allow_file_pattern": allow_file_pattern,
            }
        )
        resource_root = Path(local_dir) / "GenieData"
        (resource_root / "speaker_encoder.onnx").parent.mkdir(parents=True, exist_ok=True)
        (resource_root / "speaker_encoder.onnx").write_text("ok", encoding="utf-8")
        (resource_root / "chinese-hubert-base").mkdir(parents=True, exist_ok=True)
        (resource_root / "G2P" / "EnglishG2P").mkdir(parents=True, exist_ok=True)
        (resource_root / "G2P" / "ChineseG2P").mkdir(parents=True, exist_ok=True)
        return str(resource_root)

    result = download_target_bundle(
        target=target,
        emit=fake_emit,
        snapshot_download=fake_snapshot_download,
    )

    assert calls == [
        {
            "model_id": target.repo_id,
            "cache_dir": str(target.cache_dir),
            "local_dir": str(target.local_dir),
            "allow_file_pattern": target.allow_file_pattern,
        }
    ]
    assert [event["event"] for event in events] == [
        "download.started",
        "download.progress",
        "download.verifying",
        "download.completed",
    ]
    assert result.status == "ready"
