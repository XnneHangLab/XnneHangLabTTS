from pathlib import Path

import pytest

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
        (resource_root / "chinese-hubert-base" / "chinese-hubert-base.onnx").write_text(
            "ok",
            encoding="utf-8",
        )
        (resource_root / "G2P" / "EnglishG2P").mkdir(parents=True, exist_ok=True)
        (resource_root / "G2P" / "EnglishG2P" / "cmudict.rep").write_text(
            "ok",
            encoding="utf-8",
        )
        (resource_root / "G2P" / "ChineseG2P").mkdir(parents=True, exist_ok=True)
        (resource_root / "G2P" / "ChineseG2P" / "opencpop-strict.txt").write_text(
            "ok",
            encoding="utf-8",
        )
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
    assert events == [
        {
            "event": "download.started",
            "target": "genie-base",
            "status": "preparing",
            "message": "开始准备下载 GenieData 基础资源",
            "progressCurrent": 0,
            "progressTotal": 3,
            "progressUnit": "stage",
        },
        {
            "event": "download.progress",
            "target": "genie-base",
            "status": "downloading",
            "message": f"正在从 ModelScope 下载 {target.repo_id}",
            "progressCurrent": 1,
            "progressTotal": 3,
            "progressUnit": "stage",
        },
        {
            "event": "download.verifying",
            "target": "genie-base",
            "status": "verifying",
            "message": "开始校验 GenieData 基础资源",
            "progressCurrent": 2,
            "progressTotal": 3,
            "progressUnit": "stage",
        },
        {
            "event": "download.completed",
            "target": "genie-base",
            "status": "completed",
            "message": "GenieData 基础资源 下载完成",
            "progressCurrent": 3,
            "progressTotal": 3,
            "progressUnit": "stage",
        },
    ]
    assert result.status == "ready"


def test_download_target_bundle_raises_with_missing_paths_when_verify_not_ready(
    tmp_path: Path,
):
    _, paths = load_runtime_config(_write_runtime_config(tmp_path))
    target = get_download_target("genie-base", paths)
    events = []

    def fake_emit(payload):
        events.append(payload)

    def fake_snapshot_download(model_id, cache_dir, local_dir, allow_file_pattern):
        resource_root = Path(local_dir) / "GenieData"
        resource_root.mkdir(parents=True, exist_ok=True)
        (resource_root / "speaker_encoder.onnx").write_text("ok", encoding="utf-8")
        return str(resource_root)

    with pytest.raises(RuntimeError) as error:
        download_target_bundle(
            target=target,
            emit=fake_emit,
            snapshot_download=fake_snapshot_download,
        )

    message = str(error.value)
    assert "verify failed" in message
    assert "missing_paths=" in message
    assert "chinese-hubert-base/chinese-hubert-base.onnx" in message
    assert [event["event"] for event in events] == [
        "download.started",
        "download.progress",
        "download.verifying",
    ]
