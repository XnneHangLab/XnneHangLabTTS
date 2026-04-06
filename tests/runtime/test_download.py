import os
from pathlib import Path

import pytest

from xnnehanglab_tts.runtime.config import load_runtime_config
from xnnehanglab_tts.runtime.download import download_target_bundle
from xnnehanglab_tts.runtime.download_adapters import _TqdmCapture
from xnnehanglab_tts.runtime.models import DownloadStep, DownloadTargetSpec, ResourceState
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

    def fake_snapshot_download(model_id, local_dir, allow_file_pattern):
        calls.append(
            {
                "model_id": model_id,
                "local_dir": local_dir,
                "allow_file_pattern": allow_file_pattern,
            }
        )
        resource_root = Path(local_dir)
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
            "local_dir": str(target.local_dir),
            "allow_file_pattern": target.allow_file_pattern or None,
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

    def fake_snapshot_download(model_id, local_dir, allow_file_pattern):
        resource_root = Path(local_dir)
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


def test_tqdm_capture_emits_file_progress_events_for_real_modelscope_bars():
    from modelscope.hub.callback import TqdmCallback

    events = []

    with _TqdmCapture(events.append, "genie-base"):
        callback = TqdmCallback(
            "GenieData/chinese-hubert-base/chinese-hubert-base.onnx",
            180 * 1024 * 1024,
        )
        callback.update(75 * 1024 * 1024)
        callback.end()

    assert events[-1] == {
        "event": "download.file_progress",
        "target": "genie-base",
        "desc": "GenieData/chinese-hubert-base/chinese-hubert-base.onnx",
        "percent": 42,
        "downloaded": "75.0M",
        "total": "180M",
    }


def test_tqdm_capture_drops_modelscope_info_logs():
    events = []
    capture = _TqdmCapture(events.append, "genie-base")
    read_fd, write_fd = os.pipe()
    capture._saved_fd2 = write_fd

    try:
        capture._handle(
            "2026-04-06 08:00:00,000 - modelscope - INFO - Got 4 files, start to download ..."
        )
        os.close(write_fd)
        capture._saved_fd2 = None
        assert os.read(read_fd, 4096) == b""
    finally:
        if capture._saved_fd2 is not None:
            os.close(capture._saved_fd2)
        os.close(read_fd)
        os.close(capture._r)
        os.close(capture._w)

    assert events == []


def test_download_target_bundle_selects_provider_and_verifier_from_target(tmp_path: Path):
    events = []
    provider_calls = []
    verifier_calls = []

    target = DownloadTargetSpec(
        target_id="custom-target",
        label="自定义资源",
        provider="fake-provider",
        verifier="fake-verifier",
        repo_id="custom/source",
        allow_file_pattern=[],
        local_dir=tmp_path / "custom",
        resource_root=tmp_path / "custom",
        required_paths=["ready.txt"],
    )

    class FakeProvider:
        def download(self, *, target, step=None):
            provider_calls.append(
                {
                    "target_id": target.target_id,
                    "step_repo_id": None if step is None else step.repo_id,
                }
            )

    class FakeVerifier:
        def verify(self, target):
            verifier_calls.append(target.target_id)
            return ResourceState(
                key=target.target_id,
                label=target.label,
                status="ready",
                path=str(target.resource_root),
                missing_paths=[],
            )

    result = download_target_bundle(
        target=target,
        emit=events.append,
        provider_adapters={"fake-provider": FakeProvider()},
        verifiers={"fake-verifier": FakeVerifier()},
    )

    assert provider_calls == [{"target_id": "custom-target", "step_repo_id": None}]
    assert verifier_calls == ["custom-target"]
    assert [event["event"] for event in events] == [
        "download.started",
        "download.progress",
        "download.verifying",
        "download.completed",
    ]
    assert result.status == "ready"


def test_download_target_bundle_uses_step_provider_override_and_target_fallback(tmp_path: Path):
    events = []
    provider_calls = []

    target = DownloadTargetSpec(
        target_id="custom-bundle",
        label="自定义资源包",
        provider="provider-a",
        verifier="fake-verifier",
        repo_id="bundle/primary",
        allow_file_pattern=[],
        local_dir=tmp_path / "bundle",
        resource_root=tmp_path / "bundle",
        required_paths=[],
        download_steps=[
            DownloadStep(
                repo_id="bundle/step-a",
                local_dir=tmp_path / "bundle" / "a",
            ),
            DownloadStep(
                provider="provider-b",
                repo_id="bundle/step-b",
                local_dir=tmp_path / "bundle" / "b",
            ),
        ],
    )

    class RecordingProvider:
        def __init__(self, name: str):
            self.name = name

        def download(self, *, target, step=None):
            provider_calls.append(
                {
                    "provider": self.name,
                    "repo_id": None if step is None else step.repo_id,
                }
            )

    class FakeVerifier:
        def verify(self, target):
            return ResourceState(
                key=target.target_id,
                label=target.label,
                status="ready",
                path=str(target.resource_root),
                missing_paths=[],
            )

    result = download_target_bundle(
        target=target,
        emit=events.append,
        provider_adapters={
            "provider-a": RecordingProvider("provider-a"),
            "provider-b": RecordingProvider("provider-b"),
        },
        verifiers={"fake-verifier": FakeVerifier()},
    )

    assert provider_calls == [
        {"provider": "provider-a", "repo_id": "bundle/step-a"},
        {"provider": "provider-b", "repo_id": "bundle/step-b"},
    ]
    assert [event["event"] for event in events] == [
        "download.started",
        "download.progress",
        "download.progress",
        "download.verifying",
        "download.completed",
    ]
    assert result.status == "ready"
