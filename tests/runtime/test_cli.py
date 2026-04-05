import json
import builtins
from pathlib import Path

import pytest

from xnnehanglab_tts.cli import main
from xnnehanglab_tts.runtime.models import EnvironmentState, ResourceState


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


def test_inspect_runtime_prints_wrapped_json_result(monkeypatch, capsys, tmp_path: Path):
    config_path = _write_runtime_config(tmp_path)
    monkeypatch.setenv("XH_RUNTIME_CONFIG", str(config_path))

    exit_code = main(["inspect-runtime"])

    captured = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(captured[-1])

    assert exit_code == 0
    assert payload["kind"] == "result"
    assert payload["payload"]["runtimeDriver"] == "uv"
    assert payload["payload"]["resources"]["genie-base"]["status"] == "missing"


def test_verify_prints_named_resource_status(monkeypatch, capsys, tmp_path: Path):
    config_path = _write_runtime_config(tmp_path)
    monkeypatch.setenv("XH_RUNTIME_CONFIG", str(config_path))

    exit_code = main(["verify", "genie-base"])

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert exit_code == 0
    assert payload["payload"]["resource"]["key"] == "genie-base"
    assert payload["payload"]["resource"]["status"] == "missing"


def test_verify_rejects_unsupported_target_with_controlled_parser_error(capsys):
    with pytest.raises(SystemExit) as error:
        main(["verify", "bad-target"])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr


def test_inspect_runtime_reports_gpu_backend_contract(monkeypatch, capsys, tmp_path: Path):
    config_path = _write_runtime_config(tmp_path)
    monkeypatch.setenv("XH_RUNTIME_CONFIG", str(config_path))
    monkeypatch.setattr(
        "xnnehanglab_tts.cli.inspect_environment",
        lambda: EnvironmentState(
            mode="gpu",
            torch_available=True,
            torch_version="2.6.0+cu118",
            cuda_available=True,
        ),
    )

    exit_code = main(["inspect-runtime"])

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == 0
    assert payload["payload"]["availableBackends"] == [
        "genie-tts",
        "gsv-tts-lite",
        "faster-qwen-tts",
    ]


def test_download_emits_failed_event_for_unsupported_target(
    monkeypatch, capsys, tmp_path: Path
):
    config_path = _write_runtime_config(tmp_path)
    monkeypatch.setenv("XH_RUNTIME_CONFIG", str(config_path))

    exit_code = main(["download", "bad-target"])

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == 1
    assert payload["kind"] == "event"
    assert payload["payload"]["event"] == "download.failed"
    assert payload["payload"]["target"] == "bad-target"


def test_emit_helpers_flush_stdout_immediately(monkeypatch):
    calls = []

    def fake_print(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(builtins, "print", fake_print)

    from xnnehanglab_tts.cli import emit_event, emit_result

    emit_event({"event": "download.progress"})
    emit_result({"resource": {"key": "genie-base"}})

    assert calls == [{"flush": True}, {"flush": True}]


def test_download_streams_events_then_result_without_network(
    monkeypatch, capsys, tmp_path: Path
):
    config_path = _write_runtime_config(tmp_path)
    monkeypatch.setenv("XH_RUNTIME_CONFIG", str(config_path))

    def fake_download_target_bundle(target, emit):
        emit(
            {
                "event": "download.started",
                "target": target.target_id,
                "status": "preparing",
                "progressCurrent": 0,
                "progressTotal": 3,
                "progressUnit": "stage",
                "message": "started",
            }
        )
        emit(
            {
                "event": "download.completed",
                "target": target.target_id,
                "status": "completed",
                "progressCurrent": 3,
                "progressTotal": 3,
                "progressUnit": "stage",
                "message": "completed",
            }
        )
        return ResourceState(
            key=target.target_id,
            label=target.label,
            status="ready",
            path=str(target.resource_root),
            missing_paths=[],
        )

    monkeypatch.setattr(
        "xnnehanglab_tts.cli.download_target_bundle", fake_download_target_bundle
    )

    exit_code = main(["download", "genie-base"])

    lines = capsys.readouterr().out.strip().splitlines()
    parsed = [json.loads(line) for line in lines]

    assert exit_code == 0
    assert [entry["kind"] for entry in parsed] == ["event", "event", "result"]
    assert parsed[0]["payload"]["event"] == "download.started"
    assert parsed[1]["payload"]["event"] == "download.completed"
    assert parsed[2]["payload"]["resource"]["key"] == "genie-base"
