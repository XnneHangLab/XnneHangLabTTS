import json
from pathlib import Path

import pytest

from xnnehanglab_tts.cli import main
from xnnehanglab_tts.runtime.models import EnvironmentState


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
