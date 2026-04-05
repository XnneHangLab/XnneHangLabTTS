import json
from pathlib import Path

from xnnehanglab_tts.cli import main


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
