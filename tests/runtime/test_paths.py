from pathlib import Path

from xnnehanglab_tts.runtime.config import ensure_managed_dirs, load_runtime_config


def test_load_runtime_config_resolves_repo_relative_paths(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True)
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

    config, paths = load_runtime_config(config_path)

    assert config.runtime_driver == "uv"
    assert paths.workspace_root == repo_root.resolve()
    assert paths.models_root == (repo_root / "models").resolve()
    assert paths.genie_base_root == (repo_root / "models" / "genie" / "base").resolve()
    assert paths.genie_data_dir == (repo_root / "models" / "genie" / "base" / "GenieData").resolve()
    assert paths.modelscope_cache_root == (repo_root / "models" / "cache" / "modelscope").resolve()
    assert paths.download_logs_root == (repo_root / "logs" / "downloads").resolve()


def test_ensure_managed_dirs_creates_download_directories(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True)
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

    _, paths = load_runtime_config(config_path)
    ensure_managed_dirs(paths)

    assert paths.genie_base_root.is_dir()
    assert paths.genie_characters_root.is_dir()
    assert paths.modelscope_cache_root.is_dir()
    assert paths.download_logs_root.is_dir()


def test_load_runtime_config_uses_cwd_default_config(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True)
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

    monkeypatch.chdir(repo_root)
    config, paths = load_runtime_config()

    assert config.workspace_root == repo_root.resolve()
    assert paths.models_root == (repo_root / "models").resolve()
