# Genie Runtime Launcher Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first integrated runtime-management slice: `uv`-driven CPU/GPU inspection, `GenieData` download and verification, a serial download queue in Tauri, and launcher UI that shows queue summary, managed folders, runtime driver, and detailed console logs.

**Architecture:** Keep Python responsible for runtime facts and download execution, keep Tauri responsible for process control and queue orchestration, and keep React responsible for rendering summary and logs. All launcher-triggered runtime actions flow through `uv run python -m xnnehanglab_tts.cli ...` executed from the repository root, so phase 1 never guesses `.venv/python.exe` or conda layout details and still leaves a `runtime_driver`/`python_path` configuration entry for later expansion.

**Tech Stack:** Python 3.11, uv, Hatchling, pytest, ModelScope, Tauri 2, Rust, React 18, TypeScript, Vitest, React Testing Library

---

## File Structure

### Root Runtime

- Modify: `pyproject.toml`
- Create: `config/runtime.toml`
- Create: `src/xnnehanglab_tts/__init__.py`
- Create: `src/xnnehanglab_tts/cli.py`
- Create: `src/xnnehanglab_tts/runtime/__init__.py`
- Create: `src/xnnehanglab_tts/runtime/config.py`
- Create: `src/xnnehanglab_tts/runtime/download.py`
- Create: `src/xnnehanglab_tts/runtime/environment.py`
- Create: `src/xnnehanglab_tts/runtime/models.py`
- Create: `src/xnnehanglab_tts/runtime/paths.py`
- Create: `src/xnnehanglab_tts/runtime/targets.py`
- Create: `src/xnnehanglab_tts/runtime/verify.py`
- Create: `tests/runtime/test_cli.py`
- Create: `tests/runtime/test_download.py`
- Create: `tests/runtime/test_environment.py`
- Create: `tests/runtime/test_paths.py`
- Create: `tests/runtime/test_verify.py`

### Launcher Rust Bridge

- Modify: `launcher/src-tauri/Cargo.toml`
- Modify: `launcher/src-tauri/src/lib.rs`
- Create: `launcher/src-tauri/src/runtime/mod.rs`
- Create: `launcher/src-tauri/src/runtime/commands.rs`
- Create: `launcher/src-tauri/src/runtime/models.rs`
- Create: `launcher/src-tauri/src/runtime/process.rs`
- Create: `launcher/src-tauri/src/runtime/state.rs`

### Launcher Frontend

- Modify: `launcher/src/app/routes.tsx`
- Modify: `launcher/src/components/home/FolderCard/FolderCard.tsx`
- Modify: `launcher/src/components/home/FolderGrid/FolderGrid.tsx`
- Modify: `launcher/src/components/home/NoticePanel/NoticePanel.tsx`
- Modify: `launcher/src/data/home.ts`
- Modify: `launcher/src/data/settings.ts`
- Modify: `launcher/src/layouts/AppShell/AppShell.tsx`
- Modify: `launcher/src/layouts/AppShell/AppShell.test.tsx`
- Modify: `launcher/src/pages/ConsolePage/ConsolePage.tsx`
- Modify: `launcher/src/pages/ConsolePage/ConsolePage.test.tsx`
- Modify: `launcher/src/pages/HomePage/HomePage.tsx`
- Modify: `launcher/src/pages/HomePage/HomePage.test.tsx`
- Modify: `launcher/src/pages/SettingsPage/SettingsPage.tsx`
- Modify: `launcher/src/pages/SettingsPage/SettingsPage.test.tsx`
- Modify: `launcher/src/services/launcher/launcher.test.ts`
- Modify: `launcher/src/services/launcher/launcher.ts`
- Create: `launcher/src/pages/ModelsPage/ModelsPage.tsx`
- Create: `launcher/src/pages/ModelsPage/ModelsPage.test.tsx`
- Create: `launcher/src/services/runtime/bridge.ts`
- Create: `launcher/src/services/runtime/runtime.test.ts`
- Create: `launcher/src/services/runtime/runtime.ts`
- Create: `launcher/src/styles/models.css`

### Docs

- Modify: `README.md`

## Responsibilities

- `src/xnnehanglab_tts/runtime/*`
  Resolve repo-managed paths, inspect CPU/GPU mode, verify `GenieData`, execute ModelScope downloads, and emit structured JSON lines.
- `src/xnnehanglab_tts/cli.py`
  Single entrypoint for `inspect-runtime`, `verify`, and `download`.
- `launcher/src-tauri/src/runtime/*`
  Own the serial queue, spawn `uv run ...`, parse JSON-line events, emit frontend events, open folders, and export logs.
- `launcher/src/services/runtime/runtime.ts`
  Frontend-only runtime types and pure reducers for queue state and console entries.
- `launcher/src/services/runtime/bridge.ts`
  Thin Tauri wrapper so UI tests can mock runtime I/O cleanly.
- `launcher/src/pages/HomePage/HomePage.tsx`
  Show summary only: environment mode, `GenieData` readiness, queue summary, latest message, managed folder entries.
- `launcher/src/pages/ModelsPage/ModelsPage.tsx`
  Real download management page: `GenieData` download button and detailed queue list.
- `launcher/src/pages/ConsolePage/ConsolePage.tsx`
  Detailed runtime/event log view and export.
- `launcher/src/pages/SettingsPage/SettingsPage.tsx`
  Show `runtime_driver = uv` and reserved `python_path` entry as read-only configuration.

## Task 1: Scaffold The Root Runtime Package And Managed Paths

**Files:**
- Modify: `pyproject.toml`
- Create: `config/runtime.toml`
- Create: `src/xnnehanglab_tts/__init__.py`
- Create: `src/xnnehanglab_tts/runtime/__init__.py`
- Create: `src/xnnehanglab_tts/runtime/paths.py`
- Create: `src/xnnehanglab_tts/runtime/config.py`
- Test: `tests/runtime/test_paths.py`

- [ ] **Step 1: Write the failing managed-path tests**

```python
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
```

- [ ] **Step 2: Run the test file to verify it fails**

Run: `uv run pytest tests/runtime/test_paths.py -v`

Expected: FAIL because `src/xnnehanglab_tts/runtime/config.py` and `src/xnnehanglab_tts/runtime/paths.py` do not exist yet.

- [ ] **Step 3: Add package metadata, the tracked runtime config, and path helpers**

Update `pyproject.toml` with an installable `src` package and test discovery:

```toml
[project]
name = "XnneHangLabTTS"
version = "0.1.0"
description = "Lightweight TTS showcase repository for GSV-Lite, Genie-TTS, and faster-qwen-tts"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  "gradio>=5.24.0",
  "pydantic>=2.10.6",
  "loguru>=0.7.3",
  "soundfile>=0.13.1",
  "python-dotenv>=1.1.1",
  "tomli_w>=1.2.0",
]

[dependency-groups]
base = [
  "numpy==1.26.4",
]

dev = [
  "pytest>=8.3.4",
  "ruff>=0.9.3",
  "pyright>=1.1.391",
]

cpu = [
  "torch @ https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torchaudio @ https://download.pytorch.org/whl/cpu/torchaudio-2.6.0%2Bcpu-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torch @ https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
  "torchaudio @ https://download.pytorch.org/whl/cpu/torchaudio-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
]

gpu = [
  "torch @ https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torchaudio @ https://download.pytorch.org/whl/cu118/torchaudio-2.6.0%2Bcu118-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
  "torch @ https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
  "torchaudio @ https://download.pytorch.org/whl/cu118/torchaudio-2.6.0%2Bcu118-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux'",
]

genie-tts = [
  "modelscope>=1.28.1",
  "onnx",
  "onnxruntime==1.22.1",
  "pyyaml",
  "huggingface_hub[hf_xet]",
  "pyopenjtalk-plus",
  "nltk",
  "g2pM",
  "jamo",
  "ko_pron",
  "g2pk2",
  "eunjeon; sys_platform == 'win32'",
  "fast_langdetect",
]

[tool.hatch.build.targets.wheel]
packages = ["src/xnnehanglab_tts"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `config/runtime.toml`:

```toml
workspace_root = "."
models_root = "models"
cache_root = "models/cache"
logs_root = "logs"
default_backend = "genie-tts"
runtime_driver = "uv"
python_path = ""
```

Create `src/xnnehanglab_tts/__init__.py`:

```python
__all__ = ["__version__"]

__version__ = "0.1.0"
```

Create `src/xnnehanglab_tts/runtime/__init__.py`:

```python
from .config import ensure_managed_dirs, load_runtime_config

__all__ = ["ensure_managed_dirs", "load_runtime_config"]
```

Create `src/xnnehanglab_tts/runtime/paths.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    workspace_root: Path
    models_root: Path
    genie_base_root: Path
    genie_data_dir: Path
    genie_characters_root: Path
    cache_root: Path
    modelscope_cache_root: Path
    logs_root: Path
    download_logs_root: Path


def resolve_runtime_paths(
    workspace_root: Path,
    models_root: Path,
    cache_root: Path,
    logs_root: Path,
) -> RuntimePaths:
    genie_base_root = models_root / "genie" / "base"
    return RuntimePaths(
        workspace_root=workspace_root,
        models_root=models_root,
        genie_base_root=genie_base_root,
        genie_data_dir=genie_base_root / "GenieData",
        genie_characters_root=models_root / "genie" / "characters",
        cache_root=cache_root,
        modelscope_cache_root=cache_root / "modelscope",
        logs_root=logs_root,
        download_logs_root=logs_root / "downloads",
    )
```

Create `src/xnnehanglab_tts/runtime/config.py`:

```python
from dataclasses import dataclass
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
    return Path(__file__).resolve().parents[3] / "config" / "runtime.toml"


def _resolve_child(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def load_runtime_config(config_path: Path | None = None) -> tuple[RuntimeConfig, RuntimePaths]:
    config_file = (config_path or default_runtime_config_path()).resolve()
    repo_root = config_file.parent.parent.resolve()
    raw = tomllib.loads(config_file.read_text(encoding="utf-8"))

    workspace_root = _resolve_child(repo_root, raw["workspace_root"])
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
        paths.modelscope_cache_root,
        paths.download_logs_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run the path tests to verify they pass**

Run: `uv run pytest tests/runtime/test_paths.py -v`

Expected: PASS with both path/config tests green.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config/runtime.toml src/xnnehanglab_tts/__init__.py src/xnnehanglab_tts/runtime/__init__.py src/xnnehanglab_tts/runtime/paths.py src/xnnehanglab_tts/runtime/config.py tests/runtime/test_paths.py
git commit -m "feat: scaffold managed runtime paths"
```

## Task 2: Add Runtime Inspection, Target Verification, And The Base CLI

**Files:**
- Create: `src/xnnehanglab_tts/cli.py`
- Create: `src/xnnehanglab_tts/runtime/environment.py`
- Create: `src/xnnehanglab_tts/runtime/models.py`
- Create: `src/xnnehanglab_tts/runtime/targets.py`
- Create: `src/xnnehanglab_tts/runtime/verify.py`
- Test: `tests/runtime/test_environment.py`
- Test: `tests/runtime/test_verify.py`
- Test: `tests/runtime/test_cli.py`

- [ ] **Step 1: Write failing tests for inspect/verify/CLI output**

Create `tests/runtime/test_environment.py`:

```python
from xnnehanglab_tts.runtime.environment import inspect_environment


class FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeTorch:
    def __init__(self, version: str, cuda_available: bool):
        self.__version__ = version
        self.cuda = FakeCuda(cuda_available)


def test_inspect_environment_reports_gpu_when_cuda_is_available():
    result = inspect_environment(lambda: FakeTorch("2.6.0+cu118", True))

    assert result.mode == "gpu"
    assert result.torch_available is True
    assert result.cuda_available is True
    assert result.torch_version == "2.6.0+cu118"


def test_inspect_environment_reports_cpu_when_torch_import_fails():
    def raise_import_error():
        raise ImportError("torch missing")

    result = inspect_environment(raise_import_error)

    assert result.mode == "cpu"
    assert result.torch_available is False
    assert result.cuda_available is False
    assert result.issues == ["torch import failed: torch missing"]
```

Create `tests/runtime/test_verify.py`:

```python
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
```

Create `tests/runtime/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run the runtime inspection tests to verify they fail**

Run: `uv run pytest tests/runtime/test_environment.py tests/runtime/test_verify.py tests/runtime/test_cli.py -v`

Expected: FAIL because the runtime inspection, verify, and CLI modules do not exist yet.

- [ ] **Step 3: Implement typed runtime models, environment detection, verification, and CLI wrappers**

Create `src/xnnehanglab_tts/runtime/models.py`:

```python
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


RuntimeMode = Literal["cpu", "gpu"]
ResourceStatus = Literal["missing", "partial", "ready"]
RuntimeDriver = Literal["uv"]
DownloadTaskStatus = Literal[
    "queued",
    "preparing",
    "downloading",
    "verifying",
    "completed",
    "failed",
    "cancelled",
]


class ManagedPath(BaseModel):
    key: str
    label: str
    path: str


class ResourceState(BaseModel):
    key: str
    label: str
    status: ResourceStatus
    path: str
    missing_paths: list[str] = Field(default_factory=list)


class EnvironmentState(BaseModel):
    mode: RuntimeMode
    torch_available: bool
    torch_version: str | None = None
    cuda_available: bool = False
    issues: list[str] = Field(default_factory=list)


class RuntimeInspection(BaseModel):
    runtime_driver: RuntimeDriver
    default_backend: str
    environment: EnvironmentState
    available_backends: list[str]
    managed_paths: list[ManagedPath]
    resources: dict[str, ResourceState]
    latest_message: str


class VerifyResult(BaseModel):
    resource: ResourceState


class CliEnvelope(BaseModel):
    kind: Literal["result", "event"]
    payload: dict


class DownloadTargetSpec(BaseModel):
    target_id: str
    label: str
    provider: str
    repo_id: str
    allow_file_pattern: list[str]
    local_dir: Path
    cache_dir: Path
    resource_root: Path
    required_paths: list[str]
```

Create `src/xnnehanglab_tts/runtime/environment.py`:

```python
from collections.abc import Callable
import importlib
from typing import Any

from .models import EnvironmentState


def _default_torch_loader() -> Any:
    return importlib.import_module("torch")


def inspect_environment(torch_loader: Callable[[], Any] | None = None) -> EnvironmentState:
    loader = torch_loader or _default_torch_loader
    try:
        torch = loader()
    except ImportError as error:
        return EnvironmentState(
            mode="cpu",
            torch_available=False,
            cuda_available=False,
            issues=[f"torch import failed: {error}"],
        )

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    return EnvironmentState(
        mode="gpu" if cuda_available else "cpu",
        torch_available=True,
        torch_version=str(getattr(torch, "__version__", "unknown")),
        cuda_available=cuda_available,
    )
```

Create `src/xnnehanglab_tts/runtime/targets.py`:

```python
import os

from .models import DownloadTargetSpec, ManagedPath
from .paths import RuntimePaths

GENIE_BASE_REPO_ID = os.getenv("XH_GENIE_BASE_REPO_ID", "XnneHangLab/GenieData")
GENIE_BASE_REQUIRED_PATHS = [
    "speaker_encoder.onnx",
    "chinese-hubert-base",
    "G2P/EnglishG2P",
    "G2P/ChineseG2P",
]


def build_managed_paths(paths: RuntimePaths) -> list[ManagedPath]:
    return [
        ManagedPath(key="workspace", label="根目录", path=str(paths.workspace_root)),
        ManagedPath(key="models", label="模型目录", path=str(paths.models_root)),
        ManagedPath(key="genieBase", label="Genie 基础资源", path=str(paths.genie_base_root)),
        ManagedPath(key="modelscopeCache", label="ModelScope 缓存", path=str(paths.modelscope_cache_root)),
        ManagedPath(key="downloadLogs", label="下载日志", path=str(paths.download_logs_root)),
    ]


def get_download_target(target_id: str, paths: RuntimePaths) -> DownloadTargetSpec:
    if target_id != "genie-base":
        raise KeyError(f"unsupported target: {target_id}")

    return DownloadTargetSpec(
        target_id="genie-base",
        label="GenieData 基础资源",
        provider="modelscope",
        repo_id=GENIE_BASE_REPO_ID,
        allow_file_pattern=["GenieData/*"],
        local_dir=paths.genie_base_root,
        cache_dir=paths.modelscope_cache_root,
        resource_root=paths.genie_data_dir,
        required_paths=GENIE_BASE_REQUIRED_PATHS,
    )
```

Create `src/xnnehanglab_tts/runtime/verify.py`:

```python
from .models import ResourceState
from .targets import get_download_target


def verify_target(target) -> ResourceState:
    if not target.resource_root.exists():
        return ResourceState(
            key=target.target_id,
            label=target.label,
            status="missing",
            path=str(target.resource_root),
            missing_paths=target.required_paths,
        )

    missing_paths: list[str] = []
    for relative_path in target.required_paths:
        if not (target.resource_root / relative_path).exists():
            missing_paths.append(relative_path)

    if not missing_paths:
        status = "ready"
    elif len(missing_paths) == len(target.required_paths):
        status = "missing"
    else:
        status = "partial"

    return ResourceState(
        key=target.target_id,
        label=target.label,
        status=status,
        path=str(target.resource_root),
        missing_paths=missing_paths,
    )
```

Create `src/xnnehanglab_tts/cli.py`:

```python
import argparse
import json
import os
from pathlib import Path

from xnnehanglab_tts.runtime.config import ensure_managed_dirs, load_runtime_config
from xnnehanglab_tts.runtime.environment import inspect_environment
from xnnehanglab_tts.runtime.models import CliEnvelope, RuntimeInspection, VerifyResult
from xnnehanglab_tts.runtime.targets import build_managed_paths, get_download_target
from xnnehanglab_tts.runtime.verify import verify_target


def _config_path_from_env() -> Path | None:
    value = os.getenv("XH_RUNTIME_CONFIG")
    return Path(value) if value else None


def emit_result(payload: dict) -> None:
    print(CliEnvelope(kind="result", payload=payload).model_dump_json())


def build_runtime_inspection() -> RuntimeInspection:
    config, paths = load_runtime_config(_config_path_from_env())
    ensure_managed_dirs(paths)
    environment = inspect_environment()
    resource = verify_target(get_download_target("genie-base", paths))
    available_backends = (
        ["genie-tts"]
        if environment.mode == "cpu"
        else ["genie-tts", "gsv-tts-lite", "faster-qwen-tts"]
    )
    latest_message = f"运行驱动 {config.runtime_driver}，当前环境 {environment.mode.upper()}"
    return RuntimeInspection(
        runtime_driver=config.runtime_driver,
        default_backend=config.default_backend,
        environment=environment,
        available_backends=available_backends,
        managed_paths=build_managed_paths(paths),
        resources={"genie-base": resource},
        latest_message=latest_message,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xnnehanglab-tts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect-runtime")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("target")
    subparsers.add_parser("download").add_argument("target")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-runtime":
        emit_result(build_runtime_inspection().model_dump(by_alias=True))
        return 0

    if args.command == "verify":
        _, paths = load_runtime_config(_config_path_from_env())
        ensure_managed_dirs(paths)
        resource = verify_target(get_download_target(args.target, paths))
        emit_result(VerifyResult(resource=resource).model_dump(by_alias=True))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the runtime inspection tests and a manual CLI smoke check**

Run: `uv run pytest tests/runtime/test_environment.py tests/runtime/test_verify.py tests/runtime/test_cli.py -v`

Expected: PASS with environment/verify/CLI tests green.

Run: `uv run python -m xnnehanglab_tts.cli inspect-runtime`

Expected: one JSON line with `"kind":"result"`, `"runtimeDriver":"uv"`, and a `resources.genie-base.status` value of `missing`, `partial`, or `ready`.

- [ ] **Step 5: Commit**

```bash
git add src/xnnehanglab_tts/cli.py src/xnnehanglab_tts/runtime/environment.py src/xnnehanglab_tts/runtime/models.py src/xnnehanglab_tts/runtime/targets.py src/xnnehanglab_tts/runtime/verify.py tests/runtime/test_environment.py tests/runtime/test_verify.py tests/runtime/test_cli.py
git commit -m "feat: add runtime inspect and verify cli"
```

## Task 3: Add ModelScope Download Execution With Structured JSON Events

**Files:**
- Modify: `src/xnnehanglab_tts/cli.py`
- Create: `src/xnnehanglab_tts/runtime/download.py`
- Test: `tests/runtime/test_download.py`

- [ ] **Step 1: Write the failing download tests**

Create `tests/runtime/test_download.py`:

```python
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
```

- [ ] **Step 2: Run the download tests to verify they fail**

Run: `uv run pytest tests/runtime/test_download.py -v`

Expected: FAIL because the download executor does not exist yet.

- [ ] **Step 3: Implement the ModelScope download adapter and the `download` CLI subcommand**

Create `src/xnnehanglab_tts/runtime/download.py`:

```python
from collections.abc import Callable

from .models import ResourceState
from .verify import verify_target


SnapshotDownload = Callable[..., str]
EmitEvent = Callable[[dict], None]


def _modelscope_snapshot_download(**kwargs) -> str:
    from modelscope import snapshot_download

    return snapshot_download(**kwargs)


def download_target_bundle(
    target,
    emit: EmitEvent,
    snapshot_download: SnapshotDownload | None = None,
) -> ResourceState:
    downloader = snapshot_download or _modelscope_snapshot_download

    emit(
        {
            "event": "download.started",
            "target": target.target_id,
            "status": "preparing",
            "message": f"开始准备下载 {target.label}",
            "progressCurrent": 0,
            "progressTotal": 3,
            "progressUnit": "stage",
        }
    )
    emit(
        {
            "event": "download.progress",
            "target": target.target_id,
            "status": "downloading",
            "message": f"正在从 ModelScope 下载 {target.repo_id}",
            "progressCurrent": 1,
            "progressTotal": 3,
            "progressUnit": "stage",
        }
    )

    downloader(
        model_id=target.repo_id,
        cache_dir=str(target.cache_dir),
        local_dir=str(target.local_dir),
        allow_file_pattern=target.allow_file_pattern,
    )

    emit(
        {
            "event": "download.verifying",
            "target": target.target_id,
            "status": "verifying",
            "message": f"开始校验 {target.label}",
            "progressCurrent": 2,
            "progressTotal": 3,
            "progressUnit": "stage",
        }
    )

    result = verify_target(target)
    if result.status != "ready":
        raise RuntimeError(f"{target.target_id} verify failed: {result.status}")

    emit(
        {
            "event": "download.completed",
            "target": target.target_id,
            "status": "completed",
            "message": f"{target.label} 下载完成",
            "progressCurrent": 3,
            "progressTotal": 3,
            "progressUnit": "stage",
        }
    )
    return result
```

Update `src/xnnehanglab_tts/cli.py` to emit JSON-line events during downloads:

```python
from xnnehanglab_tts.runtime.download import download_target_bundle


def emit_event(payload: dict) -> None:
    print(CliEnvelope(kind="event", payload=payload).model_dump_json())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-runtime":
        emit_result(build_runtime_inspection().model_dump(by_alias=True))
        return 0

    if args.command == "verify":
        _, paths = load_runtime_config(_config_path_from_env())
        ensure_managed_dirs(paths)
        resource = verify_target(get_download_target(args.target, paths))
        emit_result(VerifyResult(resource=resource).model_dump(by_alias=True))
        return 0

    if args.command == "download":
        _, paths = load_runtime_config(_config_path_from_env())
        ensure_managed_dirs(paths)
        target = get_download_target(args.target, paths)
        try:
            resource = download_target_bundle(target=target, emit=emit_event)
        except Exception as error:
            emit_event(
                {
                    "event": "download.failed",
                    "target": args.target,
                    "status": "failed",
                    "message": str(error),
                    "progressCurrent": 3,
                    "progressTotal": 3,
                    "progressUnit": "stage",
                }
            )
            return 1

        emit_result(VerifyResult(resource=resource).model_dump(by_alias=True))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2
```

- [ ] **Step 4: Run the download tests and a local verify command**

Run: `uv run pytest tests/runtime/test_download.py tests/runtime/test_cli.py -v`

Expected: PASS with structured event emission verified.

Run: `uv run python -m xnnehanglab_tts.cli verify genie-base`

Expected: one JSON line with `"resource":{"key":"genie-base", ... }`.

- [ ] **Step 5: Commit**

```bash
git add src/xnnehanglab_tts/cli.py src/xnnehanglab_tts/runtime/download.py tests/runtime/test_download.py
git commit -m "feat: add genie base download events"
```

## Task 4: Add The Tauri Queue, Repo-Root Process Runner, Folder Open, And Log Export

**Files:**
- Modify: `launcher/src-tauri/Cargo.toml`
- Modify: `launcher/src-tauri/src/lib.rs`
- Create: `launcher/src-tauri/src/runtime/mod.rs`
- Create: `launcher/src-tauri/src/runtime/commands.rs`
- Create: `launcher/src-tauri/src/runtime/models.rs`
- Create: `launcher/src-tauri/src/runtime/process.rs`
- Create: `launcher/src-tauri/src/runtime/state.rs`

- [ ] **Step 1: Write failing Rust queue-state tests**

Add these tests into `launcher/src-tauri/src/runtime/state.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::{QueueState, RuntimeTaskRecord, TaskStatus};

    #[test]
    fn enqueue_adds_a_task_and_keeps_it_queued() {
        let mut queue = QueueState::default();
        let task = queue.enqueue("genie-base".to_string(), "GenieData 基础资源".to_string());

        assert_eq!(task.status, TaskStatus::Queued);
        assert_eq!(queue.tasks.len(), 1);
        assert_eq!(queue.tasks[0].target, "genie-base");
    }

    #[test]
    fn apply_status_updates_existing_task_progress() {
        let mut queue = QueueState::default();
        let task = queue.enqueue("genie-base".to_string(), "GenieData 基础资源".to_string());

        queue.apply_update(
            &task.task_id,
            TaskStatus::Downloading,
            "正在下载".to_string(),
            1,
            3,
        );

        let current = queue.tasks.iter().find(|item| item.task_id == task.task_id).unwrap();
        assert_eq!(current.status, TaskStatus::Downloading);
        assert_eq!(current.progress_current, 1);
        assert_eq!(current.progress_total, 3);
    }
}
```

- [ ] **Step 2: Run Rust tests to verify they fail**

Run: `cargo test --manifest-path launcher/src-tauri/Cargo.toml runtime::state::tests -- --nocapture`

Expected: FAIL because the `runtime` module does not exist yet.

- [ ] **Step 3: Implement typed task records, a repo-root resolver, and Tauri commands**

Update `launcher/src-tauri/Cargo.toml`:

```toml
[dependencies]
serde_json = "1.0"
serde = { version = "1.0", features = ["derive"] }
log = "0.4"
tauri = { version = "2.10.3" }
tauri-plugin-log = "2"
```

Create `launcher/src-tauri/src/runtime/models.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum TaskStatus {
    Queued,
    Preparing,
    Downloading,
    Verifying,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeTaskRecord {
    pub task_id: String,
    pub target: String,
    pub label: String,
    pub status: TaskStatus,
    pub message: String,
    pub progress_current: u64,
    pub progress_total: u64,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeEventPayload {
    pub event: String,
    pub task_id: String,
    pub target: String,
    pub status: String,
    pub message: String,
    pub progress_current: u64,
    pub progress_total: u64,
    pub progress_unit: String,
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PythonEnvelope {
    pub kind: String,
    pub payload: serde_json::Value,
}
```

Create `launcher/src-tauri/src/runtime/state.rs`:

```rust
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::models::{RuntimeTaskRecord, TaskStatus};

#[derive(Default)]
pub struct QueueState {
    next_id: u64,
    pub tasks: Vec<RuntimeTaskRecord>,
    pub waiting: VecDeque<String>,
    pub worker_running: bool,
}

impl QueueState {
    pub fn enqueue(&mut self, target: String, label: String) -> RuntimeTaskRecord {
        self.next_id += 1;
        let task_id = format!("task-{}", self.next_id);
        let task = RuntimeTaskRecord {
            task_id: task_id.clone(),
            target,
            label,
            status: TaskStatus::Queued,
            message: "已进入下载队列".to_string(),
            progress_current: 0,
            progress_total: 3,
            updated_at: current_timestamp(),
        };
        self.waiting.push_back(task_id.clone());
        self.tasks.push(task.clone());
        task
    }

    pub fn next_queued_task_id(&mut self) -> Option<String> {
        self.waiting.pop_front()
    }

    pub fn apply_update(
        &mut self,
        task_id: &str,
        status: TaskStatus,
        message: String,
        progress_current: u64,
        progress_total: u64,
    ) {
        if let Some(task) = self.tasks.iter_mut().find(|item| item.task_id == task_id) {
            task.status = status;
            task.message = message;
            task.progress_current = progress_current;
            task.progress_total = progress_total;
            task.updated_at = current_timestamp();
        }
    }
}

pub struct RuntimeState {
    pub workspace_root: PathBuf,
    pub queue: Arc<Mutex<QueueState>>,
}

impl RuntimeState {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            queue: Arc::new(Mutex::new(QueueState::default())),
        }
    }
}

pub fn current_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    seconds.to_string()
}

pub fn resolve_workspace_root() -> Result<PathBuf, String> {
    if let Ok(value) = std::env::var("XH_VOICE_WORKSPACE_ROOT") {
        return Ok(PathBuf::from(value));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .map(|path| path.to_path_buf())
        .ok_or_else(|| "failed to resolve workspace root from launcher/src-tauri".to_string())
}
```

Create `launcher/src-tauri/src/runtime/process.rs`:

```rust
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use tauri::{AppHandle, Emitter};

use super::models::{PythonEnvelope, RuntimeEventPayload, TaskStatus};
use super::state::RuntimeState;

pub fn run_inspect_command(workspace_root: &Path) -> Result<serde_json::Value, String> {
    let output = Command::new("uv")
        .args(["run", "python", "-m", "xnnehanglab_tts.cli", "inspect-runtime"])
        .current_dir(workspace_root)
        .output()
        .map_err(|error| format!("failed to run inspect-runtime: {error}"))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_line = stdout
        .lines()
        .last()
        .ok_or_else(|| "inspect-runtime returned no stdout".to_string())?;
    let envelope: PythonEnvelope =
        serde_json::from_str(last_line).map_err(|error| error.to_string())?;
    Ok(envelope.payload)
}

pub fn run_download_command(app: AppHandle, state: RuntimeState, task_id: String) -> Result<(), String> {
    let mut command = Command::new("uv");
    command
        .args(["run", "python", "-m", "xnnehanglab_tts.cli", "download", "genie-base"])
        .current_dir(&state.workspace_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|error| format!("failed to spawn download process: {error}"))?;
    let stdout = child.stdout.take().ok_or_else(|| "missing child stdout".to_string())?;
    let stderr = child.stderr.take().ok_or_else(|| "missing child stderr".to_string())?;

    let stdout_reader = BufReader::new(stdout);
    for line_result in stdout_reader.lines() {
        let line = line_result.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(envelope) = serde_json::from_str::<PythonEnvelope>(&line) {
            if envelope.kind == "event" {
                let payload = envelope.payload;
                let status = payload["status"].as_str().unwrap_or("downloading");
                {
                    let mut queue = state.queue.lock().unwrap();
                    queue.apply_update(
                        &task_id,
                        task_status_from_str(status),
                        payload["message"].as_str().unwrap_or("").to_string(),
                        payload["progressCurrent"].as_u64().unwrap_or(0),
                        payload["progressTotal"].as_u64().unwrap_or(3),
                    );
                }
                let event = RuntimeEventPayload {
                    event: payload["event"].as_str().unwrap_or("download.progress").to_string(),
                    task_id: task_id.clone(),
                    target: payload["target"].as_str().unwrap_or("genie-base").to_string(),
                    status: status.to_string(),
                    message: payload["message"].as_str().unwrap_or("").to_string(),
                    progress_current: payload["progressCurrent"].as_u64().unwrap_or(0),
                    progress_total: payload["progressTotal"].as_u64().unwrap_or(3),
                    progress_unit: payload["progressUnit"].as_str().unwrap_or("stage").to_string(),
                    timestamp: super::state::current_timestamp(),
                };
                app.emit("runtime:event", &event).map_err(|error| error.to_string())?;
            }
        } else {
            app.emit("runtime:raw-log", &line).map_err(|error| error.to_string())?;
        }
    }

    let stderr_reader = BufReader::new(stderr);
    for line_result in stderr_reader.lines() {
        let line = line_result.map_err(|error| error.to_string())?;
        if !line.trim().is_empty() {
            app.emit("runtime:raw-log", &line).map_err(|error| error.to_string())?;
        }
    }

    let status = child.wait().map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("download process exited with status {status}"))
    }
}

pub fn drain_download_queue(app: AppHandle, state: RuntimeState) {
    loop {
        let next_task_id = {
            let mut queue = state.queue.lock().unwrap();
            queue.next_queued_task_id()
        };

        let Some(task_id) = next_task_id else {
            let mut queue = state.queue.lock().unwrap();
            queue.worker_running = false;
            break;
        };

        if let Err(error) = run_download_command(app.clone(), RuntimeState {
            workspace_root: state.workspace_root.clone(),
            queue: state.queue.clone(),
        }, task_id.clone()) {
            let mut queue = state.queue.lock().unwrap();
            queue.apply_update(&task_id, TaskStatus::Failed, error, 3, 3);
        }
    }
}

pub fn open_path(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("explorer");
        command.arg(path);
        command
    };

    #[cfg(target_os = "linux")]
    let mut command = {
        let mut command = Command::new("xdg-open");
        command.arg(path);
        command
    };

    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("open");
        command.arg(path);
        command
    };

    command.spawn().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_console_log(workspace_root: &Path, contents: &str) -> Result<PathBuf, String> {
    let log_dir = workspace_root.join("logs").join("downloads");
    fs::create_dir_all(&log_dir).map_err(|error| error.to_string())?;
    let log_path = log_dir.join(format!("launcher-{}.log", super::state::current_timestamp()));
    fs::write(&log_path, contents).map_err(|error| error.to_string())?;
    Ok(log_path)
}

fn task_status_from_str(value: &str) -> TaskStatus {
    match value {
        "queued" => TaskStatus::Queued,
        "preparing" => TaskStatus::Preparing,
        "downloading" => TaskStatus::Downloading,
        "verifying" => TaskStatus::Verifying,
        "completed" => TaskStatus::Completed,
        "failed" => TaskStatus::Failed,
        "cancelled" => TaskStatus::Cancelled,
        _ => TaskStatus::Downloading,
    }
}
```

Create `launcher/src-tauri/src/runtime/commands.rs`:

```rust
use tauri::{AppHandle, Manager, State};

use super::process::{drain_download_queue, open_path, run_inspect_command, write_console_log};
use super::state::{resolve_workspace_root, RuntimeState};

#[tauri::command]
pub fn inspect_runtime(state: State<'_, RuntimeState>) -> Result<serde_json::Value, String> {
    run_inspect_command(&state.workspace_root)
}

#[tauri::command]
pub fn enqueue_download(app: AppHandle, state: State<'_, RuntimeState>, target: String) -> Result<serde_json::Value, String> {
    let task = {
        let mut queue = state.queue.lock().unwrap();
        let task = queue.enqueue(target, "GenieData 基础资源".to_string());
        if queue.worker_running {
            return serde_json::to_value(task).map_err(|error| error.to_string());
        }
        queue.worker_running = true;
        task
    };

    let app_handle = app.clone();
    let runtime_state = RuntimeState {
        workspace_root: state.workspace_root.clone(),
        queue: state.queue.clone(),
    };

    tauri::async_runtime::spawn(async move {
        drain_download_queue(app_handle.clone(), runtime_state);
    });

    serde_json::to_value(task).map_err(|error| error.to_string())
}

#[tauri::command]
pub fn list_download_tasks(state: State<'_, RuntimeState>) -> Result<serde_json::Value, String> {
    let queue = state.queue.lock().unwrap();
    serde_json::to_value(&queue.tasks).map_err(|error| error.to_string())
}

#[tauri::command]
pub fn open_managed_path(state: State<'_, RuntimeState>, path_key: String) -> Result<(), String> {
    let path = match path_key.as_str() {
        "workspace" => state.workspace_root.clone(),
        "models" => state.workspace_root.join("models"),
        "genieBase" => state.workspace_root.join("models").join("genie").join("base"),
        "modelscopeCache" => state.workspace_root.join("models").join("cache").join("modelscope"),
        "downloadLogs" => state.workspace_root.join("logs").join("downloads"),
        other => return Err(format!("unsupported managed path key: {other}")),
    };
    open_path(&path)
}

#[tauri::command]
pub fn export_console_logs(state: State<'_, RuntimeState>, contents: String) -> Result<String, String> {
    let path = write_console_log(&state.workspace_root, &contents)?;
    Ok(path.display().to_string())
}

pub fn build_runtime_state() -> Result<RuntimeState, String> {
    Ok(RuntimeState::new(resolve_workspace_root()?))
}
```

Create `launcher/src-tauri/src/runtime/mod.rs`:

```rust
pub mod commands;
pub mod models;
pub mod process;
pub mod state;
```

Update `launcher/src-tauri/src/lib.rs`:

```rust
mod runtime;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  let runtime_state = runtime::commands::build_runtime_state().expect("failed to build runtime state");

  tauri::Builder::default()
    .manage(runtime_state)
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![
      runtime::commands::inspect_runtime,
      runtime::commands::enqueue_download,
      runtime::commands::list_download_tasks,
      runtime::commands::open_managed_path,
      runtime::commands::export_console_logs,
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
```

- [ ] **Step 4: Run Rust tests and `cargo check`**

Run: `cargo test --manifest-path launcher/src-tauri/Cargo.toml runtime::state::tests -- --nocapture`

Expected: PASS with queue-state helper tests green.

Run: `cargo check --manifest-path launcher/src-tauri/Cargo.toml`

Expected: PASS with all custom commands compiling.

- [ ] **Step 5: Commit**

```bash
git -C launcher add src-tauri/Cargo.toml src-tauri/src/lib.rs src-tauri/src/runtime/mod.rs src-tauri/src/runtime/commands.rs src-tauri/src/runtime/models.rs src-tauri/src/runtime/process.rs src-tauri/src/runtime/state.rs
git -C launcher commit -m "feat: add tauri runtime queue bridge"
```

## Task 5: Add Frontend Runtime Types, Event Reducers, And The Tauri Bridge Wrapper

**Files:**
- Modify: `launcher/src/services/launcher/launcher.ts`
- Modify: `launcher/src/services/launcher/launcher.test.ts`
- Create: `launcher/src/services/runtime/runtime.ts`
- Create: `launcher/src/services/runtime/runtime.test.ts`
- Create: `launcher/src/services/runtime/bridge.ts`

- [ ] **Step 1: Write failing frontend helper tests**

Create `launcher/src/services/runtime/runtime.test.ts`:

```ts
import {
  applyRuntimeEvent,
  buildManagedFolderItems,
  createConsoleLogFromRuntimeEvent,
  type RuntimeInspection,
  type RuntimeTaskRecord,
} from './runtime';

describe('runtime helpers', () => {
  const inspection: RuntimeInspection = {
    runtimeDriver: 'uv',
    defaultBackend: 'genie-tts',
    environment: {
      mode: 'cpu',
      torchAvailable: true,
      torchVersion: '2.6.0+cpu',
      cudaAvailable: false,
      issues: [],
    },
    availableBackends: ['genie-tts'],
    managedPaths: [
      { key: 'workspace', label: '根目录', path: '/repo' },
      { key: 'genieBase', label: 'Genie 基础资源', path: '/repo/models/genie/base' },
      { key: 'modelscopeCache', label: 'ModelScope 缓存', path: '/repo/models/cache/modelscope' },
      { key: 'downloadLogs', label: '下载日志', path: '/repo/logs/downloads' },
    ],
    resources: {
      'genie-base': {
        key: 'genie-base',
        label: 'GenieData 基础资源',
        status: 'missing',
        path: '/repo/models/genie/base/GenieData',
        missingPaths: ['speaker_encoder.onnx'],
      },
    },
    latestMessage: '运行驱动 uv，当前环境 CPU',
  };

  it('builds home folder items from managed paths', () => {
    expect(buildManagedFolderItems(inspection)).toEqual([
      { key: 'workspace', title: '根目录', path: '/repo', icon: '📁' },
      { key: 'genieBase', title: 'Genie 基础资源', path: '/repo/models/genie/base', icon: '🧠' },
      { key: 'modelscopeCache', title: 'ModelScope 缓存', path: '/repo/models/cache/modelscope', icon: '⬇' },
      { key: 'downloadLogs', title: '下载日志', path: '/repo/logs/downloads', icon: '🧾' },
    ]);
  });

  it('upserts task state from a runtime event', () => {
    const current: RuntimeTaskRecord[] = [];
    const next = applyRuntimeEvent(current, {
      event: 'download.progress',
      taskId: 'task-1',
      target: 'genie-base',
      status: 'downloading',
      message: '正在下载',
      progressCurrent: 1,
      progressTotal: 3,
      progressUnit: 'stage',
      timestamp: '1712300000',
    });

    expect(next).toEqual([
      {
        taskId: 'task-1',
        target: 'genie-base',
        label: 'GenieData 基础资源',
        status: 'downloading',
        message: '正在下载',
        progressCurrent: 1,
        progressTotal: 3,
        updatedAt: '1712300000',
      },
    ]);
  });

  it('converts runtime events into console lines', () => {
    const log = createConsoleLogFromRuntimeEvent({
      event: 'download.failed',
      taskId: 'task-1',
      target: 'genie-base',
      status: 'failed',
      message: 'network error',
      progressCurrent: 3,
      progressTotal: 3,
      progressUnit: 'stage',
      timestamp: '1712300000',
    });

    expect(log.kind).toBe('stderr');
    expect(log.text).toContain('network error');
  });
});
```

Update `launcher/src/services/launcher/launcher.test.ts` to keep only console formatting helpers:

```ts
import {
  createConsoleLog,
  formatConsoleExport,
  getVisibleCommand,
} from './launcher';

describe('launcher helpers', () => {
  it('falls back to 未配置命令 when no command is configured', () => {
    expect(getVisibleCommand(null)).toBe('未配置命令');
    expect(getVisibleCommand('')).toBe('未配置命令');
    expect(getVisibleCommand('uv run python -m xnnehanglab_tts.cli inspect-runtime')).toBe(
      'uv run python -m xnnehanglab_tts.cli inspect-runtime',
    );
  });

  it('creates timestamped console logs', () => {
    const log = createConsoleLog('system', '已进入下载队列');

    expect(log.kind).toBe('system');
    expect(log.text).toBe('已进入下载队列');
    expect(log.time.length).toBeGreaterThan(0);
  });

  it('formats logs for export', () => {
    const output = formatConsoleExport([
      {
        id: 'log-1',
        time: '2026-04-05 15:00:00',
        kind: 'system',
        text: '已进入下载队列',
      },
    ]);

    expect(output).toContain('[system]');
    expect(output).toContain('已进入下载队列');
  });
});
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run: `npm --prefix launcher run test -- --run src/services/launcher/launcher.test.ts src/services/runtime/runtime.test.ts`

Expected: FAIL because the new runtime helper module does not exist and `launcher.ts` still contains launch-toggle logic.

- [ ] **Step 3: Implement pure runtime helpers and the Tauri bridge wrapper**

Update `launcher/src/services/launcher/launcher.ts`:

```ts
export type ConsoleLogKind = 'system' | 'stdout' | 'stderr';

export interface ConsoleLogEntry {
  id: string;
  time: string;
  kind: ConsoleLogKind;
  text: string;
}

export const UNCONFIGURED_COMMAND_LABEL = '未配置命令';

export function getVisibleCommand(command: string | null): string {
  const value = command?.trim();
  return value ? value : UNCONFIGURED_COMMAND_LABEL;
}

function createTimestamp() {
  const date = new Date();
  return date.toLocaleString('zh-CN', { hour12: false });
}

export function createConsoleLog(
  kind: ConsoleLogKind,
  text: string,
): ConsoleLogEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    time: createTimestamp(),
    kind,
    text,
  };
}

export function formatConsoleExport(logs: ConsoleLogEntry[]) {
  return logs
    .map((entry) => `[${entry.time}] [${entry.kind}] ${entry.text}`)
    .join('\n');
}
```

Create `launcher/src/services/runtime/runtime.ts`:

```ts
import { createConsoleLog, type ConsoleLogEntry } from '../launcher/launcher';

export type RuntimeMode = 'cpu' | 'gpu';
export type ResourceStatus = 'missing' | 'partial' | 'ready';
export type RuntimeDriver = 'uv';
export type DownloadTaskStatus =
  | 'queued'
  | 'preparing'
  | 'downloading'
  | 'verifying'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface ManagedPath {
  key: string;
  label: string;
  path: string;
}

export interface RuntimeInspection {
  runtimeDriver: RuntimeDriver;
  defaultBackend: string;
  environment: {
    mode: RuntimeMode;
    torchAvailable: boolean;
    torchVersion: string | null;
    cudaAvailable: boolean;
    issues: string[];
  };
  availableBackends: string[];
  managedPaths: ManagedPath[];
  resources: Record<
    string,
    {
      key: string;
      label: string;
      status: ResourceStatus;
      path: string;
      missingPaths: string[];
    }
  >;
  latestMessage: string;
}

export interface RuntimeTaskRecord {
  taskId: string;
  target: string;
  label: string;
  status: DownloadTaskStatus;
  message: string;
  progressCurrent: number;
  progressTotal: number;
  updatedAt: string;
}

export interface RuntimeEvent {
  event: string;
  taskId: string;
  target: string;
  status: DownloadTaskStatus;
  message: string;
  progressCurrent: number;
  progressTotal: number;
  progressUnit: string;
  timestamp: string;
}

export interface ManagedFolderItem {
  key: string;
  title: string;
  path: string;
  icon: string;
}

const folderIcons: Record<string, string> = {
  workspace: '📁',
  genieBase: '🧠',
  modelscopeCache: '⬇',
  downloadLogs: '🧾',
  models: '◫',
};

export function buildManagedFolderItems(
  inspection: RuntimeInspection,
): ManagedFolderItem[] {
  return inspection.managedPaths.map((item) => ({
    key: item.key,
    title: item.label,
    path: item.path,
    icon: folderIcons[item.key] ?? '📁',
  }));
}

export function applyRuntimeEvent(
  current: RuntimeTaskRecord[],
  event: RuntimeEvent,
): RuntimeTaskRecord[] {
  const next = [...current];
  const index = next.findIndex((item) => item.taskId === event.taskId);
  const task: RuntimeTaskRecord = {
    taskId: event.taskId,
    target: event.target,
    label: event.target === 'genie-base' ? 'GenieData 基础资源' : event.target,
    status: event.status,
    message: event.message,
    progressCurrent: event.progressCurrent,
    progressTotal: event.progressTotal,
    updatedAt: event.timestamp,
  };

  if (index === -1) {
    next.push(task);
  } else {
    next[index] = task;
  }

  return next;
}

export function createConsoleLogFromRuntimeEvent(
  event: RuntimeEvent,
): ConsoleLogEntry {
  const kind = event.status === 'failed' ? 'stderr' : 'system';
  return createConsoleLog(kind, `${event.target}: ${event.message}`);
}

export function getQueueSummary(tasks: RuntimeTaskRecord[]) {
  const activeTask =
    tasks.find((task) =>
      ['queued', 'preparing', 'downloading', 'verifying'].includes(task.status),
    ) ?? null;

  return {
    queueLength: tasks.filter((task) =>
      ['queued', 'preparing', 'downloading', 'verifying'].includes(task.status),
    ).length,
    activeTask,
  };
}
```

Create `launcher/src/services/runtime/bridge.ts`:

```ts
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type {
  RuntimeEvent,
  RuntimeInspection,
  RuntimeTaskRecord,
} from './runtime';

export function inspectRuntime() {
  return invoke<RuntimeInspection>('inspect_runtime');
}

export function enqueueDownload(target: string) {
  return invoke<RuntimeTaskRecord>('enqueue_download', { target });
}

export function listDownloadTasks() {
  return invoke<RuntimeTaskRecord[]>('list_download_tasks');
}

export function openManagedPath(pathKey: string) {
  return invoke<void>('open_managed_path', { pathKey });
}

export function exportConsoleLogs(contents: string) {
  return invoke<string>('export_console_logs', { contents });
}

export async function subscribeRuntimeEvents(
  onEvent: (event: RuntimeEvent) => void,
  onRawLog: (line: string) => void,
) {
  const unlistenEvent = await listen<RuntimeEvent>('runtime:event', (event) => {
    onEvent(event.payload);
  });
  const unlistenRaw = await listen<string>('runtime:raw-log', (event) => {
    onRawLog(event.payload);
  });

  return () => {
    unlistenEvent();
    unlistenRaw();
  };
}
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `npm --prefix launcher run test -- --run src/services/launcher/launcher.test.ts src/services/runtime/runtime.test.ts`

Expected: PASS with console helpers and runtime reducers green.

- [ ] **Step 5: Commit**

```bash
git -C launcher add src/services/launcher/launcher.ts src/services/launcher/launcher.test.ts src/services/runtime/runtime.ts src/services/runtime/runtime.test.ts src/services/runtime/bridge.ts
git -C launcher commit -m "feat: add launcher runtime helpers"
```

## Task 6: Wire AppShell To Runtime Inspection And Build Home, Models, And Settings Views

**Files:**
- Modify: `launcher/src/app/routes.tsx`
- Modify: `launcher/src/components/home/FolderCard/FolderCard.tsx`
- Modify: `launcher/src/components/home/FolderGrid/FolderGrid.tsx`
- Modify: `launcher/src/components/home/NoticePanel/NoticePanel.tsx`
- Modify: `launcher/src/data/home.ts`
- Modify: `launcher/src/data/settings.ts`
- Modify: `launcher/src/layouts/AppShell/AppShell.tsx`
- Modify: `launcher/src/layouts/AppShell/AppShell.test.tsx`
- Modify: `launcher/src/pages/HomePage/HomePage.tsx`
- Modify: `launcher/src/pages/HomePage/HomePage.test.tsx`
- Modify: `launcher/src/pages/SettingsPage/SettingsPage.tsx`
- Modify: `launcher/src/pages/SettingsPage/SettingsPage.test.tsx`
- Create: `launcher/src/pages/ModelsPage/ModelsPage.tsx`
- Create: `launcher/src/pages/ModelsPage/ModelsPage.test.tsx`
- Create: `launcher/src/styles/models.css`

- [ ] **Step 1: Write failing UI tests for runtime inspection, queue summary, and read-only driver settings**

Create `launcher/src/pages/ModelsPage/ModelsPage.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { ModelsPage } from './ModelsPage';

describe('ModelsPage', () => {
  it('renders the genie resource card and queue entries', async () => {
    const user = userEvent.setup();
    const onDownloadGenieBase = vi.fn();
    const onOpenPath = vi.fn();

    render(
      <ModelsPage
        inspection={{
          runtimeDriver: 'uv',
          defaultBackend: 'genie-tts',
          environment: {
            mode: 'cpu',
            torchAvailable: true,
            torchVersion: '2.6.0+cpu',
            cudaAvailable: false,
            issues: [],
          },
          availableBackends: ['genie-tts'],
          managedPaths: [
            { key: 'genieBase', label: 'Genie 基础资源', path: '/repo/models/genie/base' },
          ],
          resources: {
            'genie-base': {
              key: 'genie-base',
              label: 'GenieData 基础资源',
              status: 'missing',
              path: '/repo/models/genie/base/GenieData',
              missingPaths: ['speaker_encoder.onnx'],
            },
          },
          latestMessage: '运行驱动 uv，当前环境 CPU',
        }}
        tasks={[
          {
            taskId: 'task-1',
            target: 'genie-base',
            label: 'GenieData 基础资源',
            status: 'downloading',
            message: '正在下载',
            progressCurrent: 1,
            progressTotal: 3,
            updatedAt: '1712300000',
          },
        ]}
        onDownloadGenieBase={onDownloadGenieBase}
        onOpenPath={onOpenPath}
      />,
    );

    expect(screen.getByRole('heading', { name: '模型管理' })).toBeInTheDocument();
    expect(screen.getByText('GenieData 基础资源')).toBeInTheDocument();
    expect(screen.getByText('missing')).toBeInTheDocument();
    expect(screen.getByText('正在下载')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '下载 GenieData' }));
    expect(onDownloadGenieBase).toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: '打开 Genie 目录' }));
    expect(onOpenPath).toHaveBeenCalledWith('genieBase');
  });
});
```

Append this integration test to `launcher/src/layouts/AppShell/AppShell.test.tsx`:

```tsx
import { waitFor } from '@testing-library/react';
import * as runtimeBridge from '../../services/runtime/bridge';
import type { RuntimeEvent } from '../../services/runtime/runtime';

vi.mock('../../services/runtime/bridge', async () => {
  const actual = await vi.importActual<typeof import('../../services/runtime/bridge')>(
    '../../services/runtime/bridge',
  );

  const listeners = new Set<(event: RuntimeEvent) => void>();
  const rawListeners = new Set<(line: string) => void>();

  return {
    ...actual,
    inspectRuntime: vi.fn().mockResolvedValue({
      runtimeDriver: 'uv',
      defaultBackend: 'genie-tts',
      environment: {
        mode: 'cpu',
        torchAvailable: true,
        torchVersion: '2.6.0+cpu',
        cudaAvailable: false,
        issues: [],
      },
      availableBackends: ['genie-tts'],
      managedPaths: [
        { key: 'workspace', label: '根目录', path: '/repo' },
        { key: 'genieBase', label: 'Genie 基础资源', path: '/repo/models/genie/base' },
        { key: 'modelscopeCache', label: 'ModelScope 缓存', path: '/repo/models/cache/modelscope' },
        { key: 'downloadLogs', label: '下载日志', path: '/repo/logs/downloads' },
      ],
      resources: {
        'genie-base': {
          key: 'genie-base',
          label: 'GenieData 基础资源',
          status: 'missing',
          path: '/repo/models/genie/base/GenieData',
          missingPaths: ['speaker_encoder.onnx'],
        },
      },
      latestMessage: '运行驱动 uv，当前环境 CPU',
    }),
    listDownloadTasks: vi.fn().mockResolvedValue([]),
    enqueueDownload: vi.fn().mockResolvedValue({
      taskId: 'task-1',
      target: 'genie-base',
      label: 'GenieData 基础资源',
      status: 'queued',
      message: '已进入下载队列',
      progressCurrent: 0,
      progressTotal: 3,
      updatedAt: '1712300000',
    }),
    openManagedPath: vi.fn().mockResolvedValue(undefined),
    exportConsoleLogs: vi.fn().mockResolvedValue('/repo/logs/downloads/launcher.log'),
    subscribeRuntimeEvents: vi.fn().mockImplementation(async (onEvent, onRawLog) => {
      listeners.add(onEvent);
      rawListeners.add(onRawLog);
      return () => {
        listeners.delete(onEvent);
        rawListeners.delete(onRawLog);
      };
    }),
    __emitRuntimeEvent(event: RuntimeEvent) {
      listeners.forEach((listener) => listener(event));
    },
    __emitRawLog(line: string) {
      rawListeners.forEach((listener) => listener(line));
    },
  };
});

it('loads runtime inspection, navigates to models, and keeps queue state in sync', async () => {
  const user = userEvent.setup();
  render(<App />);

  await waitFor(() =>
    expect(screen.getByText('运行驱动 uv')).toBeInTheDocument(),
  );

  await user.click(screen.getByRole('button', { name: '模型管理' }));
  await user.click(screen.getByRole('button', { name: '下载 GenieData' }));

  expect(runtimeBridge.enqueueDownload).toHaveBeenCalledWith('genie-base');

  const mockedBridge = runtimeBridge as typeof runtimeBridge & {
    __emitRuntimeEvent: (event: RuntimeEvent) => void;
  };

  mockedBridge.__emitRuntimeEvent({
    event: 'download.progress',
    taskId: 'task-1',
    target: 'genie-base',
    status: 'downloading',
    message: '正在下载',
    progressCurrent: 1,
    progressTotal: 3,
    progressUnit: 'stage',
    timestamp: '1712300001',
  });

  expect(screen.getByText('正在下载')).toBeInTheDocument();

  await user.click(screen.getByRole('button', { name: '一键启动' }));
  expect(screen.getByText('队列长度 1')).toBeInTheDocument();

  await user.click(screen.getByRole('button', { name: '打开 Genie 基础资源' }));
  expect(runtimeBridge.openManagedPath).toHaveBeenCalledWith('genieBase');

  await user.click(screen.getByRole('button', { name: '设置' }));
  expect(screen.getByDisplayValue('uv')).toBeDisabled();
});
```

- [ ] **Step 2: Run the launcher UI tests to verify they fail**

Run: `npm --prefix launcher run test -- --run src/pages/ModelsPage/ModelsPage.test.tsx src/layouts/AppShell/AppShell.test.tsx src/pages/HomePage/HomePage.test.tsx src/pages/SettingsPage/SettingsPage.test.tsx`

Expected: FAIL because `ModelsPage` does not exist, `AppShell` does not call the runtime bridge, and Settings does not show the runtime driver entry yet.

- [ ] **Step 3: Implement AppShell runtime state, the summary Home page, the real Models page, and read-only runtime settings**

Update `launcher/src/data/home.ts` to keep only brand copy and notices:

```ts
export const heroCopy = {
  eyebrow: 'XnneHangLab Launcher Template',
  title: '绘心 - 启动器',
  description: '让 AI 更有温度，也更适合长期陪伴。',
};

export const versionMeta = ['启动器版本：绘心启动器 0.1.0'];

export const notices = [
  '当前阶段优先接入 GenieData 下载、环境识别和日志链路。',
  'CPU 环境仅开放 Genie-TTS 基础资源链路，GPU 环境在后续阶段再扩更多后端。',
  '模型下载进入串行队列后，可在模型管理页查看详情，在控制台页查看详细日志。',
];
```

Update `launcher/src/data/settings.ts` with runtime copy:

```ts
export const runtimeSettings = {
  driverLabel: '运行驱动',
  pythonPathLabel: 'Python 路径',
  pythonPathPlaceholder: '阶段一未启用，后续按 driver 扩展',
};
```

Update `launcher/src/components/home/FolderCard/FolderCard.tsx`:

```tsx
interface FolderCardProps {
  item: {
    key: string;
    title: string;
    path: string;
    icon: string;
  };
  onOpen: (pathKey: string) => void;
}

export function FolderCard({ item, onOpen }: FolderCardProps) {
  return (
    <button
      type="button"
      className="folder-card"
      aria-label={`打开 ${item.title}`}
      onClick={() => onOpen(item.key)}
    >
      <span className="folder-left">
        <span className="folder-icon" aria-hidden="true">
          {item.icon}
        </span>
        <span className="folder-text">
          <span className="folder-title">{item.title}</span>
          <span className="folder-sub">{item.path}</span>
        </span>
      </span>

      <span className="arrow" aria-hidden="true">
        ›
      </span>
    </button>
  );
}
```

Update `launcher/src/components/home/FolderGrid/FolderGrid.tsx`:

```tsx
interface FolderGridProps {
  items: {
    key: string;
    title: string;
    path: string;
    icon: string;
  }[];
  onOpen: (pathKey: string) => void;
}

export function FolderGrid({ items, onOpen }: FolderGridProps) {
  return (
    <div className="folder-grid">
      {items.map((item) => (
        <FolderCard key={item.key} item={item} onOpen={onOpen} />
      ))}
    </div>
  );
}
```

Update `launcher/src/components/home/NoticePanel/NoticePanel.tsx`:

```tsx
interface NoticePanelProps {
  notices: string[];
  runtimeMode: string;
  genieStatus: string;
  queueLength: number;
  latestMessage: string;
  onOpenModels: () => void;
}

export function NoticePanel({
  notices,
  runtimeMode,
  genieStatus,
  queueLength,
  latestMessage,
  onOpenModels,
}: NoticePanelProps) {
  return (
    <aside className="notice">
      <h2>公告</h2>
      <p>当前环境 {runtimeMode.toUpperCase()}</p>
      <p>GenieData 状态 {genieStatus}</p>
      <p>队列长度 {queueLength}</p>
      <p>{latestMessage}</p>

      {notices.map((notice) => (
        <p key={notice}>{notice}</p>
      ))}

      <button type="button" className="run-btn" data-state="ready" onClick={onOpenModels}>
        前往模型管理
      </button>
    </aside>
  );
}
```

Create `launcher/src/pages/ModelsPage/ModelsPage.tsx`:

```tsx
import type { RuntimeInspection, RuntimeTaskRecord } from '../../services/runtime/runtime';
import '../../styles/models.css';

interface ModelsPageProps {
  inspection: RuntimeInspection | null;
  tasks: RuntimeTaskRecord[];
  onDownloadGenieBase: () => void;
  onOpenPath: (pathKey: string) => void;
}

export function ModelsPage({
  inspection,
  tasks,
  onDownloadGenieBase,
  onOpenPath,
}: ModelsPageProps) {
  const genieResource = inspection?.resources['genie-base'];

  return (
    <div className="models-page">
      <header className="models-header">
        <div>
          <h1>模型管理</h1>
          <p>当前阶段只管理 GenieData 基础资源，角色包后续独立加入。</p>
        </div>
        <button type="button" onClick={onDownloadGenieBase}>
          下载 GenieData
        </button>
      </header>

      <section className="models-card">
        <h2>GenieData 基础资源</h2>
        <p>状态 {genieResource?.status ?? 'missing'}</p>
        <p>路径 {genieResource?.path ?? '未初始化'}</p>
        <button type="button" onClick={() => onOpenPath('genieBase')}>
          打开 Genie 目录
        </button>
      </section>

      <section className="models-card">
        <h2>下载队列</h2>
        {tasks.length === 0 ? (
          <p>当前没有下载任务</p>
        ) : (
          <div className="models-task-list">
            {tasks.map((task) => (
              <article key={task.taskId} className="models-task">
                <div>{task.label}</div>
                <div>{task.status}</div>
                <div>{task.message}</div>
                <div>
                  {task.progressCurrent} / {task.progressTotal}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
```

Create `launcher/src/styles/models.css`:

```css
.models-page {
  display: grid;
  gap: 18px;
}

.models-header,
.models-card {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 20px;
  padding: 20px;
}

.models-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.models-task-list {
  display: grid;
  gap: 12px;
}

.models-task {
  display: grid;
  gap: 4px;
  border-top: 1px solid var(--line);
  padding-top: 12px;
}
```

Update `launcher/src/pages/HomePage/HomePage.tsx`:

```tsx
import { FolderGrid } from '../../components/home/FolderGrid/FolderGrid';
import { HeroBanner } from '../../components/home/HeroBanner/HeroBanner';
import { NoticePanel } from '../../components/home/NoticePanel/NoticePanel';
import { notices, versionMeta } from '../../data/home';
import type { ManagedFolderItem, RuntimeInspection, RuntimeTaskRecord } from '../../services/runtime/runtime';
import '../../styles/home.css';

interface HomePageProps {
  inspection: RuntimeInspection | null;
  tasks: RuntimeTaskRecord[];
  folders: ManagedFolderItem[];
  onOpenPath: (pathKey: string) => void;
  onOpenModels: () => void;
}

export function HomePage({
  inspection,
  tasks,
  folders,
  onOpenPath,
  onOpenModels,
}: HomePageProps) {
  const genieStatus = inspection?.resources['genie-base']?.status ?? 'missing';
  const runtimeMode = inspection?.environment.mode ?? 'cpu';

  return (
    <div className="home-page">
      <HeroBanner />

      <div className="main-grid">
        <div>
          <h2 className="section-title">文件夹</h2>
          <FolderGrid items={folders} onOpen={onOpenPath} />

          <div className="meta">
            {versionMeta.map((line) => (
              <div key={line}>{line}</div>
            ))}
          </div>
        </div>

        <NoticePanel
          notices={notices}
          runtimeMode={runtimeMode}
          genieStatus={genieStatus}
          queueLength={tasks.filter((task) =>
            ['queued', 'preparing', 'downloading', 'verifying'].includes(task.status),
          ).length}
          latestMessage={inspection?.latestMessage ?? '正在读取运行时信息'}
          onOpenModels={onOpenModels}
        />
      </div>
    </div>
  );
}
```

Update `launcher/src/pages/SettingsPage/SettingsPage.tsx` by changing the function signature and inserting the runtime card above the existing network settings card:

```tsx
import type { RuntimeDriver } from '../../services/runtime/runtime';
import { runtimeSettings } from '../../data/settings';

interface SettingsPageProps {
  runtimeDriver: RuntimeDriver;
  pythonPath: string;
}

export function SettingsPage({
  runtimeDriver,
  pythonPath,
}: SettingsPageProps) {
  const [activeTab, setActiveTab] = useState<SettingsTabId>('general');
  const [proxyAddress, setProxyAddress] = useState(proxyDefaults.address);
  const [proxyToggles, setProxyToggles] = useState({
    git: proxyDefaults.git,
    pip: proxyDefaults.pip,
    env: proxyDefaults.env,
    modelDownload: proxyDefaults.modelDownload,
  });
  const [mirrorToggles, setMirrorToggles] = useState(
    Object.fromEntries(
      mirrorSettings.map((item) => [item.id, item.defaultValue]),
    ) as Record<string, boolean>,
  );
  const [preferenceToggles, setPreferenceToggles] = useState(
    Object.fromEntries(
      preferenceSettings.map((item) => [item.id, item.defaultValue]),
    ) as Record<string, boolean>,
  );

  return (
    <div className="settings-shell">
      <SettingsTabs
        items={settingsTabs}
        activeTab={activeTab}
        onSelect={setActiveTab}
      />

      <div className="settings-wrap">
        {activeTab === 'general' ? (
          <div
            id="settings-panel-general"
            role="tabpanel"
            aria-labelledby="settings-tab-general"
          >
            <div className="group-title group-title--standalone">运行设置</div>

            <SettingCard>
              <SettingRow
                name={runtimeSettings.driverLabel}
                description="阶段一固定通过 uv 驱动运行时"
                icon="Uv"
              >
                <input
                  className="proxy-input"
                  aria-label={runtimeSettings.driverLabel}
                  value={runtimeDriver}
                  disabled
                  readOnly
                />
              </SettingRow>

              <SettingRow
                name={runtimeSettings.pythonPathLabel}
                description="后续按运行驱动扩展显式 Python 路径"
                icon="Py"
              >
                <input
                  className="proxy-input"
                  aria-label={runtimeSettings.pythonPathLabel}
                  value={pythonPath}
                  placeholder={runtimeSettings.pythonPathPlaceholder}
                  disabled
                  readOnly
                />
              </SettingRow>
            </SettingCard>

            <div className="group-title group-title--standalone">网络设置</div>
            <SettingCard>
              <SettingRow
                name="代理设置"
                description="代理服务器设置"
                icon="🛩"
                trailing={
                  <span className="setting-chevron" aria-hidden="true">
                    ⌃
                  </span>
                }
              />

              <SettingRow name="代理服务器地址" inset>
                <input
                  className="proxy-input"
                  aria-label="代理服务器地址"
                  value={proxyAddress}
                  onChange={(event) => setProxyAddress(event.target.value)}
                />
              </SettingRow>

              <SettingRow name="将代理应用到 Git" inset>
                <ToggleSwitch
                  label="将代理应用到 Git"
                  checked={proxyToggles.git}
                  onChange={(next) =>
                    setProxyToggles((current) => ({ ...current, git: next }))
                  }
                />
              </SettingRow>

              <SettingRow name="将代理应用到 Pip" inset>
                <ToggleSwitch
                  label="将代理应用到 Pip"
                  checked={proxyToggles.pip}
                  onChange={(next) =>
                    setProxyToggles((current) => ({ ...current, pip: next }))
                  }
                />
              </SettingRow>

              <SettingRow name="将代理应用到环境变量" inset>
                <ToggleSwitch
                  label="将代理应用到环境变量"
                  checked={proxyToggles.env}
                  onChange={(next) =>
                    setProxyToggles((current) => ({ ...current, env: next }))
                  }
                />
              </SettingRow>

              <SettingRow name="将代理应用到模型下载" inset>
                <ToggleSwitch
                  label="将代理应用到模型下载"
                  checked={proxyToggles.modelDownload}
                  onChange={(next) =>
                    setProxyToggles((current) => ({
                      ...current,
                      modelDownload: next,
                    }))
                  }
                />
              </SettingRow>
            </SettingCard>

            <SettingCard>
              {mirrorSettings.map((item) => (
                <SettingRow
                  key={item.id}
                  name={item.label}
                  description={item.description}
                  icon={item.icon}
                >
                  <ToggleSwitch
                    label={item.label}
                    checked={mirrorToggles[item.id]}
                    onChange={(next) =>
                      setMirrorToggles((current) => ({
                        ...current,
                        [item.id]: next,
                      }))
                    }
                  />
                </SettingRow>
              ))}
            </SettingCard>

            <div className="group-title group-title--standalone">偏好设置</div>

            <SettingCard>
              {preferenceSettings.map((item) => (
                <SettingRow
                  key={item.id}
                  name={item.label}
                  description={item.description}
                  icon={item.icon}
                >
                  <ToggleSwitch
                    label={item.label}
                    checked={preferenceToggles[item.id]}
                    onChange={(next) =>
                      setPreferenceToggles((current) => ({
                        ...current,
                        [item.id]: next,
                      }))
                    }
                  />
                </SettingRow>
              ))}
            </SettingCard>

            <div className="footer-space" />
          </div>
        ) : (
          <div
            id="settings-panel-about"
            role="tabpanel"
            aria-labelledby="settings-tab-about"
          >
            <div className="about-card">
              {aboutInfo.map((line) => (
                <p key={line}>{line}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

Update `launcher/src/layouts/AppShell/AppShell.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { Sidebar } from '../../components/navigation/Sidebar/Sidebar';
import { Topbar } from '../../components/window/Topbar/Topbar';
import { navItems, type PageId } from '../../data/nav';
import { renderPage } from '../../app/routes';
import { createConsoleLog, formatConsoleExport, type ConsoleLogEntry } from '../../services/launcher/launcher';
import { enqueueDownload, exportConsoleLogs, inspectRuntime, listDownloadTasks, openManagedPath, subscribeRuntimeEvents } from '../../services/runtime/bridge';
import {
  applyRuntimeEvent,
  buildManagedFolderItems,
  createConsoleLogFromRuntimeEvent,
  type RuntimeInspection,
  type RuntimeTaskRecord,
} from '../../services/runtime/runtime';
import {
  readStoredTheme,
  toggleThemeMode,
  writeStoredTheme,
  type ThemeMode,
} from '../../services/theme/theme';

export function AppShell() {
  const [activePage, setActivePage] = useState<PageId>('home');
  const [theme, setTheme] = useState<ThemeMode>(() => readStoredTheme() ?? 'night');
  const [inspection, setInspection] = useState<RuntimeInspection | null>(null);
  const [tasks, setTasks] = useState<RuntimeTaskRecord[]>([]);
  const [logs, setLogs] = useState<ConsoleLogEntry[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const [wrapLines, setWrapLines] = useState(true);

  useEffect(() => {
    writeStoredTheme(theme);
  }, [theme]);

  useEffect(() => {
    void inspectRuntime().then(setInspection);
    void listDownloadTasks().then(setTasks);

    let unsubscribe = () => undefined;
    void subscribeRuntimeEvents(
      (event) => {
        setTasks((current) => applyRuntimeEvent(current, event));
        setLogs((current) => [...current, createConsoleLogFromRuntimeEvent(event)]);
      },
      (line) => {
        setLogs((current) => [...current, createConsoleLog('stdout', line)]);
      },
    ).then((cleanup) => {
      unsubscribe = cleanup;
    });

    return () => {
      unsubscribe();
    };
  }, []);

  const folders = inspection ? buildManagedFolderItems(inspection) : [];

  async function handleDownloadGenieBase() {
    const task = await enqueueDownload('genie-base');
    setTasks((current) => {
      const next = current.filter((item) => item.taskId !== task.taskId);
      next.push(task);
      return next;
    });
    setLogs((current) => [...current, createConsoleLog('system', `${task.label}: ${task.message}`)]);
    setActivePage('models');
  }

  async function handleExportLogs() {
    const output = formatConsoleExport(logs);
    const path = await exportConsoleLogs(output);
    setLogs((current) => [...current, createConsoleLog('system', `日志已导出到 ${path}`)]);
  }

  return (
    <div className="launcher-root" data-theme={theme}>
      <div className="app-shell">
        <Sidebar
          items={navItems}
          activePage={activePage}
          onSelect={setActivePage}
          theme={theme}
          onToggleTheme={() => setTheme((current) => toggleThemeMode(current))}
        />

        <main className="content-shell">
          <Topbar />
          <section className="page-shell">
            {renderPage(activePage, {
              inspection,
              tasks,
              folders,
              logs,
              autoScroll,
              wrapLines,
              onOpenModels: () => setActivePage('models'),
              onDownloadGenieBase: handleDownloadGenieBase,
              onOpenPath: (pathKey) => void openManagedPath(pathKey),
              runtimeDriver: inspection?.runtimeDriver ?? 'uv',
              pythonPath: '',
              onSetAutoScroll: setAutoScroll,
              onSetWrapLines: setWrapLines,
              onClearLogs: () => setLogs([]),
              onCopyLog: (text) => void navigator.clipboard?.writeText(text),
              onExportLogs: handleExportLogs,
            })}
          </section>
        </main>
      </div>
    </div>
  );
}
```

Update `launcher/src/app/routes.tsx`:

```tsx
import { ModelsPage } from '../pages/ModelsPage/ModelsPage';
import type { ManagedFolderItem, RuntimeInspection, RuntimeTaskRecord } from '../services/runtime/runtime';

interface RenderPageOptions {
  inspection: RuntimeInspection | null;
  tasks: RuntimeTaskRecord[];
  folders: ManagedFolderItem[];
  logs: ConsoleLogEntry[];
  autoScroll: boolean;
  wrapLines: boolean;
  onOpenModels: () => void;
  onDownloadGenieBase: () => void;
  onOpenPath: (pathKey: string) => void;
  runtimeDriver: 'uv';
  pythonPath: string;
  onSetAutoScroll: (next: boolean) => void;
  onSetWrapLines: (next: boolean) => void;
  onClearLogs: () => void;
  onCopyLog: (text: string) => void;
  onExportLogs: () => void;
}

export function renderPage(pageId: PageId, options: RenderPageOptions): ReactElement {
  switch (pageId) {
    case 'home':
      return (
        <HomePage
          inspection={options.inspection}
          tasks={options.tasks}
          folders={options.folders}
          onOpenPath={options.onOpenPath}
          onOpenModels={options.onOpenModels}
        />
      );
    case 'models':
      return (
        <ModelsPage
          inspection={options.inspection}
          tasks={options.tasks}
          onDownloadGenieBase={options.onDownloadGenieBase}
          onOpenPath={options.onOpenPath}
        />
      );
    case 'settings':
      return (
        <SettingsPage
          runtimeDriver={options.runtimeDriver}
          pythonPath={options.pythonPath}
        />
      );
    case 'advanced':
      return (
        <PlaceholderPage
          title="高级选项"
          description="预留更细粒度的运行参数与后端切换入口。"
        />
      );
    case 'troubleshooting':
      return (
        <PlaceholderPage
          title="疑难解答"
          description="预留更细粒度的运行诊断与修复入口。"
        />
      );
    case 'versions':
      return (
        <PlaceholderPage
          title="版本管理"
          description="预留运行时版本切换和回滚能力。"
        />
      );
    case 'tools':
      return (
        <PlaceholderPage
          title="小工具"
          description="预留下载修复、目录清理和附加操作入口。"
        />
      );
    case 'community':
      return (
        <PlaceholderPage
          title="交流群"
          description="预留社区入口和外链跳转。"
        />
      );
    case 'console':
      return (
        <ConsolePage
          runtimeDriver={options.runtimeDriver}
          tasks={options.tasks}
          logs={options.logs}
          autoScroll={options.autoScroll}
          wrapLines={options.wrapLines}
          onSetAutoScroll={options.onSetAutoScroll}
          onSetWrapLines={options.onSetWrapLines}
          onClearLogs={options.onClearLogs}
          onCopyLog={options.onCopyLog}
          onExportLogs={options.onExportLogs}
        />
      );
    default: {
      const exhaustiveCheck: never = pageId;
      throw new Error(`Unhandled page id: ${exhaustiveCheck}`);
    }
  }
}
```

- [ ] **Step 4: Run the launcher UI tests to verify they pass**

Run: `npm --prefix launcher run test -- --run src/pages/ModelsPage/ModelsPage.test.tsx src/layouts/AppShell/AppShell.test.tsx src/pages/HomePage/HomePage.test.tsx src/pages/SettingsPage/SettingsPage.test.tsx`

Expected: PASS with runtime inspection rendering, folder open wiring, and read-only driver configuration visible.

- [ ] **Step 5: Commit**

```bash
git -C launcher add src/app/routes.tsx src/components/home/FolderCard/FolderCard.tsx src/components/home/FolderGrid/FolderGrid.tsx src/components/home/NoticePanel/NoticePanel.tsx src/data/home.ts src/data/settings.ts src/layouts/AppShell/AppShell.tsx src/layouts/AppShell/AppShell.test.tsx src/pages/HomePage/HomePage.tsx src/pages/HomePage/HomePage.test.tsx src/pages/ModelsPage/ModelsPage.tsx src/pages/ModelsPage/ModelsPage.test.tsx src/pages/SettingsPage/SettingsPage.tsx src/pages/SettingsPage/SettingsPage.test.tsx src/styles/models.css
git -C launcher commit -m "feat: add runtime summary and models page"
```

## Task 7: Finish Console Wiring, Export Logs Through Tauri, And Update Phase 1 Docs

**Files:**
- Modify: `launcher/src/pages/ConsolePage/ConsolePage.tsx`
- Modify: `launcher/src/pages/ConsolePage/ConsolePage.test.tsx`
- Modify: `launcher/src/layouts/AppShell/AppShell.tsx`
- Modify: `README.md`

- [ ] **Step 1: Write failing console and docs-oriented tests**

Update `launcher/src/pages/ConsolePage/ConsolePage.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { ConsolePage } from './ConsolePage';

describe('ConsolePage', () => {
  it('renders runtime summary metadata and triggers toolbar actions', async () => {
    const user = userEvent.setup();
    const onCopyLog = vi.fn();
    const onClearLogs = vi.fn();
    const onExportLogs = vi.fn();

    render(
      <ConsolePage
        runtimeDriver="uv"
        tasks={[
          {
            taskId: 'task-1',
            target: 'genie-base',
            label: 'GenieData 基础资源',
            status: 'downloading',
            message: '正在下载',
            progressCurrent: 1,
            progressTotal: 3,
            updatedAt: '1712300001',
          },
        ]}
        logs={[
          {
            id: 'log-1',
            time: '2026-04-05 15:00:00',
            kind: 'system',
            text: 'genie-base: 正在下载',
          },
        ]}
        autoScroll={true}
        wrapLines={true}
        onSetAutoScroll={() => undefined}
        onSetWrapLines={() => undefined}
        onClearLogs={onClearLogs}
        onCopyLog={onCopyLog}
        onExportLogs={onExportLogs}
      />,
    );

    expect(screen.getByText('运行驱动 uv')).toBeInTheDocument();
    expect(screen.getByText('当前任务 GenieData 基础资源')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '复制日志 1' }));
    expect(onCopyLog).toHaveBeenCalledWith('genie-base: 正在下载');

    await user.click(screen.getByRole('button', { name: '导出日志' }));
    expect(onExportLogs).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run the console test to verify it fails**

Run: `npm --prefix launcher run test -- --run src/pages/ConsolePage/ConsolePage.test.tsx`

Expected: FAIL because `ConsolePage` still expects launch-state props and does not show runtime driver or queue metadata yet.

- [ ] **Step 3: Update ConsolePage and add the minimal phase 1 README section**

Update `launcher/src/pages/ConsolePage/ConsolePage.tsx`:

```tsx
import type { ConsoleLogEntry } from '../../services/launcher/launcher';
import type { RuntimeTaskRecord } from '../../services/runtime/runtime';
import '../../styles/console.css';

interface ConsolePageProps {
  runtimeDriver: 'uv';
  tasks: RuntimeTaskRecord[];
  logs: ConsoleLogEntry[];
  autoScroll: boolean;
  wrapLines: boolean;
  onSetAutoScroll: (next: boolean) => void;
  onSetWrapLines: (next: boolean) => void;
  onClearLogs: () => void;
  onCopyLog: (text: string) => void;
  onExportLogs: () => void;
}

export function ConsolePage({
  runtimeDriver,
  tasks,
  logs,
  autoScroll,
  wrapLines,
  onSetAutoScroll,
  onSetWrapLines,
  onClearLogs,
  onCopyLog,
  onExportLogs,
}: ConsolePageProps) {
  const activeTask =
    tasks.find((task) =>
      ['queued', 'preparing', 'downloading', 'verifying'].includes(task.status),
    ) ?? null;
  const lastLog = logs[logs.length - 1];

  return (
    <div className="console-page">
      <header className="console-toolbar">
        <div className="console-toolbar__status">
          <span className="console-status console-status--running">
            {activeTask ? activeTask.status : 'idle'}
          </span>
          <div className="console-toolbar__meta">
            <span className="console-toolbar__label">运行驱动 {runtimeDriver}</span>
            <span className="console-command">
              {activeTask ? `当前任务 ${activeTask.label}` : '当前没有活动任务'}
            </span>
          </div>
        </div>

        <div className="console-toolbar__actions">
          <button type="button" onClick={onClearLogs}>
            清空日志
          </button>
          <button type="button" onClick={onExportLogs}>
            导出日志
          </button>
          <button
            type="button"
            aria-pressed={autoScroll}
            onClick={() => onSetAutoScroll(!autoScroll)}
          >
            自动滚动
          </button>
          <button
            type="button"
            aria-pressed={wrapLines}
            onClick={() => onSetWrapLines(!wrapLines)}
          >
            换行
          </button>
        </div>
      </header>

      <section className={`console-log-panel${wrapLines ? ' is-wrap' : ''}`}>
        {logs.length === 0 ? (
          <div className="console-empty">
            <h2>尚无运行日志</h2>
            <p>开始检查环境或下载资源后，这里会显示结构化事件和原始输出</p>
          </div>
        ) : (
          <div className="console-log-list">
            {logs.map((entry, index) => (
              <article key={entry.id} className={`console-log console-log--${entry.kind}`}>
                <div className="console-log__meta">
                  <span>{entry.time}</span>
                  <span>{entry.kind}</span>
                </div>
                <pre className="console-log__text">{entry.text}</pre>
                <button
                  type="button"
                  className="console-log__copy"
                  aria-label={`复制日志 ${index + 1}`}
                  onClick={() => onCopyLog(entry.text)}
                >
                  复制
                </button>
              </article>
            ))}
          </div>
        )}
      </section>

      <footer className="console-footer">
        <span>日志条数 {logs.length}</span>
        <span>队列任务 {tasks.length}</span>
        <span>最后更新时间 {lastLog ? lastLog.time : '暂无'}</span>
      </footer>
    </div>
  );
}
```

Update `launcher/src/layouts/AppShell/AppShell.tsx` so `renderPage` passes `runtimeDriver={inspection?.runtimeDriver ?? 'uv'}` and `tasks={tasks}` into console rendering.

Append this phase 1 section to `README.md`:

````md
## 第一阶段开发环境

当前联调主路径统一使用 `uv` 和 `pyproject.toml` 的依赖分组，不再围绕 `environment.gpu.yml` 组织启动链路。

CPU：

```bash
uv sync --group cpu --group genie-tts
```

GPU：

```bash
uv sync --group gpu --group genie-tts
```

运行时检查：

```bash
uv run python -m xnnehanglab_tts.cli inspect-runtime
```

GenieData 状态校验：

```bash
uv run python -m xnnehanglab_tts.cli verify genie-base
```

说明：

- Launcher 第一阶段固定通过 `uv` 驱动运行时
- `runtime_driver` 和 `python_path` 已在设置页预留，但目前只读
- `GenieData` 下载到 `models/genie/base/GenieData`
````

- [ ] **Step 4: Run the console test and the full verification set**

Run: `npm --prefix launcher run test -- --run src/pages/ConsolePage/ConsolePage.test.tsx`

Expected: PASS with runtime-driver metadata and export action visible.

Run: `uv run pytest tests/runtime -v`

Expected: PASS with all root runtime tests green.

Run: `npm --prefix launcher run test -- --run`

Expected: PASS with launcher component and integration tests green.

Run: `cargo test --manifest-path launcher/src-tauri/Cargo.toml`

Expected: PASS with runtime-state tests green.

Run: `cargo check --manifest-path launcher/src-tauri/Cargo.toml`

Expected: PASS without Rust compile errors.

- [ ] **Step 5: Commit**

```bash
git add README.md
git -C launcher add src/pages/ConsolePage/ConsolePage.tsx src/pages/ConsolePage/ConsolePage.test.tsx src/layouts/AppShell/AppShell.tsx
git -C launcher commit -m "feat: wire launcher console to runtime events"
git add README.md
git commit -m "docs: add phase 1 runtime workflow"
```
