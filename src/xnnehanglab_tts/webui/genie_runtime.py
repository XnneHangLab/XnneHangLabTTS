from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from xnnehanglab_tts.runtime.config import load_runtime_config


@dataclass
class GenieRuntimeState:
    loaded_character: str | None = None
    genie_data_dir: str | None = None


_STATE = GenieRuntimeState()


def _config_path_from_env() -> Path | None:
    value = os.getenv("XH_RUNTIME_CONFIG")
    return Path(value) if value else None


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bundled_genie_src(repo_root: Path) -> Path:
    return repo_root / "packages" / "Genie-TTS" / "src"


def _ensure_bundled_genie_on_path(repo_root: Path) -> Path:
    bundled_src = _bundled_genie_src(repo_root)
    if not bundled_src.is_dir():
        raise RuntimeError(f"内置 Genie-TTS 包不存在: {bundled_src}")

    bundled_str = str(bundled_src)
    if bundled_str in sys.path:
        sys.path.remove(bundled_str)
    sys.path.insert(0, bundled_str)
    return bundled_src


def _clear_imported_genie_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "genie_tts" or module_name.startswith("genie_tts."):
            sys.modules.pop(module_name, None)


def _load_runtime_paths():
    _, paths = load_runtime_config(_config_path_from_env())
    return paths


def _load_genie_module(paths):
    repo_root = _resolve_repo_root()
    bundled_src = _ensure_bundled_genie_on_path(repo_root)
    genie_data_dir = str(paths.genie_base_root)

    if _STATE.genie_data_dir and _STATE.genie_data_dir != genie_data_dir:
        _clear_imported_genie_modules()
        _STATE.loaded_character = None

    os.environ["GENIE_DATA_DIR"] = genie_data_dir
    _STATE.genie_data_dir = genie_data_dir

    try:
        return import_module("genie_tts")
    except ModuleNotFoundError as exc:
        if exc.name == "genie_tts":
            raise RuntimeError(f"找不到内置 genie_tts 包: {bundled_src}") from exc
        raise RuntimeError(f"Genie-TTS 依赖缺失: {exc.name}") from exc
    except Exception as exc:
        raise RuntimeError(f"加载 Genie-TTS 失败: {exc}") from exc


def _resolve_character_model_dir(character_name: str, paths) -> Path:
    model_dir = paths.genie_tts_root / character_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"角色模型目录不存在: {model_dir}")
    return model_dir


def list_available_models() -> list[str]:
    try:
        paths = _load_runtime_paths()
    except Exception as exc:
        print(f"ERROR: 读取运行时配置失败: {exc}", flush=True)
        return []

    if not paths.genie_tts_root.is_dir():
        return []

    return sorted(
        entry.name
        for entry in paths.genie_tts_root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def get_genie_tts_status() -> dict[str, object]:
    return {
        "loaded": _STATE.loaded_character is not None,
        "loaded_character": _STATE.loaded_character,
    }


def load_genie_tts_model_by_name(character_name: str) -> None:
    paths = _load_runtime_paths()
    model_dir = _resolve_character_model_dir(character_name, paths)
    genie = _load_genie_module(paths)

    if _STATE.loaded_character and _STATE.loaded_character != character_name:
        try:
            genie.unload_character(_STATE.loaded_character)
        except Exception as exc:
            print(f"WARNING: 卸载旧模型失败: {exc}", flush=True)

    genie.load_character(
        character_name=character_name,
        onnx_model_dir=str(model_dir),
        language="auto",
        use_roberta=False,
    )
    _STATE.loaded_character = character_name


async def synthesize_once(
    text: str,
    ref_audio: Path | None,
    ref_text: str | None,
) -> bytes:
    if not _STATE.loaded_character:
        raise RuntimeError("模型尚未加载，请先选择角色并点击「加载模型」")
    if ref_audio is None:
        raise ValueError("请先提供参考音频")
    if not ref_audio.is_file():
        raise ValueError(f"参考音频不存在: {ref_audio}")

    normalized_ref_text = (ref_text or "").strip()
    if not normalized_ref_text:
        raise ValueError("请先提供参考文本")

    paths = _load_runtime_paths()
    genie = _load_genie_module(paths)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        await asyncio.to_thread(
            genie.set_reference_audio,
            character_name=_STATE.loaded_character,
            audio_path=str(ref_audio),
            audio_text=normalized_ref_text,
            language="auto",
            use_roberta=False,
        )
        await asyncio.to_thread(
            genie.tts,
            character_name=_STATE.loaded_character,
            text=text,
            play=False,
            split_sentence=True,
            save_path=str(output_path),
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Genie-TTS 未生成音频输出")
        return output_path.read_bytes()
    finally:
        output_path.unlink(missing_ok=True)
