from __future__ import annotations

from datetime import datetime
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from xnnehanglab_tts.runtime.config import load_runtime_config


@dataclass
class GenieRuntimeState:
    loaded_character: str | None = None
    genie_data_dir: str | None = None
    genie_module: object = None
    ref_audio_key: tuple[str, str] | None = None


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


def _patch_genie_resource_paths(genie_data_dir: Path) -> None:
    base_dir = genie_data_dir.resolve()
    fasttext_model_path = base_dir / "lid.176.bin"
    overrides = {
        "GENIE_DATA_DIR": str(base_dir),
        "English_G2P_DIR": str(base_dir / "G2P" / "EnglishG2P"),
        "Chinese_G2P_DIR": str(base_dir / "G2P" / "ChineseG2P"),
        "HUBERT_MODEL_DIR": str(base_dir / "chinese-hubert-base"),
        "SV_MODEL": str(base_dir / "speaker_encoder.onnx"),
        "ROBERTA_MODEL_DIR": str(base_dir / "RoBERTa"),
        "FTLANG_CACHE": str(base_dir),
    }

    os.environ.update(overrides)

    resources_module = sys.modules.get("genie_tts.Core.Resources")
    if resources_module is not None:
        for key, value in overrides.items():
            setattr(resources_module, key, value)

    model_manager_module = sys.modules.get("genie_tts.ModelManager")
    if model_manager_module is not None:
        setattr(model_manager_module, "HUBERT_MODEL_DIR", overrides["HUBERT_MODEL_DIR"])
        setattr(model_manager_module, "SV_MODEL", overrides["SV_MODEL"])
        setattr(model_manager_module, "ROBERTA_MODEL_DIR", overrides["ROBERTA_MODEL_DIR"])
        gsv_model_file = getattr(model_manager_module, "GSVModelFile", None)
        if gsv_model_file is not None:
            setattr(gsv_model_file, "HUBERT_MODEL", str(base_dir / "chinese-hubert-base" / "chinese-hubert-base.onnx"))
            setattr(
                gsv_model_file,
                "HUBERT_MODEL_WEIGHT_FP16",
                str(base_dir / "chinese-hubert-base" / "chinese-hubert-base_weights_fp16.bin")
            )
            setattr(gsv_model_file, "ROBERTA_MODEL", str(base_dir / "RoBERTa" / "RoBERTa.onnx"))
            setattr(gsv_model_file, "ROBERTA_TOKENIZER", str(base_dir / "RoBERTa" / "roberta_tokenizer"))

    fast_langdetect_infer = sys.modules.get("fast_langdetect.infer")
    if fast_langdetect_infer is not None:
        default_detector = getattr(fast_langdetect_infer, "_default_detector", None)
        previous_cache_directory = getattr(fast_langdetect_infer, "CACHE_DIRECTORY", None)
        setattr(fast_langdetect_infer, "CACHE_DIRECTORY", overrides["FTLANG_CACHE"])
        detector_config = getattr(default_detector, "config", None)
        should_clear_model_cache = previous_cache_directory != overrides["FTLANG_CACHE"]
        if detector_config is not None:
            should_clear_model_cache = should_clear_model_cache or (
                getattr(detector_config, "cache_dir", None) != overrides["FTLANG_CACHE"]
            )
            should_clear_model_cache = should_clear_model_cache or (
                getattr(detector_config, "custom_model_path", None) != str(fasttext_model_path)
            )
            should_clear_model_cache = should_clear_model_cache or (
                getattr(detector_config, "model", None) != "full"
            )
            setattr(detector_config, "cache_dir", overrides["FTLANG_CACHE"])
            setattr(detector_config, "custom_model_path", str(fasttext_model_path))
            setattr(detector_config, "model", "full")
        if should_clear_model_cache and default_detector is not None:
            model_cache = getattr(default_detector, "_models", None)
            if isinstance(model_cache, dict):
                model_cache.clear()


def _load_genie_module(paths):
    repo_root = _resolve_repo_root()
    bundled_src = _ensure_bundled_genie_on_path(repo_root)
    genie_data_dir = paths.genie_base_root.resolve()

    existing_resources = sys.modules.get("genie_tts.Core.Resources")
    if existing_resources is not None:
        existing_data_dir = getattr(existing_resources, "GENIE_DATA_DIR", None)
        if existing_data_dir and Path(existing_data_dir).resolve() != genie_data_dir:
            _clear_imported_genie_modules()
            _STATE.loaded_character = None

    if _STATE.genie_data_dir and Path(_STATE.genie_data_dir).resolve() != genie_data_dir:
        _clear_imported_genie_modules()
        _STATE.loaded_character = None

    _patch_genie_resource_paths(genie_data_dir)
    _STATE.genie_data_dir = str(genie_data_dir)

    try:
        module = import_module("genie_tts")
        _patch_genie_resource_paths(genie_data_dir)
        return module
    except ModuleNotFoundError as exc:
        traceback.print_exc(file=sys.stdout)
        if exc.name == "genie_tts":
            raise RuntimeError(f"找不到内置 genie_tts 包: {bundled_src}") from exc
        raise RuntimeError(f"Genie-TTS 依赖缺失: {exc.name}") from exc
    except Exception as exc:
        traceback.print_exc(file=sys.stdout)
        raise RuntimeError(f"加载 Genie-TTS 失败: {exc}") from exc


def _resolve_character_model_dir(character_name: str, paths) -> Path:
    model_dir = paths.genie_tts_root / character_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"角色模型目录不存在: {model_dir}")
    return model_dir


def _sanitize_output_name_fragment(text: str, *, max_length: int = 24) -> str:
    normalized = re.sub(r"\s+", "_", text.strip())
    normalized = "".join(char if char.isalnum() else "_" for char in normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    if not normalized:
        return "audio"
    clipped = normalized[:max_length].rstrip("._")
    return clipped or "audio"


def _build_synthesis_output_path(text: str, paths) -> Path:
    output_dir = paths.cache_root / "genie-tts"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"{_sanitize_output_name_fragment(text)}_{timestamp}.wav"
    return output_dir / filename

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
        use_roberta=True,
    )
    _STATE.loaded_character = character_name
    _STATE.genie_module = genie
    _STATE.ref_audio_key = None


def synthesize_once(
    text: str,
    ref_audio: Path | None,
    ref_text: str | None,
) -> Path:
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
    genie = _STATE.genie_module
    if genie is None:
        genie = _load_genie_module(paths)
    total_started_at = time.perf_counter()
    output_path = _build_synthesis_output_path(text, paths)

    try:
        ref_audio_key = (str(ref_audio), normalized_ref_text)
        reference_started_at = time.perf_counter()
        if _STATE.ref_audio_key != ref_audio_key:
            genie.set_reference_audio(
                character_name=_STATE.loaded_character,
                audio_path=str(ref_audio),
                audio_text=normalized_ref_text,
                language="auto",
                use_roberta=True,
            )
            _STATE.ref_audio_key = ref_audio_key
        reference_elapsed = time.perf_counter() - reference_started_at
        synth_started_at = time.perf_counter()
        genie.tts(
            character_name=_STATE.loaded_character,
            text=text,
            play=False,
            split_sentence=True,
            save_path=str(output_path),
        )
        synth_elapsed = time.perf_counter() - synth_started_at
        if not output_path.is_file() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("Genie-TTS 未生成音频输出")
        total_elapsed = time.perf_counter() - total_started_at
        print(
            "INFO: Genie-TTS timing: "
            f"reference={reference_elapsed:.2f}s, "
            f"synthesis={synth_elapsed:.2f}s, "
            f"total={total_elapsed:.2f}s",
            flush=True,
        )
        return output_path
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
