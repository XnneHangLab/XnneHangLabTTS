from __future__ import annotations

import gc
import io
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from xnnehanglab_tts.runtime.config import load_runtime_config

Float32Array = NDArray[np.float32]

DEFAULT_SAMPLE_RATE = 24000

_qwen_tts_engine: Any | None = None
_loaded_model_name: str | None = None
_loaded_model_source: str | None = None
_sample_rate: int = DEFAULT_SAMPLE_RATE
_model_lock = threading.Lock()


def _config_path_from_env() -> Path | None:
    value = os.getenv("XH_RUNTIME_CONFIG")
    return Path(value) if value else None


def _load_runtime_paths():
    _, paths = load_runtime_config(_config_path_from_env())
    return paths


def _resolve_model_source(model_name: str) -> str:
    paths = _load_runtime_paths()
    if model_name == "0.6b":
        local = paths.qwen_tts_0_6b_root
        hf_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    else:
        local = paths.qwen_tts_1_7b_root
        hf_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    if local.exists() and any(local.iterdir()):
        return str(local)
    return hf_id


def _resolve_device() -> str:
    if device := os.environ.get("XH_QWEN_TTS_DEVICE", "").strip():
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _release_engine() -> None:
    global _qwen_tts_engine, _loaded_model_name, _loaded_model_source
    old = _qwen_tts_engine
    _qwen_tts_engine = None
    _loaded_model_name = None
    _loaded_model_source = None
    if old is not None:
        del old
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_qwen_tts_status() -> dict[str, Any]:
    return {
        "loaded": _qwen_tts_engine is not None,
        "loaded_model": _loaded_model_name,
        "loaded_model_source": _loaded_model_source,
        "sample_rate": _sample_rate,
        "device": _resolve_device(),
    }


def load_qwen_tts_model(model_name: str = "0.6b") -> None:
    global _qwen_tts_engine, _loaded_model_name, _loaded_model_source, _sample_rate

    if model_name not in ("0.6b", "1.7b"):
        raise ValueError(f"Unsupported Qwen-TTS model: {model_name!r}. Use '0.6b' or '1.7b'.")

    with _model_lock:
        if _qwen_tts_engine is not None and _loaded_model_name == model_name:
            return

        if _qwen_tts_engine is not None:
            print(
                f"INFO: qwen-tts: releasing {_loaded_model_name} before loading {model_name}",
                flush=True,
            )
            _release_engine()

        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as exc:
            raise RuntimeError("faster-qwen3-tts is not installed") from exc

        model_source = _resolve_model_source(model_name)
        device = _resolve_device()
        print(
            f"INFO: qwen-tts: loading model={model_name}, source={model_source}, device={device}",
            flush=True,
        )

        kwargs: dict[str, Any] = {"device": device}
        if device == "cuda":
            try:
                import torch
                kwargs["dtype"] = torch.bfloat16
            except Exception:
                pass

        _qwen_tts_engine = FasterQwen3TTS.from_pretrained(model_source, **kwargs)
        _sample_rate = DEFAULT_SAMPLE_RATE
        _loaded_model_name = model_name
        _loaded_model_source = model_source
        print(f"INFO: qwen-tts: model loaded: {model_name}", flush=True)


def synthesize_once(
    *,
    text: str,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
) -> Path:
    if _qwen_tts_engine is None:
        raise RuntimeError("Qwen-TTS model is not loaded. Load it first.")

    ref_audio_str = str(ref_audio) if ref_audio is not None else None
    ref_text_str = ref_text or ""

    started = time.perf_counter()
    with _model_lock:
        result = _qwen_tts_engine.generate_voice_clone(
            text=text,
            language="Auto",
            ref_audio=ref_audio_str,
            ref_text=ref_text_str,
        )

    audio_arrays, sample_rate_raw = result
    sr = int(sample_rate_raw) if isinstance(sample_rate_raw, (int, float)) else _sample_rate

    audio = _concat_audio(audio_arrays)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)

    wav_bytes = _to_wav_bytes(audio, sr)

    paths = _load_runtime_paths()
    output_dir = paths.cache_root / "qwen-tts"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    fragment = re.sub(r"_+", "_", re.sub(r"[^\w]", "_", text[:24].strip())).strip("_") or "audio"
    output_path = output_dir / f"{fragment}_{timestamp}.wav"
    output_path.write_bytes(wav_bytes)

    elapsed = time.perf_counter() - started
    print(
        f"INFO: qwen-tts: synthesize_once done: "
        f"text_len={len(text)}, audio_bytes={len(wav_bytes)}, elapsed={elapsed:.2f}s",
        flush=True,
    )
    return output_path


def _concat_audio(audio_arrays: Any) -> Float32Array:
    if isinstance(audio_arrays, np.ndarray):
        return np.asarray(audio_arrays, dtype=np.float32).squeeze()

    parts: list[Float32Array] = []
    try:
        for chunk in audio_arrays:
            arr = np.asarray(chunk, dtype=np.float32).squeeze()
            if arr.size > 0:
                parts.append(arr)
    except TypeError:
        arr = np.asarray(audio_arrays, dtype=np.float32).squeeze()
        if arr.size > 0:
            parts.append(arr)

    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _to_wav_bytes(pcm: Float32Array, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
