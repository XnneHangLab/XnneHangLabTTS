from __future__ import annotations

import asyncio
import gc
import io
import json
import os
import pickle
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf

from xnnehanglab_tts.runtime.config import load_runtime_config


DEFAULT_SAMPLE_RATE = 32000
_GSV_LITE_GPT_CACHE = [(1, 512), (1, 1024), (1, 2048), (4, 512), (4, 1024)]
_GSV_LITE_SEGMENT_MAX_CHARS = 80
_GSV_LITE_SEGMENT_SILENCE_S = 0.08
_JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9f]")
_GSV_LITE_MODEL_DIRNAME = "gsv-tts-lite"
_gsv_lite_monkey_patch_applied = False

_gsv_lite_engine: Any | None = None
_loaded_gpt_path: str | None = None
_loaded_sovits_path: str | None = None
_loaded_character_name: str | None = None
_model_lock = threading.Lock()


@dataclass(frozen=True)
class GSVLiteModelSpec:
    character_name: str
    character_dir: Path
    gpt_path: Path
    sovits_path: Path
    models_dir: Path


# ── Config / path helpers ────────────────────────────────────────────────────

def _config_path_from_env() -> Path | None:
    value = os.getenv("XH_RUNTIME_CONFIG")
    return Path(value) if value else None


def _load_runtime_paths():
    _, paths = load_runtime_config(_config_path_from_env())
    return paths


def _resolve_gsv_tts_lite_root(paths) -> Path:
    return paths.models_root / _GSV_LITE_MODEL_DIRNAME


def list_available_characters() -> list[str]:
    try:
        paths = _load_runtime_paths()
    except Exception as exc:
        print(f"ERROR: 读取运行时配置失败: {exc}", flush=True)
        return []

    chars_root = _resolve_gsv_tts_lite_root(paths)
    if not chars_root.is_dir():
        return []

    return sorted(
        entry.name
        for entry in chars_root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def _resolve_infer_config_path(character_dir: Path) -> Path:
    for name in ("infer_config.json", "infer.json"):
        candidate = character_dir / name
        if candidate.exists():
            return candidate
    return character_dir / "infer_config.json"


def _load_infer_config(character_dir: Path) -> dict[str, Any]:
    config_path = _resolve_infer_config_path(character_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"infer config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid infer config: {config_path}")
    return cast("dict[str, Any]", data)


def _resolve_model_file(character_dir: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(f"{label} missing in infer config: {character_dir}")
    resolved = (character_dir / raw_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _get_model_spec(character_name: str, paths) -> GSVLiteModelSpec:
    character_dir = _resolve_gsv_tts_lite_root(paths) / character_name
    if not character_dir.exists():
        raise FileNotFoundError(f"角色目录不存在: {character_dir}")
    infer_config = _load_infer_config(character_dir)
    gpt_path = _resolve_model_file(character_dir, infer_config.get("gpt_path"), "gpt_path")
    sovits_path = _resolve_model_file(character_dir, infer_config.get("sovits_path"), "sovits_path")
    return GSVLiteModelSpec(
        character_name=character_name,
        character_dir=character_dir,
        gpt_path=gpt_path,
        sovits_path=sovits_path,
        models_dir=paths.gsv_lite_root,
    )


def get_gsv_lite_status() -> dict[str, Any]:
    return {
        "loaded": _gsv_lite_engine is not None,
        "loaded_character": _loaded_character_name,
    }


# ── Engine lifecycle ─────────────────────────────────────────────────────────

def _release_engine() -> None:
    global _gsv_lite_engine, _loaded_gpt_path, _loaded_sovits_path, _loaded_character_name

    old_engine = _gsv_lite_engine
    _gsv_lite_engine = None
    _loaded_gpt_path = None
    _loaded_sovits_path = None
    _loaded_character_name = None
    if old_engine is not None:
        del old_engine

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        pass


# ── Japanese / OpenJTalk ─────────────────────────────────────────────────────

def _prepend_env_path(var_name: str, path: Path) -> None:
    path_str = str(path)
    current = os.environ.get(var_name, "")
    parts = [p for p in current.split(os.pathsep) if p]
    os.environ[var_name] = os.pathsep.join([path_str, *[p for p in parts if p != path_str]])


def _configure_openjtalk(models_dir: Path) -> None:
    ja_dir = models_dir / "g2p" / "ja"
    openjtalk_dict_dir = ja_dir / "open_jtalk_dic_utf_8-1.11"
    user_dict_csv = ja_dir / "userdict.csv"
    user_dict_bin = ja_dir / "user.dict"

    if not (openjtalk_dict_dir.is_dir() or user_dict_csv.is_file() or user_dict_bin.is_file()):
        return

    if openjtalk_dict_dir.is_dir():
        os.environ["OPEN_JTALK_DICT_DIR"] = str(openjtalk_dict_dir)

    try:
        import pyopenjtalk
    except Exception as exc:
        print(f"WARNING: gsv-lite: failed to import pyopenjtalk: {exc}", flush=True)
        return

    if openjtalk_dict_dir.is_dir():
        try:
            pyopenjtalk.OPEN_JTALK_DICT_DIR = str(openjtalk_dict_dir).encode("utf-8")
            unset = getattr(pyopenjtalk, "unset_user_dict", None)
            if callable(unset):
                unset()
        except Exception as exc:
            print(f"WARNING: gsv-lite: OpenJTalk dict activation failed: {exc}", flush=True)

    if user_dict_csv.is_file() and not user_dict_bin.is_file():
        try:
            cast("Any", pyopenjtalk).mecab_dict_index(str(user_dict_csv), str(user_dict_bin))
        except Exception as exc:
            print(f"WARNING: gsv-lite: failed to build user dict: {exc}", flush=True)

    if user_dict_bin.is_file():
        try:
            cast("Any", pyopenjtalk).update_global_jtalk_with_user_dict(str(user_dict_bin))
        except Exception as exc:
            print(f"WARNING: gsv-lite: failed to activate user dict: {exc}", flush=True)


# ── English CMU dict ─────────────────────────────────────────────────────────

def _read_cmu_dict(file_path: Path) -> dict[str, list[list[str]]]:
    g2p_dict: dict[str, list[list[str]]] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) < 2:
                continue
            word = re.sub(r"\(\d+\)$", "", parts[0].lower())
            g2p_dict.setdefault(word, []).append(parts[1].split(" "))
    return g2p_dict


def _normalize_cached_dict(raw: object) -> dict[str, list[list[str]]]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[list[str]]] = {}
    for raw_word, raw_prons in cast("dict[object, object]", raw).items():
        if not isinstance(raw_prons, list):
            continue
        prons: list[list[str]] = []
        for p in cast("list[object]", raw_prons):
            if isinstance(p, (list, tuple)):
                prons.append([str(x) for x in p])
        if prons:
            result[str(raw_word).lower()] = prons
    return result


def _load_english_dict(models_dir: Path) -> dict[str, list[list[str]]]:
    en_dir = models_dir / "g2p" / "en"
    g2p_dict: dict[str, list[list[str]]] = {}

    cache_path = en_dir / "engdict_cache.pickle"
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                g2p_dict = _normalize_cached_dict(pickle.load(f))
        except Exception as exc:
            print(f"WARNING: gsv-lite: failed to load EN dict cache: {exc}", flush=True)

    if not g2p_dict:
        for fname in ("cmudict.rep",):
            p = en_dir / fname
            if p.is_file():
                g2p_dict.update(_read_cmu_dict(p))

    for fname in ("cmudict-fast.rep",):
        p = en_dir / fname
        if p.is_file():
            for word, prons in _read_cmu_dict(p).items():
                g2p_dict.setdefault(word, prons)

    hot = en_dir / "engdict-hot.rep"
    if hot.is_file():
        g2p_dict.update(_read_cmu_dict(hot))

    return g2p_dict


def _configure_nltk(models_dir: Path) -> None:
    nltk_dir = models_dir / "g2p" / "en" / "nltk"
    if not nltk_dir.is_dir():
        return

    _prepend_env_path("NLTK_DATA", nltk_dir)
    try:
        import nltk
        nltk_dir_str = str(nltk_dir)
        current = [str(p) for p in cast("list[object]", getattr(nltk.data, "path", []))]
        nltk.data.path = [nltk_dir_str, *[p for p in current if p != nltk_dir_str]]
    except Exception as exc:
        print(f"WARNING: gsv-lite: failed to configure NLTK path: {exc}", flush=True)
        return

    cmu_dict = _load_english_dict(models_dir)
    if cmu_dict:
        try:
            from nltk.corpus import cmudict as nltk_cmudict
            cast("Any", nltk_cmudict).dict = lambda: cmu_dict
        except Exception as exc:
            print(f"WARNING: gsv-lite: failed to configure local CMU dict: {exc}", flush=True)


# ── Japanese G2P monkey patch ────────────────────────────────────────────────

def _redistribute_word2ph(words: list[str], phone_count: int) -> dict[str, list[Any]]:
    if phone_count <= 0:
        return {"word": words[:1], "ph": [0] if words else []}
    if not words:
        return {"word": ["?"], "ph": [phone_count]}
    if phone_count < len(words):
        return {"word": ["".join(words)], "ph": [phone_count]}
    base = phone_count // len(words)
    extra = phone_count % len(words)
    return {"word": list(words), "ph": [base + (1 if i < extra else 0) for i in range(len(words))]}


def _repair_word2ph(word2ph: object, phone_count: int) -> dict[str, list[Any]]:
    if not isinstance(word2ph, dict):
        return _redistribute_word2ph([], phone_count)
    mapping = cast("dict[str, object]", word2ph)
    raw_words = mapping.get("word")
    raw_counts = mapping.get("ph")
    if not isinstance(raw_words, list) or not isinstance(raw_counts, list):
        return _redistribute_word2ph([], phone_count)
    words = [str(w) for w in cast("list[object]", raw_words)]
    try:
        counts = [max(int(cast("Any", c)), 0) for c in cast("list[object]", raw_counts)]
    except Exception:
        return _redistribute_word2ph(words, phone_count)
    if not words or len(words) != len(counts):
        return _redistribute_word2ph(words, phone_count)

    diff = phone_count - sum(counts)
    if diff == 0:
        return {"word": words, "ph": counts}

    indices = list(range(len(counts) - 1, -1, -1))
    if diff > 0:
        step = 0
        while diff > 0:
            counts[indices[step % len(indices)]] += 1
            diff -= 1
            step += 1
        return {"word": words, "ph": counts}

    remaining = -diff
    step = 0
    limit = max(1, len(indices) * (remaining + 1))
    while remaining > 0 and step < limit:
        idx = indices[step % len(indices)]
        if counts[idx] > 1:
            counts[idx] -= 1
            remaining -= 1
        step += 1
    if remaining > 0:
        return _redistribute_word2ph(words, phone_count)
    return {"word": words, "ph": counts}


def _apply_monkey_patch() -> None:
    global _gsv_lite_monkey_patch_applied
    if _gsv_lite_monkey_patch_applied:
        return

    # torchaudio 2.11+ defaults to torchcodec as the audio backend on Windows,
    # which requires FFmpeg DLLs.  When torchcodec DLLs are missing, every
    # torchaudio.load() call raises ImportError at inference time.
    # Patch torchaudio.load to use soundfile instead (no FFmpeg required).
    try:
        import torchaudio
        if hasattr(torchaudio, "_torchcodec"):
            import soundfile as _sf
            import torch as _torch

            def _load_via_soundfile(uri, *_args, **_kwargs):
                data, sr = _sf.read(str(uri), dtype="float32", always_2d=True)
                # soundfile: (samples, channels) → torchaudio: (channels, samples)
                return _torch.from_numpy(data.T.copy()), sr

            torchaudio.load = _load_via_soundfile
            print(
                "INFO: gsv-lite: patched torchaudio.load → soundfile "
                "(torchcodec requires FFmpeg which is unavailable)",
                flush=True,
            )
    except Exception as exc:
        print(f"WARNING: gsv-lite: torchaudio.load patch failed: {exc}", flush=True)

    try:
        from gsv_tts.GPT_SoVITS.G2P.Japanese.japanese import JapaneseG2P
    except Exception:
        _gsv_lite_monkey_patch_applied = True
        return

    original = cast("Any", JapaneseG2P).g2p
    if getattr(original, "_xnnehanglab_patched", False):
        _gsv_lite_monkey_patch_applied = True
        return

    def patched(self: Any, norm_text: str, with_prosody: bool = True) -> tuple[list[str], dict[str, list[Any]]]:
        phones, word2ph = original(self, norm_text, with_prosody)
        try:
            original_total = sum(int(c) for c in word2ph["ph"])
        except Exception:
            original_total = None
        if original_total == len(phones):
            return phones, word2ph
        repaired = _repair_word2ph(word2ph, len(phones))
        print(
            f"WARNING: gsv-lite: repaired Japanese word2ph: text={norm_text!r}, "
            f"phones={len(phones)}, was={original_total}, now={sum(repaired['ph'])}",
            flush=True,
        )
        return phones, repaired

    cast("Any", patched)._xnnehanglab_patched = True
    JapaneseG2P.g2p = patched
    _gsv_lite_monkey_patch_applied = True
    print("INFO: gsv-lite: applied JapaneseG2P.g2p monkey patch", flush=True)


# ── Model loading ────────────────────────────────────────────────────────────

def load_gsv_lite_model(character_name: str, *, use_bert: bool = False) -> None:
    global _gsv_lite_engine, _loaded_gpt_path, _loaded_sovits_path, _loaded_character_name

    paths = _load_runtime_paths()
    spec = _get_model_spec(character_name, paths)

    with _model_lock:
        if (
            _gsv_lite_engine is not None
            and _loaded_gpt_path == str(spec.gpt_path)
            and _loaded_sovits_path == str(spec.sovits_path)
        ):
            return

        if _gsv_lite_engine is not None:
            _release_engine()

        _configure_openjtalk(spec.models_dir)
        _configure_nltk(spec.models_dir)

        try:
            from gsv_tts import TTS
        except Exception as exc:
            raise RuntimeError("gsv-tts-lite 未安装") from exc

        _apply_monkey_patch()

        print(
            f"INFO: gsv-lite: loading character={spec.character_name} "
            f"gpt={spec.gpt_path.name} sovits={spec.sovits_path.name} use_bert={use_bert}",
            flush=True,
        )
        engine = TTS(
            models_dir=str(spec.models_dir),
            gpt_cache=_GSV_LITE_GPT_CACHE,
            use_bert=use_bert,
        )
        engine.load_gpt_model(str(spec.gpt_path))
        engine.load_sovits_model(str(spec.sovits_path))

        _gsv_lite_engine = engine
        _loaded_gpt_path = str(spec.gpt_path)
        _loaded_sovits_path = str(spec.sovits_path)
        _loaded_character_name = spec.character_name
        print(f"INFO: gsv-lite: load complete: character={spec.character_name}", flush=True)


# ── Inference ────────────────────────────────────────────────────────────────

def _wav_bytes_from_clips(clips: list[Any]) -> bytes:
    if not clips:
        raise RuntimeError("clip list is empty")
    samplerate = int(clips[0].samplerate)
    silence_samples = max(int(samplerate * _GSV_LITE_SEGMENT_SILENCE_S), 0)
    silence = np.zeros(silence_samples, dtype=np.float32) if silence_samples else None
    arrays: list[Any] = []
    for i, clip in enumerate(clips):
        if int(clip.samplerate) != samplerate:
            raise RuntimeError(f"samplerate mismatch: {int(clip.samplerate)} != {samplerate}")
        arrays.append(np.asarray(clip.audio_data, dtype=np.float32))
        if silence is not None and i < len(clips) - 1:
            arrays.append(silence)
    merged = np.concatenate(arrays, axis=0).astype(np.float32, copy=False) if len(arrays) > 1 else arrays[0]
    buf = io.BytesIO()
    cast("Any", sf).write(buf, merged, samplerate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _should_retry_text(exc: Exception, text: str) -> bool:
    return (
        isinstance(exc, AssertionError)
        and "length mismatch" in str(exc)
        and _JAPANESE_CHAR_RE.search(text) is not None
    )


def _normalize_retry_text(text: str) -> str:
    t = text.strip().replace("......", "…").replace("...", "…")
    t = t.translate(str.maketrans({"?": "？", "!": "！"}))
    for src, dst in (("洒れる", "こぼれる"), ("洒れた", "こぼれた"), ("洒れて", "こぼれて"), ("洒れ", "こぼれ")):
        t = t.replace(src, dst)
    return t


def _should_retry_chunking(exc: Exception, text: str) -> bool:
    msg = str(exc)
    return (
        isinstance(exc, RuntimeError)
        and "expanded size of the tensor" in msg
        and "existing size" in msg
        and len(text.strip()) > _GSV_LITE_SEGMENT_MAX_CHARS
    )


def _split_long_text(text: str) -> list[str]:
    normalized = "".join(text.split())
    if len(normalized) <= _GSV_LITE_SEGMENT_MAX_CHARS:
        return [normalized]

    primary = [p for p in re.split(r"(?<=[。！？!?…])", normalized) if p]
    parts = primary if len(primary) > 1 else [p for p in re.split(r"(?<=[、，,])", normalized) if p]
    if not parts:
        parts = [normalized]

    segments: list[str] = []
    current = ""

    def flush(val: str) -> None:
        s = val.strip()
        if s:
            segments.append(s)

    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        if len(chunk) > _GSV_LITE_SEGMENT_MAX_CHARS:
            if current:
                flush(current)
                current = ""
            remaining = chunk
            while len(remaining) > _GSV_LITE_SEGMENT_MAX_CHARS:
                split_at = max(
                    remaining.rfind("。", 0, _GSV_LITE_SEGMENT_MAX_CHARS),
                    remaining.rfind("！", 0, _GSV_LITE_SEGMENT_MAX_CHARS),
                    remaining.rfind("？", 0, _GSV_LITE_SEGMENT_MAX_CHARS),
                    remaining.rfind("、", 0, _GSV_LITE_SEGMENT_MAX_CHARS),
                    remaining.rfind("，", 0, _GSV_LITE_SEGMENT_MAX_CHARS),
                    remaining.rfind(",", 0, _GSV_LITE_SEGMENT_MAX_CHARS),
                )
                split_at = (split_at + 1) if split_at > 0 else _GSV_LITE_SEGMENT_MAX_CHARS
                flush(remaining[:split_at])
                remaining = remaining[split_at:].strip()
            current = remaining
            continue
        candidate = f"{current}{chunk}"
        if current and len(candidate) > _GSV_LITE_SEGMENT_MAX_CHARS:
            flush(current)
            current = chunk
        else:
            current = candidate

    if current:
        flush(current)
    return segments or [normalized]


async def _infer_clip(
    model: Any,
    *,
    text: str,
    speaker_audio_path: Path,
    ref_audio: Path,
    prompt_text: str,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    noise_scale: float,
    speed: float,
) -> Any:
    return await cast("Any", model).infer_async(
        spk_audio_path=str(speaker_audio_path),
        prompt_audio_path=str(ref_audio),
        prompt_audio_text=prompt_text,
        text=text,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        noise_scale=noise_scale,
        speed=speed,
    )


async def _synthesize_async(
    *,
    text: str,
    ref_audio: Path,
    ref_text: str,
    speaker_audio: Path | None,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    noise_scale: float,
    speed: float,
) -> bytes:
    if _gsv_lite_engine is None:
        raise RuntimeError("模型尚未加载，请先选择角色并点击「加载模型」")

    prompt_text = ref_text.strip()
    speaker_audio_path = speaker_audio or ref_audio
    candidate_text = text

    kwargs = dict(
        speaker_audio_path=speaker_audio_path,
        ref_audio=ref_audio,
        prompt_text=prompt_text,
        top_k=top_k, top_p=top_p, temperature=temperature,
        repetition_penalty=repetition_penalty, noise_scale=noise_scale, speed=speed,
    )

    try:
        clip = await _infer_clip(_gsv_lite_engine, text=candidate_text, **kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        last_exc = exc

        if _should_retry_text(exc, candidate_text):
            normalized = _normalize_retry_text(candidate_text)
            if normalized != candidate_text:
                print(f"WARNING: gsv-lite: retry with normalized text: {normalized!r}", flush=True)
                candidate_text = normalized
                try:
                    clip = await _infer_clip(_gsv_lite_engine, text=candidate_text, **kwargs)  # type: ignore[arg-type]
                    return _wav_bytes_from_clips([clip])
                except Exception as retry_exc:
                    last_exc = retry_exc

        if _should_retry_chunking(last_exc, candidate_text):
            chunks = _split_long_text(candidate_text)
            if len(chunks) > 1:
                print(f"WARNING: gsv-lite: splitting long text into {len(chunks)} chunks", flush=True)
                clips = [
                    await _infer_clip(_gsv_lite_engine, text=chunk, **kwargs)  # type: ignore[arg-type]
                    for chunk in chunks
                ]
                return _wav_bytes_from_clips(clips)

        raise last_exc from None

    return _wav_bytes_from_clips([clip])


def synthesize_once(
    *,
    text: str,
    ref_audio: Path,
    ref_text: str,
    speaker_audio: Path | None = None,
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.35,
    noise_scale: float = 0.5,
    speed: float = 1.0,
) -> Path:
    paths = _load_runtime_paths()
    started = time.perf_counter()

    loop = asyncio.new_event_loop()
    try:
        wav_bytes = loop.run_until_complete(
            _synthesize_async(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                speaker_audio=speaker_audio,
                top_k=top_k, top_p=top_p, temperature=temperature,
                repetition_penalty=repetition_penalty, noise_scale=noise_scale, speed=speed,
            )
        )
    finally:
        loop.close()

    output_dir = paths.cache_root / "gsv-lite"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    fragment = re.sub(r"_+", "_", re.sub(r"[^\w]", "_", text[:24].strip())).strip("_") or "audio"
    output_path = output_dir / f"{fragment}_{timestamp}.wav"
    output_path.write_bytes(wav_bytes)

    elapsed = time.perf_counter() - started
    print(
        f"INFO: gsv-lite: synthesize_once done: "
        f"text_len={len(text)}, audio_bytes={len(wav_bytes)}, elapsed={elapsed:.2f}s",
        flush=True,
    )
    return output_path
