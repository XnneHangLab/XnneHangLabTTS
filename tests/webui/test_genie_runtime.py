import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from xnnehanglab_tts.runtime.paths import RuntimePaths


def _build_paths(tmp_path: Path) -> RuntimePaths:
    workspace_root = tmp_path / "workspace"
    models_root = workspace_root / "models"
    cache_root = models_root / "cache"
    logs_root = workspace_root / "logs"
    return RuntimePaths(
        workspace_root=workspace_root,
        models_root=models_root,
        genie_base_root=models_root / "GenieData",
        genie_data_dir=models_root / "GenieData",
        genie_characters_root=models_root / "genie" / "characters",
        genie_tts_root=models_root / "genie-tts",
        genie_tts_luming_v2_pro_plus_root=models_root / "genie-tts" / "luming-v2-pro-plus",
        gsv_lite_root=models_root / "GSVLiteData",
        qwen_tts_0_6b_root=models_root / "Qwen3-TTS-0.6B",
        qwen_tts_1_7b_root=models_root / "Qwen3-TTS-1.7B",
        cache_root=cache_root,
        logs_root=logs_root,
        download_logs_root=logs_root / "downloads",
    )


def test_load_genie_module_uses_bundled_package_path_and_runtime_env(
    monkeypatch, tmp_path: Path
):
    from xnnehanglab_tts.webui import genie_runtime

    paths = _build_paths(tmp_path)
    repo_root = tmp_path / "repo"
    bundled_src = repo_root / "packages" / "Genie-TTS" / "src"
    bundled_src.mkdir(parents=True)

    imported = []
    stub_module = object()

    monkeypatch.setattr(genie_runtime, "_resolve_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        genie_runtime,
        "import_module",
        lambda name: imported.append(name) or stub_module,
    )

    module = genie_runtime._load_genie_module(paths)

    assert module is stub_module
    assert imported == ["genie_tts"]
    assert sys.path[0] == str(bundled_src)
    assert os.environ["GENIE_DATA_DIR"] == str(paths.genie_base_root)


def test_patch_genie_resource_paths_updates_imported_modules(
    monkeypatch, tmp_path: Path
):
    from xnnehanglab_tts.webui import genie_runtime

    genie_data_dir = (tmp_path / "models" / "GenieData").resolve()
    resources_module = SimpleNamespace(
        GENIE_DATA_DIR="/wrong",
        English_G2P_DIR="/wrong",
        Chinese_G2P_DIR="/wrong",
        HUBERT_MODEL_DIR="/wrong",
        SV_MODEL="/wrong",
        ROBERTA_MODEL_DIR="/wrong",
    )
    gsv_model_file = SimpleNamespace(
        HUBERT_MODEL="/wrong",
        HUBERT_MODEL_WEIGHT_FP16="/wrong",
        ROBERTA_MODEL="/wrong",
        ROBERTA_TOKENIZER="/wrong",
    )
    model_manager_module = SimpleNamespace(
        HUBERT_MODEL_DIR="/wrong",
        SV_MODEL="/wrong",
        ROBERTA_MODEL_DIR="/wrong",
        GSVModelFile=gsv_model_file,
    )

    monkeypatch.setitem(sys.modules, "genie_tts.Core.Resources", resources_module)
    monkeypatch.setitem(sys.modules, "genie_tts.ModelManager", model_manager_module)

    genie_runtime._patch_genie_resource_paths(genie_data_dir)

    assert resources_module.GENIE_DATA_DIR == str(genie_data_dir)
    assert resources_module.SV_MODEL == str(genie_data_dir / "speaker_encoder.onnx")
    assert model_manager_module.ROBERTA_MODEL_DIR == str(genie_data_dir / "RoBERTa")
    assert gsv_model_file.ROBERTA_MODEL == str(genie_data_dir / "RoBERTa" / "RoBERTa.onnx")


def test_load_genie_tts_model_by_name_uses_local_runtime_model_dir(
    monkeypatch, tmp_path: Path
):
    from xnnehanglab_tts.webui import genie_runtime

    paths = _build_paths(tmp_path)
    model_dir = paths.genie_tts_root / "luming-v2-pro-plus"
    model_dir.mkdir(parents=True)

    load_calls = []
    genie_module = SimpleNamespace(
        load_character=lambda **kwargs: load_calls.append(kwargs),
        unload_character=lambda character_name: None,
    )

    monkeypatch.setattr(
        genie_runtime,
        "load_runtime_config",
        lambda config_path=None: (SimpleNamespace(), paths),
    )
    monkeypatch.setattr(genie_runtime, "_load_genie_module", lambda _paths: genie_module)
    monkeypatch.setattr(
        genie_runtime,
        "_STATE",
        genie_runtime.GenieRuntimeState(),
    )

    genie_runtime.load_genie_tts_model_by_name("luming-v2-pro-plus")
    status = genie_runtime.get_genie_tts_status()

    assert load_calls == [
        {
            "character_name": "luming-v2-pro-plus",
            "onnx_model_dir": str(model_dir),
            "language": "auto",
            "use_roberta": False,
        }
    ]
    assert status == {
        "loaded": True,
        "loaded_character": "luming-v2-pro-plus",
    }


def test_synthesize_once_requires_reference_audio_and_text(monkeypatch, tmp_path: Path):
    from xnnehanglab_tts.webui import genie_runtime

    monkeypatch.setattr(
        genie_runtime,
        "_STATE",
        genie_runtime.GenieRuntimeState(loaded_character="luming-v2-pro-plus"),
    )

    with pytest.raises(ValueError, match="参考音频"):
        import asyncio

        asyncio.run(
            genie_runtime.synthesize_once(
                text="你好",
                ref_audio=None,
                ref_text="参考文本",
            )
        )
