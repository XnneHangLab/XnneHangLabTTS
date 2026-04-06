from __future__ import annotations

import io
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf


def _get_logic():
    return import_module("lab.api.logic.genie_tts")


def _wav_bytes_to_audio(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return sr, data


def _build_genie_tts_tab(gr):
    def list_characters() -> list[str]:
        try:
            return _get_logic().list_genie_tts_characters()
        except Exception:
            return []

    def refresh_character_list():
        choices = list_characters()
        value = choices[0] if choices else None
        return gr.Dropdown(choices=choices, value=value)

    def refresh_status() -> str:
        try:
            status = _get_logic().get_genie_tts_status()
            loaded_char = status.get("loaded_character")
            if status.get("loaded"):
                return f"✅ 已加载: {loaded_char}"
            return "⚠️ 未加载"
        except Exception as exc:
            return f"❌ {exc}"

    def load_model(character_name: str | None):
        if not character_name:
            yield "❌ 请先选择角色模型"
            return
        yield f"⏳ 正在加载 {character_name}，请稍候…"
        try:
            logic = _get_logic()
            logic.load_genie_tts_model_by_name(character_name)
            status = logic.get_genie_tts_status()
            yield f"✅ 已加载: {status.get('loaded_character')}" if status.get("loaded") else "❌ 加载失败（状态异常）"
        except Exception as exc:
            yield f"❌ 加载失败: {exc}"

    async def synthesize(
        text: str,
        ref_audio_path: str | None,
        ref_text: str | None,
    ) -> tuple[int, np.ndarray]:
        text = (text or "").strip()
        if not text:
            raise gr.Error("合成文本不能为空")

        logic = _get_logic()
        if not logic.get_genie_tts_status().get("loaded"):
            raise gr.Error("模型尚未加载，请先选择角色并点击「加载模型」")

        wav_bytes = await logic.synthesize_once(
            text=text,
            ref_audio=Path(ref_audio_path) if ref_audio_path else None,
            ref_text=(ref_text or "").strip() or None,
        )
        return _wav_bytes_to_audio(wav_bytes)

    initial_choices = list_characters()
    initial_value = initial_choices[0] if initial_choices else None

    with gr.Tab("Genie-TTS"):
        with gr.Row():
            status_box = gr.Textbox(
                label="模型状态",
                value="⚠️ 未加载",
                interactive=False,
                scale=4,
            )
            with gr.Column(scale=1, min_width=120):
                load_btn = gr.Button("加载模型")
                refresh_status_btn = gr.Button("刷新状态")

        with gr.Row():
            character_dropdown = gr.Dropdown(
                label="角色模型",
                choices=initial_choices,
                value=initial_value,
                scale=4,
            )
            refresh_chars_btn = gr.Button("刷新列表", scale=1, min_width=100)

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="合成文本",
                    placeholder="请输入要合成的文字…",
                    lines=4,
                )
                ref_audio_input = gr.Audio(
                    label="参考音频",
                    type="filepath",
                )
                ref_text_input = gr.Textbox(
                    label="参考文本（与参考音频内容一致）",
                    lines=2,
                )
                synth_btn = gr.Button("合成", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="合成结果", interactive=False)

        load_btn.click(fn=load_model, inputs=[character_dropdown], outputs=status_box)
        refresh_status_btn.click(fn=refresh_status, outputs=status_box)
        refresh_chars_btn.click(fn=refresh_character_list, outputs=character_dropdown)
        synth_btn.click(
            fn=synthesize,
            inputs=[text_input, ref_audio_input, ref_text_input],
            outputs=audio_output,
        )

        return status_box


def _build_demo():
    import gradio as gr

    with gr.Blocks(title="XnneHangLab TTS") as demo:
        gr.Markdown("# XnneHangLab TTS 语音合成")

        _build_genie_tts_tab(gr)

        with gr.Tab("GSV-Lite"):
            gr.Markdown("## GSV-Lite\n\nGSV-Lite 功能开发中，敬请期待。")

        with gr.Tab("Faster-Qwen-TTS"):
            gr.Markdown("## Faster-Qwen-TTS\n\nFaster-Qwen-TTS 功能开发中，敬请期待。")

    return demo


def launch(*, host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    import logging
    import os
    import sys
    import threading
    import time

    # Route all Python logging (uvicorn, Gradio internals) to stdout so the
    # Tauri log pipe captures them alongside print() output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    # Gradio 5 does a self-check GET to localhost after uvicorn starts.
    # Corporate proxies / VPNs intercept even loopback traffic and return
    # 502, which causes Gradio to raise an Exception and abort.
    # Ensure localhost and 127.0.0.1 are always in NO_PROXY / no_proxy.
    for _key in ("NO_PROXY", "no_proxy"):
        _existing = {e.strip() for e in os.environ.get(_key, "").split(",") if e.strip()}
        _existing.update({"localhost", "127.0.0.1"})
        os.environ[_key] = ",".join(sorted(_existing))

    demo = _build_demo()

    # Gradio 5 bug: blocks.py calls httpx.get() for a version-check with no
    # surrounding try/except, so a proxy-related ReadError propagates uncaught
    # out of launch().  Suppress httpx.HTTPError from this thread only; every
    # other exception class still goes through the original excepthook.
    _orig_excepthook = threading.excepthook

    def _excepthook(args: threading.ExceptHookArgs) -> None:
        exc = args.exc_value
        try:
            import httpx
            if isinstance(exc, httpx.HTTPError):
                return
        except ImportError:
            pass
        # Gradio 5 raises a plain Exception when its post-launch self-check
        # GET to /gradio_api/startup-events is intercepted by a proxy (502).
        # The server is already running at this point; log a warning and carry on.
        if isinstance(exc, Exception) and "startup-events" in str(exc):
            print(f"WARNING: Gradio startup self-check failed (proxy/network issue): {exc}", flush=True)
            print("WARNING: WebUI may still be accessible at the URL shown above.", flush=True)
            return
        _orig_excepthook(args)

    threading.excepthook = _excepthook
    try:
        threading.Thread(
            target=demo.launch,
            kwargs={"server_name": host, "server_port": port, "share": share},
            daemon=False,
        ).start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        threading.excepthook = _orig_excepthook
