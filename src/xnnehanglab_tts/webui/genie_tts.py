from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from xnnehanglab_tts.webui import genie_runtime


def _wav_bytes_to_audio(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return sample_rate, np.asarray(data)


def _build_genie_tts_tab(gr):
    def list_characters() -> list[str]:
        return genie_runtime.list_available_models()

    def refresh_character_list():
        choices = list_characters()
        value = choices[0] if choices else None
        return gr.Dropdown(choices=choices, value=value)

    def refresh_status() -> str:
        try:
            status = genie_runtime.get_genie_tts_status()
            loaded_char = status.get("loaded_character")
            if status.get("loaded"):
                return f"已加载: {loaded_char}"
            return "未加载"
        except Exception as exc:
            print(f"ERROR: get_genie_tts_status failed: {exc}", flush=True)
            return f"加载状态失败: {exc}"

    def load_model(character_name: str | None):
        if not character_name:
            yield "请先选择角色模型"
            return
        yield f"正在加载 {character_name}，请稍候…"
        try:
            genie_runtime.load_genie_tts_model_by_name(character_name)
            status = genie_runtime.get_genie_tts_status()
            if status.get("loaded"):
                yield f"已加载: {status.get('loaded_character')}"
                return
            yield "加载失败：状态异常"
        except Exception as exc:
            print(f"ERROR: load model failed: {exc}", flush=True)
            yield f"加载失败: {exc}"

    async def synthesize(
        text: str,
        ref_audio_path: str | None,
        ref_text: str | None,
    ) -> tuple[int, np.ndarray]:
        text = (text or "").strip()
        if not text:
            raise gr.Error("合成文本不能为空")

        try:
            wav_bytes = await genie_runtime.synthesize_once(
                text=text,
                ref_audio=None if not ref_audio_path else Path(ref_audio_path),
                ref_text=ref_text,
            )
            return _wav_bytes_to_audio(wav_bytes)
        except gr.Error:
            raise
        except Exception as exc:
            print(f"ERROR: synthesize failed: {exc}", flush=True)
            raise gr.Error(str(exc)) from exc

    initial_choices = list_characters()
    initial_value = initial_choices[0] if initial_choices else None

    with gr.Tab("Genie-TTS"):
        with gr.Row():
            status_box = gr.Textbox(
                label="模型状态",
                value="未加载",
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
    import threading
    import time

    workspace = os.environ.get("XH_VOICE_WORKSPACE_ROOT", "")
    if workspace and Path(workspace).is_dir():
        print(f"INFO: workspace root set to {workspace}", flush=True)
    else:
        print(
            f"WARNING: XH_VOICE_WORKSPACE_ROOT not set or invalid ({workspace!r})",
            flush=True,
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    for env_key in ("NO_PROXY", "no_proxy"):
        existing = {value.strip() for value in os.environ.get(env_key, "").split(",") if value.strip()}
        existing.update({"localhost", "127.0.0.1"})
        os.environ[env_key] = ",".join(sorted(existing))

    demo = _build_demo()
    original_excepthook = threading.excepthook

    def _excepthook(args: threading.ExceptHookArgs) -> None:
        exc = args.exc_value
        try:
            import httpx

            if isinstance(exc, httpx.HTTPError):
                return
        except ImportError:
            pass
        if isinstance(exc, Exception) and "startup-events" in str(exc):
            print(f"WARNING: Gradio startup self-check failed: {exc}", flush=True)
            print("WARNING: WebUI may still be accessible at the URL shown above.", flush=True)
            return
        original_excepthook(args)

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
        threading.excepthook = original_excepthook
