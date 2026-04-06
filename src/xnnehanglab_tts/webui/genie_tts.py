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


def _build_demo():
    import gradio as gr

    def load_model() -> str:
        try:
            logic = _get_logic()
            logic.load_genie_tts_model()
            status = logic.get_genie_tts_status()
            return "✅ 模型已加载" if status.get("loaded") else "❌ 模型加载失败"
        except Exception as exc:
            return f"❌ {exc}"

    def refresh_status() -> str:
        try:
            status = _get_logic().get_genie_tts_status()
            return "✅ 已加载" if status.get("loaded") else "⚠️ 未加载"
        except Exception as exc:
            return f"❌ {exc}"

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
            raise gr.Error("模型尚未加载，请先点击「加载模型」")

        wav_bytes = await logic.synthesize_once(
            text=text,
            ref_audio=Path(ref_audio_path) if ref_audio_path else None,
            ref_text=(ref_text or "").strip() or None,
        )
        return _wav_bytes_to_audio(wav_bytes)

    with gr.Blocks(title="Genie-TTS") as demo:
        gr.Markdown("# Genie-TTS 语音合成")

        with gr.Row():
            status_box = gr.Textbox(
                label="模型状态",
                value=refresh_status,
                interactive=False,
                scale=4,
            )
            with gr.Column(scale=1, min_width=120):
                load_btn = gr.Button("加载模型")
                refresh_btn = gr.Button("刷新状态")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="合成文本",
                    placeholder="请输入要合成的文字…",
                    lines=4,
                )
                ref_audio_input = gr.Audio(
                    label="参考音频（可选）",
                    type="filepath",
                )
                ref_text_input = gr.Textbox(
                    label="参考文本（可选，与参考音频内容一致）",
                    lines=2,
                )
                synth_btn = gr.Button("合成", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="合成结果", interactive=False)

        load_btn.click(fn=load_model, outputs=status_box)
        refresh_btn.click(fn=refresh_status, outputs=status_box)
        synth_btn.click(
            fn=synthesize,
            inputs=[text_input, ref_audio_input, ref_text_input],
            outputs=audio_output,
        )

    return demo


def launch(*, host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    import sys
    import time
    import traceback

    try:
        _build_demo().launch(server_name=host, server_port=port, share=share)
    except Exception as exc:
        try:
            import httpx
            is_network_exc = isinstance(exc, httpx.HTTPError)
        except ImportError:
            is_network_exc = False

        if not is_network_exc:
            raise

        # Print the real traceback so it appears in the launcher console.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        display_host = "127.0.0.1" if host == "0.0.0.0" else host
        print(
            f"[webui] 上方网络错误已忽略（代理拦截）；服务器持续运行中: http://{display_host}:{port}",
            flush=True,
        )
        while True:
            time.sleep(3600)
