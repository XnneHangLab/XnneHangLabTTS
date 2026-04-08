from __future__ import annotations

import sys
import traceback
from pathlib import Path

from xnnehanglab_tts.webui import qwen_tts_runtime


def build_qwen_tts_tab(gr) -> None:
    def refresh_status() -> str:
        status = qwen_tts_runtime.get_qwen_tts_status()
        if status["loaded"]:
            return f"已加载: Qwen3-TTS-{status['loaded_model']}"
        return "未加载"

    def load_model(model_name: str):
        yield f"正在加载 Qwen3-TTS-{model_name}，请稍候…"
        try:
            qwen_tts_runtime.load_qwen_tts_model(model_name)
            yield f"已加载: Qwen3-TTS-{model_name}"
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)
            print(f"ERROR: qwen-tts load model failed: {exc}", flush=True)
            yield f"加载失败: {exc}"

    def synthesize(
        text: str,
        ref_audio_path: str | None,
        ref_text: str | None,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise gr.Error("合成文本不能为空")

        try:
            output_path = qwen_tts_runtime.synthesize_once(
                text=text,
                ref_audio=Path(ref_audio_path) if ref_audio_path else None,
                ref_text=ref_text,
            )
            return str(output_path)
        except gr.Error:
            raise
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)
            print(f"ERROR: qwen-tts synthesize failed: {exc}", flush=True)
            raise gr.Error(str(exc)) from exc

    with gr.Tab("Faster-Qwen-TTS"):
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
            model_dropdown = gr.Dropdown(
                label="模型版本",
                choices=["0.6b", "1.7b"],
                value="0.6b",
            )

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
                    label="参考文本（与参考音频内容一致，可选）",
                    lines=2,
                )
                synth_btn = gr.Button("合成", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="合成结果", interactive=False)

        load_btn.click(fn=load_model, inputs=[model_dropdown], outputs=status_box)
        refresh_status_btn.click(fn=refresh_status, outputs=status_box)
        synth_btn.click(
            fn=synthesize,
            inputs=[text_input, ref_audio_input, ref_text_input],
            outputs=audio_output,
        )
