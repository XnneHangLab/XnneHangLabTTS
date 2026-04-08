from __future__ import annotations

import sys
import traceback
from pathlib import Path

from xnnehanglab_tts.webui import qwen_tts_runtime
from xnnehanglab_tts.webui.batch_ui import build_batch_section
from xnnehanglab_tts.webui.preset_ui import build_preset_section


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

    def synthesize(text: str, ref_audio_path: str | None, ref_text: str | None) -> str:
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

    def batch_synthesize(text: str, ref_audio_path: str | None, ref_text: str | None) -> Path:
        text = (text or "").strip()
        if not text:
            raise ValueError("合成文本不能为空")
        return qwen_tts_runtime.synthesize_once(
            text=text,
            ref_audio=Path(ref_audio_path) if ref_audio_path else None,
            ref_text=ref_text,
        )

    with gr.Tab("Faster-Qwen-TTS"):
        # ── 模型加载 ──────────────────────────────────────────────────
        with gr.Row():
            status_box = gr.Textbox(
                label="模型状态", value="未加载", interactive=False, scale=4,
            )
            load_btn = gr.Button("加载模型", variant="primary", scale=1, min_width=110)
            refresh_status_btn = gr.Button("刷新状态", variant="secondary", scale=1, min_width=110)

        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="模型版本", choices=["0.6b", "1.7b"], value="0.6b",
            )

        # ── 预设（渲染在参考音频上方，事件在后面 wire）─────────────────
        preset = build_preset_section(gr, "qwen-tts")

        # ── 共享参考音频（可选）────────────────────────────────────────
        with gr.Row():
            ref_audio_input = gr.Audio(
                label="参考音频（可选）", type="filepath", scale=1,
            )
            ref_text_input = gr.Textbox(
                label="参考文本（与参考音频内容一致，可选）", lines=3, scale=1,
            )

        preset.wire(gr, ref_audio_input, ref_text_input)

        # ── 单句 / 批处理 sub-tabs ────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("单句合成"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要合成的文字…",
                            lines=4,
                        )
                        synth_btn = gr.Button("合成", variant="primary")
                    with gr.Column():
                        audio_output = gr.Audio(label="合成结果", interactive=False)

            with gr.Tab("批处理"):
                build_batch_section(
                    gr,
                    batch_synthesize,
                    [ref_audio_input, ref_text_input],
                )

        # ── 事件绑定 ──────────────────────────────────────────────────
        load_btn.click(fn=load_model, inputs=[model_dropdown], outputs=status_box)
        refresh_status_btn.click(fn=refresh_status, outputs=status_box)
        synth_btn.click(
            fn=synthesize,
            inputs=[text_input, ref_audio_input, ref_text_input],
            outputs=audio_output,
        )
