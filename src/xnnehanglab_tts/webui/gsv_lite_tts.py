from __future__ import annotations

import sys
import traceback
from pathlib import Path

from xnnehanglab_tts.webui import gsv_lite_runtime
from xnnehanglab_tts.webui.batch_ui import build_batch_section


def build_gsv_lite_tab(gr) -> None:
    def list_characters() -> list[str]:
        return gsv_lite_runtime.list_available_characters()

    def refresh_character_list():
        choices = list_characters()
        value = choices[0] if choices else None
        return gr.Dropdown(choices=choices, value=value)

    def refresh_status() -> str:
        try:
            status = gsv_lite_runtime.get_gsv_lite_status()
            if status.get("loaded"):
                return f"已加载: {status.get('loaded_character')}"
            return "未加载"
        except Exception as exc:
            print(f"ERROR: get_gsv_lite_status failed: {exc}", flush=True)
            return f"加载状态失败: {exc}"

    def load_model(character_name: str | None, use_bert: bool):
        if not character_name:
            yield "请先选择角色模型"
            return
        yield f"正在加载 {character_name}，请稍候…"
        try:
            gsv_lite_runtime.load_gsv_lite_model(character_name, use_bert=use_bert)
            status = gsv_lite_runtime.get_gsv_lite_status()
            if status.get("loaded"):
                yield f"已加载: {status.get('loaded_character')}"
            else:
                yield "加载失败：状态异常"
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)
            print(f"ERROR: load model failed: {exc}", flush=True)
            yield f"加载失败: {exc}"

    def synthesize_dispatch(
        text: str,
        ref_audio_path: str | None,
        ref_text: str | None,
        speaker_audio_path: str | None,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        noise_scale: float,
        speed: float,
        use_streaming: bool,
    ):
        text = (text or "").strip()
        if not text:
            raise gr.Error("合成文本不能为空")
        if not ref_audio_path:
            raise gr.Error("请提供参考音频")
        if not (ref_text or "").strip():
            raise gr.Error("请提供参考文本")

        kw = dict(
            ref_audio=Path(ref_audio_path),
            ref_text=ref_text or "",
            speaker_audio=Path(speaker_audio_path) if speaker_audio_path else None,
            top_k=int(top_k), top_p=float(top_p), temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            noise_scale=float(noise_scale), speed=float(speed),
        )

        try:
            if use_streaming:
                for sr, chunk in gsv_lite_runtime.stream_synthesize(text=text, **kw):
                    yield gr.update(), (sr, chunk)
            else:
                output_path = gsv_lite_runtime.synthesize_once(text=text, **kw)
                yield str(output_path), gr.update(value=None)
        except gr.Error:
            raise
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)
            print(f"ERROR: gsv-lite synthesize failed: {exc}", flush=True)
            raise gr.Error(str(exc)) from exc

    def batch_synthesize(
        text: str,
        ref_audio_path: str | None,
        ref_text: str | None,
        speaker_audio_path: str | None,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        noise_scale: float,
        speed: float,
    ) -> Path:
        text = (text or "").strip()
        if not text:
            raise ValueError("合成文本不能为空")
        if not ref_audio_path:
            raise ValueError("请提供参考音频")
        if not (ref_text or "").strip():
            raise ValueError("请提供参考文本")
        return gsv_lite_runtime.synthesize_once(
            text=text,
            ref_audio=Path(ref_audio_path),
            ref_text=ref_text or "",
            speaker_audio=Path(speaker_audio_path) if speaker_audio_path else None,
            top_k=int(top_k), top_p=float(top_p), temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            noise_scale=float(noise_scale), speed=float(speed),
        )

    initial_choices = list_characters()
    initial_value = initial_choices[0] if initial_choices else None

    with gr.Tab("GSV-Lite"):
        # ── 模型加载 ──────────────────────────────────────────────────
        with gr.Row():
            status_box = gr.Textbox(
                label="模型状态", value="未加载", interactive=False, scale=4,
            )
            with gr.Column(scale=1, min_width=120):
                load_btn = gr.Button("加载模型")
                refresh_status_btn = gr.Button("刷新状态")

        with gr.Row():
            character_dropdown = gr.Dropdown(
                label="角色模型", choices=initial_choices, value=initial_value, scale=4,
            )
            refresh_chars_btn = gr.Button("刷新列表", scale=1, min_width=100)

        with gr.Row():
            use_bert_checkbox = gr.Checkbox(
                label="使用 BERT", value=False,
                info="启用 BERT 文本编码（加载时生效，需要更多显存）",
            )

        # ── 共享参考音频 ───────────────────────────────────────────────
        with gr.Row():
            ref_audio_input = gr.Audio(label="参考音频", type="filepath", scale=1)
            ref_text_input = gr.Textbox(
                label="参考文本（与参考音频内容一致）", lines=3, scale=1,
            )

        with gr.Row():
            speaker_audio_input = gr.Audio(
                label="说话人音频（可选，留空则与参考音频相同）", type="filepath",
            )

        # ── 共享推理参数 ───────────────────────────────────────────────
        with gr.Accordion("推理参数", open=False):
            with gr.Row():
                top_k_slider = gr.Slider(label="Top-K", minimum=1, maximum=50, value=15, step=1)
                top_p_slider = gr.Slider(label="Top-P", minimum=0.1, maximum=1.0, value=1.0, step=0.05)
                temperature_slider = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.05,
                )
            with gr.Row():
                repetition_penalty_slider = gr.Slider(
                    label="Repetition Penalty", minimum=1.0, maximum=2.0, value=1.35, step=0.05,
                )
                noise_scale_slider = gr.Slider(
                    label="Noise Scale", minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                )
                speed_slider = gr.Slider(
                    label="语速", minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                )

        shared_inference_inputs = [
            ref_audio_input, ref_text_input, speaker_audio_input,
            top_k_slider, top_p_slider, temperature_slider,
            repetition_penalty_slider, noise_scale_slider, speed_slider,
        ]

        # ── 单句 / 批处理 sub-tabs ────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("单句合成"):
                with gr.Row():
                    stream_checkbox = gr.Checkbox(
                        label="流式播放",
                        value=False,
                        info="边生成边播放，降低首字延迟（实验性）",
                    )
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
                        audio_stream = gr.Audio(
                            label="合成结果（流式）",
                            streaming=True,
                            autoplay=True,
                            interactive=False,
                            visible=False,
                        )

            with gr.Tab("批处理"):
                build_batch_section(gr, batch_synthesize, shared_inference_inputs)

        # ── 事件绑定 ──────────────────────────────────────────────────
        load_btn.click(
            fn=load_model,
            inputs=[character_dropdown, use_bert_checkbox],
            outputs=status_box,
        )
        refresh_status_btn.click(fn=refresh_status, outputs=status_box)
        refresh_chars_btn.click(fn=refresh_character_list, outputs=character_dropdown)
        stream_checkbox.change(
            fn=lambda v: (gr.update(visible=not v), gr.update(visible=v)),
            inputs=[stream_checkbox],
            outputs=[audio_output, audio_stream],
        )
        synth_btn.click(
            fn=synthesize_dispatch,
            inputs=[text_input] + shared_inference_inputs + [stream_checkbox],
            outputs=[audio_output, audio_stream],
        )
