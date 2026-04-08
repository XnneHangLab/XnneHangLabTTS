from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from xnnehanglab_tts.webui import genie_runtime
from xnnehanglab_tts.webui.batch_ui import build_batch_section
from xnnehanglab_tts.webui.gsv_lite_tts import build_gsv_lite_tab
from xnnehanglab_tts.webui.qwen_tts import build_qwen_tts_tab
from xnnehanglab_tts.webui.preset_ui import build_preset_section


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

    def load_model(character_name: str | None, onnx_threads: int):
        if not character_name:
            yield "请先选择角色模型"
            return
        yield f"正在加载 {character_name}，请稍候…"
        try:
            genie_runtime.load_genie_tts_model_by_name(character_name, onnx_intra_threads=int(onnx_threads))
            status = genie_runtime.get_genie_tts_status()
            if status.get("loaded"):
                yield f"已加载: {status.get('loaded_character')}"
                return
            yield "加载失败：状态异常"
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)
            print(f"ERROR: load model failed: {exc}", flush=True)
            yield f"加载失败: {exc}"

    def synthesize(text: str, ref_audio_path: str | None, ref_text: str | None) -> str:
        text = (text or "").strip()
        if not text:
            raise gr.Error("合成文本不能为空")
        try:
            output_path = genie_runtime.synthesize_once(
                text=text,
                ref_audio=None if not ref_audio_path else Path(ref_audio_path),
                ref_text=ref_text,
            )
            return str(output_path)
        except gr.Error:
            raise
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)
            print(f"ERROR: synthesize failed: {exc}", flush=True)
            raise gr.Error(str(exc)) from exc

    def batch_synthesize(text: str, ref_audio_path: str | None, ref_text: str | None) -> Path:
        text = (text or "").strip()
        if not text:
            raise ValueError("合成文本不能为空")
        return genie_runtime.synthesize_once(
            text=text,
            ref_audio=None if not ref_audio_path else Path(ref_audio_path),
            ref_text=ref_text,
        )

    initial_choices = list_characters()
    initial_value = initial_choices[0] if initial_choices else None

    with gr.Tab("Genie-TTS"):
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
            onnx_threads_slider = gr.Slider(
                label="ONNX 推理线程数", minimum=1, maximum=16, value=4, step=1,
                info="限制 T2S 解码器使用的 CPU 线程数，可防止 Windows 热降频。加载模型时生效。",
            )

        # ── 共享参考音频 ───────────────────────────────────────────────
        with gr.Row():
            ref_audio_input = gr.Audio(label="参考音频", type="filepath", scale=1)
            ref_text_input = gr.Textbox(
                label="参考文本（与参考音频内容一致）", lines=3, scale=1,
            )

        build_preset_section(gr, "genie-tts", ref_audio_input, ref_text_input)

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
        load_btn.click(
            fn=load_model,
            inputs=[character_dropdown, onnx_threads_slider],
            outputs=status_box,
        )
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
        build_gsv_lite_tab(gr)
        build_qwen_tts_tab(gr)

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
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stdout)
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
