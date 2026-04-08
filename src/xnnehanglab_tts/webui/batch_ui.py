from __future__ import annotations

import sys
import traceback
from typing import Callable

MAX_BATCH = 20


def build_batch_section(gr, synthesize_fn: Callable, extra_inputs: list) -> None:
    """
    Render a batch synthesis UI into the current Gradio block.

    synthesize_fn(text: str, *extra_args) -> Path
        Called once per line; should raise on error (caught internally).
    extra_inputs: list[gr.Component]
        Passed as extra_args after `text`.
    """
    gr.Markdown("每行文字独立生成一句，可对每句单独试听或重新生成。")
    batch_text = gr.Textbox(
        label="批量文本（每行一句）",
        placeholder="你好，世界。\n今天天气很好。\n一行一句话，独立生成。",
        lines=5,
    )
    batch_btn = gr.Button("批量生成", variant="primary")
    batch_status = gr.Textbox(
        label="进度",
        interactive=False,
        visible=False,
        max_lines=1,
    )

    slots: list[tuple] = []  # (group, label_md, audio_out, regen_btn)
    for i in range(MAX_BATCH):
        with gr.Group(visible=False) as grp:
            with gr.Row():
                label_md = gr.Markdown(f"**{i + 1}.** —")
                regen_btn = gr.Button("↺ 重新生成", size="sm", min_width=100)
            audio_out = gr.Audio(interactive=False, show_label=False)
        slots.append((grp, label_md, audio_out, regen_btn))

    all_groups = [s[0] for s in slots]
    all_labels = [s[1] for s in slots]
    all_audios = [s[2] for s in slots]

    # outputs: [status] + 20 groups + 20 labels + 20 audios  =  61 items
    def run_batch(text_val: str, *extra_args):
        lines = [ln.strip() for ln in (text_val or "").splitlines() if ln.strip()]
        lines = lines[:MAX_BATCH]

        if not lines:
            yield (
                [gr.update(value="请输入至少一句文字", visible=True)]
                + [gr.update(visible=False)] * MAX_BATCH
                + [gr.update()] * MAX_BATCH
                + [gr.update(value=None)] * MAX_BATCH
            )
            return

        n = len(lines)
        cur_groups = [gr.update(visible=False)] * MAX_BATCH
        cur_labels = [gr.update()] * MAX_BATCH
        cur_audios = [gr.update(value=None)] * MAX_BATCH

        yield (
            [gr.update(value=f"生成中 0/{n}…", visible=True)]
            + cur_groups + cur_labels + cur_audios
        )

        for i, line in enumerate(lines):
            preview = line[:40] + ("…" if len(line) > 40 else "")
            try:
                result = synthesize_fn(line, *extra_args)
                audio_val = str(result)
            except Exception as exc:
                traceback.print_exc(file=sys.stdout)
                print(f"ERROR: batch[{i}]: {exc}", flush=True)
                audio_val = None

            cur_groups = list(cur_groups)
            cur_labels = list(cur_labels)
            cur_audios = list(cur_audios)
            cur_groups[i] = gr.update(visible=True)
            cur_labels[i] = gr.update(value=f"**{i + 1}.** {preview}")
            cur_audios[i] = gr.update(value=audio_val)

            yield (
                [gr.update(value=f"生成中 {i + 1}/{n}…", visible=(i + 1 < n))]
                + cur_groups + cur_labels + cur_audios
            )

    batch_btn.click(
        fn=run_batch,
        inputs=[batch_text] + list(extra_inputs),
        outputs=[batch_status] + all_groups + all_labels + all_audios,
    )

    for i, (_grp, _label, audio, btn) in enumerate(slots):
        def make_regen(idx: int):
            def regen(text_val: str, *extra_args):
                lines = [ln.strip() for ln in (text_val or "").splitlines() if ln.strip()]
                if idx >= len(lines):
                    raise gr.Error(f"第 {idx + 1} 句不存在，请重新执行批量生成")
                result = synthesize_fn(lines[idx], *extra_args)
                return str(result)
            return regen

        btn.click(
            fn=make_regen(i),
            inputs=[batch_text] + list(extra_inputs),
            outputs=[audio],
        )
