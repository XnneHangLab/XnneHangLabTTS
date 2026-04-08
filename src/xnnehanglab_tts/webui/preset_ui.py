from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


# ── Storage helpers ──────────────────────────────────────────────────────────

def _presets_dir(tab_name: str) -> Path:
    workspace = os.environ.get("XH_VOICE_WORKSPACE_ROOT", "")
    if workspace and Path(workspace).is_dir():
        base = Path(workspace)
    else:
        base = Path.home() / ".xnnehanglab"
    return base / "presets" / tab_name


def _load_json(tab_name: str) -> dict[str, Any]:
    path = _presets_dir(tab_name) / "presets.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(tab_name: str, data: dict[str, Any]) -> None:
    d = _presets_dir(tab_name)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "presets.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_preset_names(tab_name: str) -> list[str]:
    return sorted(_load_json(tab_name).keys())


def _resolve_audio(tab_name: str, entry: dict[str, Any], field: str) -> str | None:
    fname = entry.get(field)
    if not fname:
        return None
    p = _presets_dir(tab_name) / fname
    return str(p) if p.is_file() else None


def _copy_audio(tab_name: str, src: str | None, safe_name: str, field: str) -> str | None:
    if not src or not Path(src).is_file():
        return None
    suffix = Path(src).suffix or ".wav"
    dest_name = f"{safe_name}_{field}{suffix}"
    dest = _presets_dir(tab_name) / dest_name
    _presets_dir(tab_name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest_name


def _save_preset(
    tab_name: str,
    name: str,
    ref_audio: str | None,
    ref_text: str | None,
    speaker_audio: str | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    safe_name = name.strip()[:60]
    if not safe_name:
        return
    data = _load_json(tab_name)
    data[safe_name] = {
        "ref_text": (ref_text or "").strip(),
        "ref_audio": _copy_audio(tab_name, ref_audio, safe_name, "ref"),
        "speaker_audio": _copy_audio(tab_name, speaker_audio, safe_name, "spk"),
        "params": params or {},
    }
    _save_json(tab_name, data)


def _delete_preset(tab_name: str, name: str) -> None:
    data = _load_json(tab_name)
    entry = data.pop(name, None)
    if entry:
        for field in ("ref_audio", "speaker_audio"):
            fname = entry.get(field)
            if fname:
                try:
                    (_presets_dir(tab_name) / fname).unlink(missing_ok=True)
                except Exception:
                    pass
    _save_json(tab_name, data)


# ── Gradio UI ────────────────────────────────────────────────────────────────

class PresetControls:
    """
    Holds the preset UI components.  Call .wire() after defining all the
    audio / text / slider components the preset should save and restore.

    Usage::

        preset = build_preset_section(gr, "gsv-lite")

        # ... define ref_audio_input, ref_text_input, speaker_audio_input, sliders ...

        preset.wire(
            gr, ref_audio_input, ref_text_input,
            speaker_audio_comp=speaker_audio_input,
            param_comps=[("top_k", top_k_slider), ...],
        )
    """

    def __init__(self, tab_name: str, dropdown, name_input, save_btn, delete_btn):
        self._tab_name = tab_name
        self._dropdown = dropdown
        self._name_input = name_input
        self._save_btn = save_btn
        self._delete_btn = delete_btn

    def wire(
        self,
        gr,
        ref_audio_comp,
        ref_text_comp,
        speaker_audio_comp=None,
        param_comps: list[tuple[str, Any]] | None = None,
    ) -> None:
        tab_name = self._tab_name
        has_speaker = speaker_audio_comp is not None
        param_comps = param_comps or []

        load_outputs = [ref_audio_comp, ref_text_comp]
        if has_speaker:
            load_outputs.append(speaker_audio_comp)
        for _key, comp in param_comps:
            load_outputs.append(comp)

        def on_select(preset_name: str | None):
            if not preset_name:
                return [gr.update()] * len(load_outputs)
            entry = _load_json(tab_name).get(preset_name, {})
            saved_params = entry.get("params", {})
            updates = [
                gr.update(value=_resolve_audio(tab_name, entry, "ref_audio")),
                gr.update(value=entry.get("ref_text") or None),
            ]
            if has_speaker:
                updates.append(gr.update(value=_resolve_audio(tab_name, entry, "speaker_audio")))
            for key, _comp in param_comps:
                val = saved_params.get(key)
                updates.append(gr.update(value=val) if val is not None else gr.update())
            return updates

        self._dropdown.change(fn=on_select, inputs=[self._dropdown], outputs=load_outputs)

        save_inputs = [self._name_input, ref_audio_comp, ref_text_comp]
        if has_speaker:
            save_inputs.append(speaker_audio_comp)
        for _key, comp in param_comps:
            save_inputs.append(comp)

        def on_save(name: str, ref_audio, ref_text, *rest):
            name = (name or "").strip()
            if not name:
                return gr.update()
            idx = 0
            spk = None
            if has_speaker:
                spk = rest[idx]; idx += 1
            params = {key: rest[idx + i] for i, (key, _) in enumerate(param_comps)}
            _save_preset(tab_name, name, ref_audio, ref_text, spk, params)
            return gr.update(choices=list_preset_names(tab_name), value=name)

        self._save_btn.click(fn=on_save, inputs=save_inputs, outputs=[self._dropdown])

        def on_delete(preset_name: str | None):
            if preset_name:
                _delete_preset(tab_name, preset_name)
            return gr.update(choices=list_preset_names(tab_name), value=None)

        self._delete_btn.click(fn=on_delete, inputs=[self._dropdown], outputs=[self._dropdown])


def build_preset_section(gr, tab_name: str) -> PresetControls:
    """
    Render the preset accordion at the current position in the Gradio layout
    and return a PresetControls object.  Call .wire() afterwards once all
    audio / text / slider components have been defined.
    """
    with gr.Accordion("预设", open=False):
        with gr.Row():
            dropdown = gr.Dropdown(
                label="选择预设",
                choices=list_preset_names(tab_name),
                value=None,
                scale=4,
            )
            name_input = gr.Textbox(
                label="新预设名称",
                placeholder="输入名称后点保存…",
                scale=3,
            )
            save_btn = gr.Button("保存", scale=1, min_width=80)
            delete_btn = gr.Button("删除", scale=1, min_width=80, variant="stop")

    return PresetControls(tab_name, dropdown, name_input, save_btn, delete_btn)
