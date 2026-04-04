<p align="center">
  <a href="https://xnnehang.top/">
    <img src="./assets/imgs/logo.svg" alt="魔女の实验室" width="270" height="180" />
  </a>
</p>

# XnneHangLabTTS

XnneHangLabTTS is a lightweight showcase repository derived from [XnneHangLab](https://github.com/XnneHangLab/XnneHangLab).

It focuses on one thing: providing a clean, fast demo experience for multiple TTS backends without bringing in the full complexity of the main project.

## What this repository is for

This repository is intended to showcase and compare several TTS engines in a unified interface:

- GSV-Lite
- Genie-TTS
- faster-qwen-tts

The goal is to make these engines easier to try, compare, and demonstrate.

## What we want to build

This repository will be organized around three parts:

1. Launcher
   - environment check
   - startup mode selection
   - resource and dependency preparation

2. Gradio WebUI
   - select backend
   - select character / style
   - enter text
   - generate and preview audio

3. Packaging and deployment
   - uv
   - Docker
   - Conda environment

## Target

XnneHangLabTTS is for:

- quick local demos
- backend comparison
- lightweight sharing
- easier onboarding for new users

## Planned support

We plan to support three ways to run the project:

- `uv`
- `docker`
- `conda`

Recommended order:

1. uv for local development and fastest setup
2. docker for reproducible deployment
3. conda as an alternative environment option

## Current status

This repository is being prepared.

The first milestone is:

- a minimal launcher
- a Gradio-based inference UI
- unified backend adapters for GSV-Lite / Genie-TTS / faster-qwen-tts

## Relationship to XnneHangLab

If you want the full agent system, conversation pipeline, and integrated project experience, use the main repository:

- [XnneHangLab](https://github.com/XnneHangLab/XnneHangLab)

If you want a smaller repository focused on TTS demo and distribution, use this repository.
