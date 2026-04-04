<p align="center">
  <a href="https://xnnehang.top/">
    <img src="./assets/imgs/logo.svg" alt="魔女の实验室" width="270" height="180" />
  </a>
</p>

<h1 align="center">XnneHangLabTTS</h1>

<p align="center">
  来自 <a href="https://github.com/XnneHangLab/XnneHangLab">XnneHangLab</a> 的轻量级 TTS 展示仓库
</p>

<p align="center">
  <img src="https://img.shields.io/badge/语言-中文优先-blue" />
  <img src="https://img.shields.io/badge/WebUI-Gradio-orange" />
  <img src="https://img.shields.io/badge/运行方式-uv%20%7C%20docker%20%7C%20conda-6f42c1" />
  <img src="https://img.shields.io/badge/状态-WIP-ff69b4" />
</p>

---

> [!NOTE]
> XnneHangLabTTS 专注做一件事：
> 用更轻的仓库结构，快速展示和对比多个 TTS 后端。
>
> 当前计划展示的后端：
> - GSV-Lite
> - Genie-TTS
> - faster-qwen-tts

## 项目定位

这个仓库是从 [XnneHangLab](https://github.com/XnneHangLab/XnneHangLab) 中拆出来的轻量展示版本。

它不会一开始就承载完整 Agent、复杂对话链路、Live2D 联动和整套系统集成，而是先把最容易看到成果的部分单独做好：

- 更轻的启动方式
- 更直观的 Gradio 推理界面
- 更统一的 TTS 后端展示体验
- 更容易分发的打包与部署方式

## 我们要做什么

这个仓库会围绕三部分构建：

### 1. 启动管理器

负责：
- 环境检查
- 启动模式选择
- 资源与依赖检查
- 一键拉起 WebUI / 服务

### 2. Gradio WebUI

负责：
- 选择 TTS 后端
- 选择角色 / 风格
- 输入文本
- 生成并试听音频
- 展示不同后端的效果差异

### 3. 打包与部署

支持三种方式：

- `uv`
- `docker`
- `conda`

当前建议顺序：

1. `uv`：本地开发与最快启动
2. `docker`：复现和部署
3. `conda`：作为兼容性方案保留

## 为什么单独做这个仓库

主仓库 [XnneHangLab](https://github.com/XnneHangLab/XnneHangLab) 的目标更完整，也更复杂。

而 XnneHangLabTTS 更适合做这些事情：

- 快速展示 TTS 能力
- 对比不同推理后端
- 给新用户更低门槛的体验入口
- 给传播、演示、录视频、做整合包提供更轻的基础仓库

## 计划支持的后端

- GSV-Lite
- Genie-TTS
- faster-qwen-tts

后续会通过统一的 provider 适配层来组织调用方式，而不是让前端直接耦合到具体实现。

## 第一阶段目标

第一阶段先做最小可用版本：

- 启动管理器
- Gradio WebUI
- 统一的 provider 适配层
- 基础 voices 资源组织
- `uv / docker / conda` 三种运行方式

## 与 XnneHangLab 的关系

如果你想使用完整项目，请前往主仓库：

- [XnneHangLab](https://github.com/XnneHangLab/XnneHangLab)

如果你想要的是：

- 更轻的仓库
- 更快的展示效果
- 更聚焦的 TTS Demo

那这个仓库就是为此准备的。
