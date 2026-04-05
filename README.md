<p align="center">
  <a href="https://xnnehang.top/">
    <img src="./assets/imgs/logo.svg" alt="魔女の实验室" width="270" height="180" />
  </a>
</p>

<h1 align="center">绘心 Voice</h1>

<p align="center">
  来自 <a href="https://github.com/XnneHangLab/XnneHangLab">XnneHangLab</a> 的语音产品仓库
</p>

<p align="center">
  <img src="https://img.shields.io/badge/语言-中文优先-blue" />
  <img src="https://img.shields.io/badge/TTS-GSV--Lite%20%7C%20Genie--TTS%20%7C%20faster--qwen--tts-orange" />
  <img src="https://img.shields.io/badge/运行方式-uv%20%7C%20docker%20%7C%20conda-6f42c1" />
  <img src="https://img.shields.io/badge/状态-WIP-ff69b4" />
</p>

---

> [!NOTE]
> 当前仓库不再负责桌面启动器本身。
>
> 启动器与桌面壳层将放在独立仓库中继续演进；
> 而当前仓库收敛为语音产品本体，聚焦 TTS 能力、资源组织、后端适配与运行分发。

> [!TIP]
> 当前产品线可以这样理解：
>
> <p align="center">
  <a href="https://xnnehang.top/">
    <img src="./assets/imgs/logo.svg" alt="魔女の实验室" width="270" height="180" />
  </a>
</p>

<h1 align="center">绘心 Voice</h1>

<p align="center">
  来自 <a href="https://github.com/XnneHangLab/XnneHangLab">XnneHangLab</a> 的语音产品仓库
</p>

<p align="center">
  <img src="https://img.shields.io/badge/语言-中文优先-blue" />
  <img src="https://img.shields.io/badge/TTS-GSV--Lite%20%7C%20Genie--TTS%20%7C%20faster--qwen--tts-orange" />
  <img src="https://img.shields.io/badge/运行方式-uv%20%7C%20docker%20%7C%20conda-6f42c1" />
  <img src="https://img.shields.io/badge/状态-WIP-ff69b4" />
</p>

---

> [!NOTE]
> 当前仓库不再负责桌面启动器本身。
>
> 启动器与桌面壳层将放在独立仓库中继续演进；
> 而当前仓库收敛为语音产品本体，聚焦 TTS 能力、资源组织、后端适配与运行分发。

> [!TIP]
> 当前产品线可以这样理解：
>
> - 绘心启动器 [HuixinLauncherTemplate](https://github.com/XnneHangLab/HuixinLauncherTemplate) 负责桌面启动器 / Launcher
> - 绘心 Voice 负责语音产品本体
>
> 两者分离开发，后续由启动器接入本仓库能力。

## 项目定位

绘心 Voice 是一个聚焦语音生成与展示体验的产品仓库，当前主要负责：

- 多个 TTS 后端的统一接入
- 角色语音资源组织
- provider 适配层
- 运行时依赖与分发方式整理
- 用更轻的方式展示和验证语音能力

## 当前目标

当前仓库聚焦三件事：

### 1. TTS 后端统一接入

支持并整理这些后端：

- GSV-Lite
- Genie-TTS
- faster-qwen-tts

### 2. 语音资源与推理体验

负责：

- 角色 / 情绪 / 风格资源组织
- provider 参数收敛
- 统一调用入口
- 生成与试听体验

### 3. 运行与分发

支持三种方式：

- `uv`
- `docker`
- `conda`

并逐步明确：

- CPU 版本怎么装
- GPU 版本怎么装
- 模型如何按需下载

## 为什么这样拆分

> [!IMPORTANT]
> 语音产品本体和桌面启动器不是一个层级的问题。
>
> 拆开后更清楚：
>
> - 本仓库专心做语音能力
> - 启动器仓库专心做桌面入口
> - 后续可以分别迭代，不互相拖累

## 与 XnneHangLab 的关系

如果你想使用完整项目，请前往主仓库：

- [XnneHangLab](https://github.com/XnneHangLab/XnneHangLab)

> - 绘心 Voice 负责语音产品本体
>
> 两者分离开发，后续由启动器接入本仓库能力。

## 项目定位

绘心 Voice 是一个聚焦语音生成与展示体验的产品仓库，当前主要负责：

- 多个 TTS 后端的统一接入
- 角色语音资源组织
- provider 适配层
- 运行时依赖与分发方式整理
- 用更轻的方式展示和验证语音能力

## 当前目标

当前仓库聚焦三件事：

### 1. TTS 后端统一接入

支持并整理这些后端：

- GSV-Lite
- Genie-TTS
- faster-qwen-tts

### 2. 语音资源与推理体验

负责：

- 角色 / 情绪 / 风格资源组织
- provider 参数收敛
- 统一调用入口
- 生成与试听体验

### 3. 运行与分发

支持三种方式：

- `uv`
- `docker`
- `conda`

并逐步明确：

- CPU 版本怎么装
- GPU 版本怎么装
- 模型如何按需下载

## 为什么这样拆分

> [!IMPORTANT]
> 语音产品本体和桌面启动器不是一个层级的问题。
>
> 拆开后更清楚：
>
> - 本仓库专心做语音能力
> - 启动器仓库专心做桌面入口
> - 后续可以分别迭代，不互相拖累

## 与 XnneHangLab 的关系

如果你想使用完整项目，请前往主仓库：

- [XnneHangLab](https://github.com/XnneHangLab/XnneHangLab)
