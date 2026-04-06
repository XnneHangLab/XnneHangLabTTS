<p align="center">
  <a href="https://xnnehang.top/">
    <img src="./assets/imgs/logo-full.jpg" alt="魔女の实验室" width="270" />
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
> 这个仓库现在不再负责桌面启动器本身。
>
> 启动器与桌面壳层将放在独立仓库中继续演进；
> 而当前仓库收敛为语音产品本体，聚焦 TTS 能力、资源组织、后端适配与运行分发。

> [!TIP]
> 当前方向可以理解为：
>
> - [HuixinLauncherTemplate](https://github.com/XnneHangLab/HuixinLauncherTemplate) 负责桌面启动器 / Launcher
> - 绘心 Voice 负责语音产品本体
>
> 两者分离开发，后续由启动器接入本仓库能力。

## 项目定位

绘心 Voice 是一个聚焦语音生成与展示体验的产品仓库。

它当前主要负责：

- 多个 TTS 后端的统一接入
- 角色语音资源组织
- provider 适配层
- 运行时依赖与分发方式整理
- 用更轻的方式展示和验证语音能力

启动器、桌面壳层、exe 入口不再作为这个仓库的核心职责。

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
> 把它们拆开后会更清楚：
>
> - 本仓库专心做语音能力
> - 启动器仓库专心做桌面入口
> - 后续可以分别迭代，不互相拖累

这意味着当前仓库更像：

- Voice product repository
- TTS runtime repository
- provider integration repository

而不是 Launcher 仓库。

## 计划支持的后端

- GSV-Lite
- Genie-TTS
- faster-qwen-tts

后续会通过统一的 provider 适配层来组织调用方式，而不是让上层直接耦合到具体实现。

## 第一阶段目标

第一阶段先把这些事情做稳：

- provider 适配层
- 基础 voices 资源组织
- `uv / docker / conda` 三种运行方式
- CPU / GPU 安装策略区分
- 模型下载职责与运行仓库职责划清

## 快速开始

需要先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和 [just](https://just.systems/man/en/)。

### 安装依赖

```bash
uv sync
```

### 选择 PyTorch 版本

默认配置：**Windows → CUDA 12.8（RTX 50 系 / Blackwell）**，**Linux → CPU**。

先运行 `nvidia-smi`，查看右上角的 **CUDA Version**，对照下表：

| GPU | CUDA Version | 操作 |
|-----|-------------|------|
| 无独显 / 纯测试 | — | 无需修改（Linux 默认 CPU） |
| GTX 10xx ~ RTX 20/30 系旧驱动 | ≤ 11.8 | 切换到 `pytorch-cu118`（见 `pyproject.toml` 注释） |
| RTX 20/30/40 系新驱动 | 12.4 | 切换到 `pytorch-cu124`（见 `pyproject.toml` 注释） |
| RTX 50 系 (Blackwell) | ≥ 12.8 | 默认，无需修改 |

切换版本时，按 `pyproject.toml` 中各 index 块的注释步骤操作。

> [!WARNING]
> 切换 torch 版本前，**必须先删除 `.venv` 和 `uv.lock`**，否则旧版本不会被替换：
>
> ```bash
> # Linux / macOS
> rm -rf .venv uv.lock
>
> # Windows PowerShell
> Remove-Item -Recurse -Force .venv, uv.lock
> ```
>
> 然后重新执行 `uv sync`。

安装完成后验证：

```
just torch-check
```

## 与 XnneHangLab 的关系

如果你想使用完整项目，请前往主仓库：

- [XnneHangLab](https://github.com/XnneHangLab/XnneHangLab)

如果你想关注语音产品线本身，那当前仓库就是这个方向。
