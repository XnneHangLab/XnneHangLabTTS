# Genie Runtime And Launcher Phase 1 Design

**目标**

在当前仓库内完成第一阶段联调设计，优先打通 Windows 下的 CPU/GPU 环境识别、GenieData 下载管理、Launcher 状态展示和日志链路，为第二阶段接入 Gradio WebUI 启停保留稳定接口。

**范围**

第一阶段只覆盖这些能力：

- 自动识别当前 Python 环境是 CPU 还是 GPU
- 识别 `GenieData` 是否处于 `missing / partial / ready`
- 通过 Launcher 发起 `GenieData` 下载
- 下载任务进入串行队列
- 下载目标落到仓库内固定目录
- 首页显示下载摘要和目录入口
- 控制台页显示详细日志事件
- 下载完成后自动校验并刷新状态

第一阶段明确不做这些：

- 不做角色模型包下载
- 不做 `gsv-tts-lite` 和 `faster-qwen-tts` 的实际接入
- 不做 Python 环境安装
- 不做用户手动切换 CPU/GPU
- 不做 WebUI 热加载和热拆卸
- 不做持久化下载队列恢复

**总体思路**

`launcher/` 作为桌面入口，负责界面、任务操作、日志展示和目录打开。Tauri Rust 负责本地命令调用、子进程生命周期、下载队列执行和事件分发。Python 主仓负责环境检测、目录校验、ModelScope 下载和后续 Gradio 启动壳。

第一阶段不引入常驻 Python 服务，不使用本地 HTTP API。Launcher 通过 Tauri 命令间接调用 Python CLI。这样做可以先把下载和日志链路做稳，第二阶段再用同一条链路接入 `launch-webui`。

**子项目拆分**

这次需求拆成三个连续子项目：

1. 第一阶段：环境识别 + GenieData 下载管理 + Launcher 联调
2. 第二阶段：Genie-TTS 最小 Gradio WebUI 启停
3. 第三阶段：角色包下载、多后端注册表、后端切换、热加载探索

当前文档只定义第一阶段。

## 架构边界

### Python 主仓职责

- 提供统一 CLI 入口
- 识别当前运行时环境模式
- 维护下载目标定义
- 执行 ModelScope 下载
- 校验 `GenieData` 目录是否完整
- 输出结构化事件或结构化结果

### Tauri Rust 职责

- 暴露前端可调用命令
- 启动和管理 Python 子进程
- 抓取 `stdout / stderr`
- 维护串行下载队列
- 将结构化事件转发给前端
- 管理日志导出和受管目录打开

### React Frontend 职责

- 展示环境模式、资源状态和队列摘要
- 提供下载按钮和目录入口
- 展示控制台事件流
- 根据后端返回能力决定允许操作

## 目录与配置

第一阶段统一使用仓库根目录下的受管路径：

- `models/genie/base/`
  放 `GenieData` 基础资源
- `models/genie/characters/`
  预留给后续角色模型包，第一阶段只建立目录约定
- `models/cache/modelscope/`
  放 ModelScope 缓存
- `logs/downloads/`
  放导出后的日志文件
- `config/runtime.toml`
  放运行时共享配置

`config/runtime.toml` 第一阶段包含这些键：

- `workspace_root`
- `models_root`
- `cache_root`
- `logs_root`
- `default_backend = "genie-tts"`

所有路径都基于仓库根目录推导，不写死 Windows 分隔符。这样可以优先打通 Windows，同时尽量避免后续 Linux 兼容时的大规模返工。

## 环境模式与能力表

环境模式由 Python 端自动识别，不允许用户手动切换。

识别逻辑：

- 能导入 `torch`
- `torch.cuda.is_available()` 为真时，模式为 `gpu`
- 其他情况都视为 `cpu`

第一阶段能力表：

- `cpu`
  - 允许下载 `genie-base`
  - 允许查看目录和日志
  - 后续允许启动 `genie-tts`
- `gpu`
  - 第一阶段仍只允许下载 `genie-base`
  - 能力表中保留“后续可扩展全部后端”的标记

这样做的目的，是先把环境识别和下载链路做稳，不在第一阶段把多后端启动混进来。

## 下载模型与资源边界

第一阶段下载目标只有一个：`genie-base`。

它只负责下载 `GenieData` 基础资源，不与角色模型包绑定。角色模型包即使后续加入，也必须作为独立下载动作和独立队列任务。

下载源以你们自己的 ModelScope 仓库为主，不沿用当前 `Genie-TTS` 内部的 HuggingFace 下载逻辑作为产品主路径。为了避免把下载源写死在代码里，第一阶段需要把下载目标元数据单独定义，让 CLI 根据目标定义执行下载。

`genie-base` 的下载定义至少要包含：

- `target_id`
- `provider`
- `repo_id`
- `allow_patterns`
- `local_dir`
- `cache_dir`
- `required_paths`

其中：

- `provider` 第一阶段固定为 `modelscope`
- `local_dir` 指向 `models/genie/base/`
- `cache_dir` 指向 `models/cache/modelscope/`
- `required_paths` 用来做下载后校验

## 队列与任务状态

下载任务从第一阶段开始就按正式队列对象设计，不把下载按钮做成一次性动作。

队列策略：

- 只做单 worker 串行执行
- 允许一次排入多个任务
- 后进任务等待前一个任务完成或失败

这样做可以避免大模型并发下载导致的日志混乱、缓存争用和失败处理复杂化。

任务状态机固定为：

- `queued`
- `preparing`
- `downloading`
- `verifying`
- `completed`
- `failed`
- `cancelled`

第一阶段的队列状态只保存在内存中，不做任务恢复。应用重启后，通过重新扫盘得到资源状态，而不是试图恢复中断下载。

## Launcher 界面分工

首页和控制台页分工必须从第一阶段就固定：

- 首页看摘要
- 控制台看细节

首页至少显示：

- 当前环境模式 `cpu / gpu`
- `genie-base` 当前状态 `missing / partial / ready`
- 当前下载任务
- 队列长度
- 最近一条状态描述
- 模型目录、缓存目录、日志目录入口

控制台页至少显示：

- 结构化事件生成的日志条目
- 原始 `stdout / stderr` 输出
- 日志导出
- 清空当前视图

进度条和任务摘要不应只靠控制台文字表达。用户需要在首页就能一眼看出当前谁在下载、是否在排队、是否完成。

## Launcher 与 Python 的调用协议

第一阶段不启用本地 HTTP 服务，统一使用 CLI。

Python 侧提供这几个命令：

- `uv run python -m xnnehanglab_tts.cli inspect-runtime`
- `uv run python -m xnnehanglab_tts.cli download genie-base`
- `uv run python -m xnnehanglab_tts.cli verify genie-base`

Tauri Rust 侧对前端暴露这几个命令：

- `inspect_runtime()`
- `enqueue_download(target)`
- `list_download_tasks()`
- `open_managed_path(path_key)`
- `export_console_logs()`

调用链路固定为：

1. 前端调用 Tauri 命令
2. Rust 将任务排入队列或执行即时检查
3. Rust 启动 Python CLI
4. Python 输出结构化结果和事件
5. Rust 转发给前端
6. 前端更新状态并生成控制台展示

## 事件与日志

控制台不是状态源，只是事件视图。业务状态必须由结构化事件驱动，不允许前端解析散乱文本来判断任务成功失败。

事件最小字段：

- `event`
- `task_id`
- `timestamp`
- `target`
- `status`
- `message`
- `progress_current`
- `progress_total`
- `progress_unit`

第一阶段标准事件名：

- `runtime.inspected`
- `download.queued`
- `download.started`
- `download.progress`
- `download.verifying`
- `download.completed`
- `download.failed`
- `download.cancelled`

Rust 需要同时处理两类输出：

1. 结构化事件
2. 原始 `stdout / stderr`

结构化事件用于状态更新，原始输出用于控制台完整可见性。即使第三方库只输出纯文本，Rust 也要把它们转成控制台日志项，而不是丢弃。

## 目录校验规则

资源状态不依赖历史记录，而依赖实际文件检查。

校验结果只允许三种：

- `missing`
  必要路径全部缺失，或主目录不存在
- `partial`
  主目录存在，但 `required_paths` 不完整
- `ready`
  所有必要路径齐全

下载完成后必须自动执行一次校验，并把最终结果回写到前端状态。

## 与第二阶段的衔接

第二阶段接入 `Genie-TTS` 最小 `Gradio` WebUI 时，不改第一阶段主链路，只新增一个 CLI 命令，例如：

- `uv run python -m xnnehanglab_tts.cli launch-webui --backend genie-tts`

这样可以直接复用：

- Rust 子进程管理
- `stdout / stderr` 捕获
- 前端控制台展示
- 首页状态入口

CPU 下只允许启动 `genie-tts`。GPU 下的 `gsv-tts-lite` 和 `faster-qwen-tts` 只在后续阶段接入，不提前混入第一阶段。

## 验证策略

第一阶段每个子模块都要有小验证点，避免做到最后才发现链路断掉。

验证层次：

1. Python 单元级
   - 环境识别函数
   - 路径推导函数
   - 下载目标定义
   - 目录校验函数
2. Tauri / Rust 单元级
   - 队列状态流转
   - 事件转发
   - 日志聚合
3. React 单元级
   - 首页摘要展示
   - 控制台事件展示
   - 队列状态映射
4. 联调级
   - 点击下载后能看到排队、开始、完成、校验
   - 下载后能打开目标目录
   - 重启后能重新扫出 `ready`

## 实施顺序

第一阶段建议按这条顺序实现：

1. Python 侧先做路径、配置、环境识别、校验
2. 再做 `genie-base` 下载命令
3. Rust 侧补 Tauri 命令和串行队列
4. 前端补首页摘要和目录入口
5. 前端补控制台事件展示
6. 最后做一轮 Windows 优先联调

这样每一步都能独立验证，出问题时也更容易定位是在 Python、Rust 还是前端。

## 风险与约束

第一阶段的主要风险：

- ModelScope 下载进度格式不稳定，导致细粒度百分比难统一
- Windows 路径和权限细节容易影响目录打开和日志导出
- `torch` 环境不完整时，自动识别逻辑需要给出明确降级结果

对应处理方式：

- 进度事件允许先支持“已开始 / 已完成 / 已校验”，细粒度字节进度作为可选增强
- 目录操作统一经由 Rust 封装
- 环境识别失败时返回结构化错误和可读消息，而不是直接崩溃

## 结论

第一阶段不是做“完整 TTS 产品”，而是做一条稳定主链路：

自动识别环境，下载并校验 `GenieData`，让 Launcher 看到明确状态、明确队列、明确日志。

只要这条链路长对，第二阶段接 `Gradio`，第三阶段接角色包和多后端，都不需要大拆架构。
