# LingYu 设计文档（v0.2 架构）

> 这份文档是**唯一权威来源**。跨多轮实现，每轮开工前先读它，改设计先改它。
> 最后更新：2026-09-19

---

## 0. 一句话定位

> **一个引擎，从 13M 到万亿参数。**
> 同一套配置语法、同一个 GUI、同一条代码路径；只有「执行后端」随规模切换。

面向四类人，但**不牺牲任何一类的体验**：

| 用户 | 规模 | 用什么 |
|---|---|---|
| 完全小白 | 13M | GUI 点几下，30 秒看到效果（`lingyu demo` 计划中，P7，命令尚不存在） |
| 高中生 / 爱好者 | 10M ~ 100M | GUI + 配置文件，单卡 |
| 研究者 / 企业 | 1B ~ 1T+ | 同一份配置，换后端（FSDP2 / DeepSpeed / Megatron） |
| 想要"能问答的模型"的人 | 任意 | 预训练 → SFT 后训练 → 聊天界面 |

**核心承诺一：13M 和 1T 共用同一份配置 schema。**
别人把你的小模型配置 `n_layer` 改大、加上 `parallel` 段，就能原样拿去集群跑。

**核心承诺二：预训练与后训练是一条链。**
`预训练` 得到会续写的基座 → `SFT` 得到会按指令回答的对话模型 →
同一套词表和模型结构，只在训练目标与数据格式上分岔。

---

## 1. 关键设计决策（含理由，别改）

| 决策 | 选择 | 理由 |
|---|---|---|
| **项目名** | **PyPI `lingyu-lm` / import `lingyu` / 命令 `lingyu`** | 仓库名和 PyPI 名要独特好搜；import 名短。符合社区惯例（scikit-learn / sklearn）。PyPI 上已查证可用 |
| **代码组织** | **标准 Python 包 + `pyproject.toml` + `pip install -e .`** | 社区熟悉、能提 PR、能发 PyPI。**代码生成器（`_part_*.py` + base64）在 P1 阶段退场** |
| GUI 框架 | **PySide6** | PyQt6 是 **GPL**，会传染，企业不敢用；PySide6 是 **LGPL**，别人能拿去做闭源商业产品。这条决定项目能不能被采用 |
| 项目许可证 | **Apache-2.0** | 带专利授权，企业更敢用。文档 CC-BY-4.0 |
| 单机多卡 | **FSDP2（`fully_shard`）** | PyTorch 2.4+ 已 production ready，DTensor 逐参数分片，与 TP/PP/EP 可组合 |
| 大模型（多机） | **适配层 + 启动脚本生成** | **不自己实现 Megatron**。诚实算出"需要多少卡"并生成可用脚本 |
| 打包 | **~15MB 启动器 .exe + runtime/ 子目录** | PyInstaller onefile 装 CUDA 版 torch 出来 2.6GB 且每次解压，体验极差。物理上不可能"单文件全打包" |
| 语言 | **i18n 字典，中/英可切** | 开源受众是英文，你自己看中文 |
| 测试 | **单测 + 真实小模型端到端回归** | 每改必跑，防"改一处碎一处" |

---

## 2. 目录结构

```
lingyu-engine\                        ← 仓库根（开发 / 源码安装）
├── src/lingyu/                       ← 引擎核心（唯一开发目标；editable 安装）
│   ├── __init__.py                   公开 API + 版本转出
│   ├── __main__.py                   python -m lingyu 入口
│   ├── version.py                    版本号 + 断点/配置/分词器格式常量
│   ├── registry.py                   组件注册表（三层优先级，见 §3）
│   ├── components.py                 组件分发（刻意不 import model，破循环导入）
│   ├── cli.py                        CLI 入口（11 个子命令）
│   ├── paths.py                      唯一的根目录/配置查找实现（见 §9.6）
│   ├── utils.py                      控制台/日志工具
│   ├── tokenizer.py                  字符级 + 字符级 BPE + 指纹
│   ├── model.py                      RMSNorm / RoPE / GQA / SwiGLU / MoE
│   ├── data.py                       内存映射数据集 + 验证集评估 + BPB
│   ├── prepare.py                    编码嗅探 / 清洗 / 分词器训练 / 切分
│   ├── train.py                      单卡训练循环 + 断点存取
│   ├── sft.py                        SFT：loss mask（target 左移一位）/ 扩词表
│   ├── chat.py                       交互式对话（多轮、流式、上下文按 token 裁剪）
│   ├── chat_template.py              对话模板 + loss mask + 四种 JSONL 读取
│   ├── sample.py                     预训练采样（真正循环在 model.generate）
│   ├── config/
│   │   ├── __init__.py
│   │   ├── spec.py                   参数字典（类型/范围/默认/帮助），单一事实源
│   │   ├── loader.py                 INI/JSON/YAML 读写、合并、--set、patch_config
│   │   └── validate.py               组合校验（整除/依赖/候选/规模警告）
│   └── backends/
│       ├── __init__.py               后端能力声明（IMPLEMENTED = ("local",)）
│       ├── plan.py                   全项目唯一显存估算器（plan/validate 共用）
│       └── launch.py                 分布式启动脚本文本生成（fsdp/deepspeed/megatron）
├── gui/                              ← PySide6 桌面界面（启动 <2s，绝不 import torch）
│   ├── app.py                        主窗口：侧边栏 + 页面堆栈 + 状态栏
│   ├── theme.py                      配色表 + 整份 QSS（深浅双主题共用语义色名）
│   ├── i18n.py                       中英文案表（Python 字典，非 json 文件）
│   ├── context.py                    全局状态 + 信号总线 + 后台环境探测
│   ├── widgets.py                    Card / StatTile / Field / LogView / PathPicker…
│   ├── chart.py                      手绘 loss 曲线（QPainter，不引 matplotlib）
│   ├── paramform.py                  由 spec 自动生成的参数表单 + tier 折叠
│   ├── runner.py                     QProcess 子进程管理 + 进度行解析
│   ├── audit.py                      界面巡检/关键状态截图（开发流程用）
│   ├── shot.py                       截图工具
│   └── pages/                        九个页面，一页一个文件
│       ├── overview.py / corpus.py / train.py / monitor.py / checkpoints.py
│       └── sample.py / sft.py / chat.py / settings.py
├── packaging/                        ← 打包成「一个文件夹 + 一个 exe」
│   ├── launcher.py                   启动器（标准库 + tkinter；装环境/解包/启动）
│   └── build_exe.py                  构建脚本（PyInstaller + 源码 zip + 发布 zip）
├── configs/                          出厂配置：nano65 / small / medium / moe / wide / sft.ini
│                                      + chat.ini（预训练→SFT→chat 一条配置）
├── chat/                             对话数据目录（sample_chat.jsonl + README
│                                      + fetch_sft_data.py 下载 Alpaca-GPT4 中文）
├── plugins/                          第三方插件目录（丢 .py 即生效）+ README.md
├── docs/
│   ├── DESIGN.md                     本文件（唯一权威设计文档）
│   ├── PROJECT_STATE.md              压缩后的权威状态快照（开发/发布背景）
│   └── OPTIMIZATION*.md              2.1–2.6 / Round2–4 的优化实测记录
├── tests/                            pytest 套件（32 个收集文件 / 619 条用例）
│   ├── 配置：test_config_system 43 / test_config_integrity 39 / test_config_writeback 22
│   ├── 核心：test_model_config 30 / test_model 9 / test_data 4 / test_train 18 / test_prepare 12
│   ├── 生成与对话：test_chat 22 / test_chat_template 28 / test_sft 21 / test_tokenizer_roundtrip 8
│   ├── 组件与工程：test_components 17 / test_registry 18 / test_backends 32 / test_paths 11
│   ├── 组件级：test_gui（offscreen）84 / test_packaging（静态检查 + 子进程）59
│   └── 端到端：test_cli（`-m slow`：真训练 12 步的 scan→train→sample→resume→SFT→chat）25
├── 读我先看.md / 命令手册.md / 引擎原理.md   中文文档
├── README.md / README.zh-CN.md        英文门面 / 中文完整版
├── LICENSE                           Apache-2.0
├── pyproject.toml                    安装、依赖、ruff、pytest 配置
├── lingyu.ps1                        便捷入口（$PSScriptRoot 相对定位）
├── setup_env.ps1                     本机开发环境一键安装
├── gpu_smoke_test.py                 本机 GPU 冒烟测试（开发工具）
├── _archive/                         已删旧结构的归档（pre-P1-flat-package.zip，不进 git）
├── _shots/                           gui/shot.py 截图产物（不进 git）
├── data/ data1/ raw/ out/            用户数据与训练产物（gitignore）
├── venv/                             开发环境（gitignore）
├── dist/ build/ wheels/ python312/   打包/构建/离线 wheel（gitignore）
└── runtime/                          打包版运行时（只出现在 dist/LingYu/ 下，gitignore）
```

---

## 3. 组件注册表（`registry.py`）—— 架构可插拔的核心

**目标**：别人不改我的核心代码，就能换掉注意力/前馈/位置编码/归一化。

### 三层优先级（同名时后面的覆盖前面的）

```
1. 内置实现      @register("gqa")              ← lingyu/model/*
2. 插件目录      plugins/*.py 自动加载          ← 用户放文件即可
3. 用户配置指定  model.attention = "my_attn"    ← 配置文件里点名
```

### 接口约定（写死，插件必须遵守）

```python
# 注意力：输入 (B,T,d) 输出 (B,T,d)，支持 KV cache
class AttentionProtocol:
    def __init__(self, cfg: dict, layer_idx: int): ...
    def forward(self, x, cache=None, cache_pos=0): ...

# 前馈：输入 (B,T,d) 输出 (B,T,d)
class FFNProtocol:
    def __init__(self, cfg: dict): ...
    def forward(self, x): ...

# 位置编码
class PosEncodingProtocol:
    def forward(self, seq_len, device, dtype) -> (cos, sin): ...
```

### 配置里怎么用

```ini
[model]
attention  = gqa        # 内置：mha / gqa / mqa
ffn        = moe        # 内置：swiglu / moe / geglu
norm       = rmsnorm    # 内置：rmsnorm / layernorm
pos        = rope       # 内置：rope / learned / alibi / none
```

改成 `attention = my_attention`（插件里注册的名字）就能换掉。

---

## 4. 配置系统：单一事实源（`config/spec.py`）

**问题**：现在参数散落在 `config.py` 的 `KNOWN_KEYS`、`train.py` 的 `get()` 调用、
`model.py` 的 `build_config()` 三处，加参数要改三个地方，容易漏。

**方案**：一份 `spec.py` 描述所有参数，其他全从它生成。

```python
PARAMS = {
    "model": {
        "n_layer":  dict(type=int,   default=8,  min=1, max=1024,
                         help_zh="层数", help_en="Number of layers",
                         tier="basic"),          # basic/advanced/expert 决定 GUI 折叠层级
        "n_embd":   dict(type=int,   default=512, min=8, max=65536,
                         help_zh="宽度", help_en="Embedding dimension",
                         tier="basic",
                         constraints=["divisible_by:n_head"]),
        ...
    },
    "parallel": {                                # ★新：分布式段
        "backend":       dict(type=str, choices=["local","fsdp","deepspeed","megatron"],
                              default="local"),
        "tensor_parallel": dict(type=int, default=1, min=1),
        "pipeline_parallel": dict(type=int, default=1, min=1),
        "zero_stage":    dict(type=int, choices=[0,1,2,3], default=2),
        "cpu_offload":   dict(type=bool, default=False),
        ...
    },
}
```

**一份 spec 同时驱动**：
1. CLI 的参数解析与校验
2. GUI 的表单生成（按 `tier` 折叠：基础/进阶/专家）
3. `--help` 文案
4. 配置文件的注释模板
5. 文档里的参数表

**统一校验入口**：所有 CLI 子命令都走 `cli.load_cfg()`（读配置 → `--set` 覆盖 → `resolve()` 补全 spec 默认值 → `_validate_cfg()`）；配置错误在开跑前以退出码 2 报人话 —— 例如 `--set model.n_head=0` 不会再拖到建模型才抛 `ZeroDivisionError: integer modulo by zero`，`n_layer=-5` 也不会建出负层数模型。

**收益**：加一个参数只改一处，不会出现"文档说能改但代码不认"这类 bug。

---

## 5. 执行后端（`backends/`）

| 后端 | 适用 | 做什么 | 状态 |
|---|---|---|---|
| `local` | 单卡装得下 | 直接训练 | 已有，重构 |
| `fsdp` | 单机多卡 | `torchrun` + FSDP2 + activation checkpointing | **实验性：仅生成启动脚本，多卡训练循环未实现** |
| `deepspeed` | 多机 | 生成 `ds_config.json` + 启动脚本 | 适配层 |
| `megatron` | 超大 | 生成 Megatron 启动脚本 + 参数映射 | 适配层 |
| （未来）`ray` | 集群调度 | — | 预留 |

### 诚实原则（写进代码）

训练前**一定**先算：
```
本机可用显存 vs 这个配置需要多少显存
```
装不下时**不启动训练**，而是：

```
[提示] 这个配置需要约 2.4 TB 显存（bf16 + AdamW + ZeRO-3）
       本机只有 7.93 GB

       它不可能在本机训练。三个选择：
       ① 调小配置（把 n_layer 从 80 改成 8）
       ② 换 FSDP 后端在多卡机器上跑：  --set parallel.backend=fsdp
       ③ 生成集群启动脚本：  lingyu launch --backend deepspeed
```

GUI 里同样逻辑：参数一改就实时显示"需要多少显存 / 本机装不装得下"。

**估算口径（实测标定，`backends/plan.py`，2026-10-18）**：用 5 份出厂配置各跑 8~10 步、读 `torch.cuda.max_memory_allocated` 峰值做最小二乘标定 —— 激活系数 **29**（旧值 22 偏小）+ **0.7 GB 固定开销**（CUDA 上下文/分配器碎片/临时张量）+ MoE 再乘 **1.4**。实测峰值：nano65 1.94 / small 2.68 / medium 5.84 / moe 4.34 / wide 5.96 GB，公式误差 **−7% ~ +1%**（旧公式低估 20%~50%）。判定：`usable = 可用显存 × 0.92`、`估算 × FITS_MARGIN(1.03) ≤ usable` 才算够。

---

## 6. GUI 设计（PySide6）

### 启动速度是第一优先级

用户要求"小破显卡也得能打开"。**GUI 本身绝不导入 torch** ——
训练跑在**独立子进程**里，GUI 只通过管道读它的日志和指标。

这样：
- GUI 启动 **< 2 秒**，内存 < 150 MB
- 训练吃满显存也不影响界面响应
- 训练崩了 GUI 不会跟着崩

### 页面

| 页面 | 内容 |
|---|---|
| **概览** | 环境自检结果、「按当前配置」的参数量/显存结论、最近项目入口 |
| **语料准备** | 选目录 → 文件清单（大小/编码）→ 清洗选项 → 预计词表/压缩比 → 进度 |
| **训练** | 参数表单（按 tier 折叠）+ 实时 loss 曲线 + 日志 + 断点管理 + 停止按钮（优雅停止**已实现**） |
| **监控** | loss/学习率/tok-s 曲线与运行状态 |
| **断点** | 列出 out_dir 的 ckpt/best、续训与结构诊断 |
| **续写** | 给开头 → 温度/top-k/top-p → 实时出文字 |
| **微调 SFT** | 对话数据导入 + loss mask 预览 + SFT 超参 |
| **聊天** | 多轮对话（保持 KV cache）、流式输出、长 prompt 自动裁剪 |
| **设置** | 语言/主题/显存探测/路径诊断 |

> **训练页的「停止」（已实现，0.3.1）**：引擎（`train.py` / `sft.py`）每步轮询环境变量 `LINGYU_STOP_FILE` 指向的文件，收到停止请求就在**当前这一步结束时**保存最终断点、正常退出；GUI 点停止 = 先请求保存再退出（4 秒未退出才兜底强杀，且先校验目标进程身份）。实测 SFT 第 60 步停止成功写盘。

### 参数表单的三层折叠

```
[基础]    n_layer / n_embd / n_head / block_size / batch_size / max_steps   ← 小白只看这些
[进阶]    d_ff / dropout / lr / warmup / 分辨率相关                        ← 想调优的看
[专家]    MoE 全部 / GQA / 注册表选择 / parallel 全部 / 混合精度 / 累积     ← 大模型用户看
```

### i18n

所有界面文案走 `gui/i18n.py` 里的中英双字典（相当于 `i18n/zh_CN` + `en_US` 两张表），顶部有语言下拉。**从第一天就这么做**，
后补的成本高得多。

### 界面改版（1.1 / 1.3 / 1.5，2026-10-24 起）

用户对 GUI 提了一整套要求（完整需求与分批计划见 `docs/PROJECT_STATE.md`），分多轮做。第一轮先把
「地基」立住，改完的东西必须能长期长出别的页面：

1. **参数说明改成 `?` 徽标**：可调参数后面一个灰色小圆圈问号，完整说明放
   `QToolTip`（富文本 + 显式宽度，Qt 的纯文本 tooltip 不换行），悬停才出现。
   理由：参数一多，常驻灰色说明就是「一墙小字」，既挤又没人读。
   实现：`gui/widgets.HelpBadge` + `Field(help_text=...)`；`ParamForm` 自动取
   `spec` 的 `help_zh/help_en`；SFT 页的 8 个非 spec 字段手写 `help.sft.*` 文案。
   **没写 help_text 的字段不会长徽标** —— 这是刻意的：宁缺毋滥。
2. **训练页「运行看板」**：训练中不再是一屏 PowerShell 滚动日志，而是
   `状态行 + 进度条 + 2×2 数字格（步数 / loss / lr / 速度）`。
   数据走 `ctx.task_progress` 总线（页面重建后仍能收到），日志收进可折叠的
   「详细日志」（`QPlainTextEdit`，只有用户点开才渲染）。
   速度用「相邻两次 progress 的增量」算，不能拿 `elapsed/step`（续训会清零）。
3. **滚动条误改参数**：鼠标滚轮停在未获焦点的输入框上时，把滚轮事件转交给
   外层滚动区，而不是改数值。实现：`gui/widgets._WheelGuard` + `guard_wheel()`，
   焦点策略 `StrongFocus`（点进去才能滚轮调值）。
4. **侧栏与切页**：品牌区加色块 + 版本号（`gui.app`，版本从 `lingyu.version` 读），
   当前导航项左侧 2px 强调条；切页 140ms 淡入（`LINGYU_GUI_NO_ANIM=1` 可关，
   截图/测试用）。动效只做「不打断阅读」的淡入，不做飞入/缩放。

配套：`gui/audit.py` 新增 `LINGYU_GUI_NO_ANIM`，截图看到的是最终稳定画面；
`tests/test_gui.py` 用离线 Qt 钉住徽标、看板、滚轮行为。

---


### 实测 bug 修复批次（2026-10-24，B-01…B-11）

用户拿打包版实测后报的一轮问题，全部记在 `docs/PROJECT_STATE.md` 的 GUI 批次记录里，修完的落点：

- **参数帮助**：`HelpBadge` 用 `enterEvent` 显式 `QToolTip.showText()`（不再赌 Qt
  tooltip 计时器），QToolTip 两套主题都有 QSS；测试 `test_help_badge_shows_tooltip_explicitly`。
- **续训不再盲训**：训练页「续训起点」下拉 = 自动 / 当前 run 的各断点 / 断点页全部断点；
  跨结构、跨 run 时自动 `--set model.*` 对齐或改走 `--init-from`（从 SFT 断点继续
  预训练也走这条路）。
- **断点页**：「来源」= 全部 / 当前项目；行前缀「预训练 ·」「SFT ·」；新断点写同名
  `.json` sidecar（`ckpt_format/step/best_val/model_cfg/tokenizer_fingerprint/config_path`），
  断点页优先读 sidecar，旧 `.pt` 回退异步加载。
- **配置**：训练页配置下拉有「自定义（默认参数）」；「另存为配置」把当前参数写成
  新 `configs/*.ini`。
- **JSON 语料**：`extract_json_text` 通用递归抽取；配置文件类 JSON 按文件名/顶层键整份
  跳过；`prepare`/`scan` 日志限行（前 10 + 每 500 一行 + 跳过汇总）。
- **语言**：切换语言时 `set_lang()` 后再重建窗口（此前只重建不换字典）。
- **显存**：概览页加「刷新」+「检测于 HH:MM:SS」，`ctx.refresh_env()` 清缓存重探。

## 7. 后训练与对话（让它能回答问题）

### 7.1 为什么必须单独做一段

**基座模型本质上只会续写。** 输入"中国的首都是"，它接"北京"；
输入"请回答：中国的首都是哪里？"，它会把整句当成文章开头，续写更多"请回答：…"。
**因为它从没见过"问答"这种格式。**

要让它回答问题，必须再做一步 **SFT（指令微调）**。

### 7.2 四个阶段的分工（已实现：预训练 + SFT；DPO/LoRA 仍在计划中）

| 阶段 | 目标 | 数据形态 | 代码 |
|---|---|---|---|
| **预训练** | 学语言规律与世界知识 | 连续文本，无标注 | `train.py` ✅ |
| **SFT** | 学"按指令回答"的格式与口吻 | 对话对：`(指令, 回答)` | `sft.py` ✅ |
| **DPO**（可选） | 让回答更符合人类偏好 | 偏好对：`(好回答, 坏回答)` | 计划中（P7），**未实现** |
| **LoRA**（可选） | 低成本改造已有模型 | 同 SFT，但只训少量参数 | 计划中（P7），**未实现** |

### 7.3 对话模板（最容易出错的地方）

**训练时的拼接格式必须和推理时完全一致**，否则模型看到的是它没学过的格式，直接崩。

```
<|im_start|>system
你是一个乐于助人的助手。<|im_end|>
<|im_start|>user
中国的首都是哪里？<|im_end|>
<|im_start|>assistant
北京。<|im_end|>
```

设计要点：
- 模板配置化：`chat.template = chatml | llama3 | qwen | 自定义`
- 特殊 token 在分词器训练阶段就预留（**不能等 SFT 时才加**，否则词表对不上）
- `lingyu chat-template check` 子命令（把拼接结果原样打出来核对）：**计划中，尚未实现**

### 7.4 Loss mask —— SFT 的核心技巧

**只对"助手回答"那部分算 loss，问题和模板部分不算。**

```
<|im_start|>user       ← loss 忽略
中国的首都是哪里？      ← loss 忽略
<|im_end|>             ← loss 忽略
<|im_start|>assistant  ← loss 忽略
北京。                  ← ★ 只有这里算 loss
<|im_end|>             ← ★ 算 loss（这是「我说完了」的信号）
```

**为什么**：如果全算 loss，模型会连"用户会问什么"一起学，
结果就是它开始**自己生成问题**，而不是回答你。这是新手最常踩的坑。
而且问题通常比回答长，不 mask 的话大部分梯度都花在"抄题"上。

**实现方式（P2 定稿）：不改模型。**
把不该学的位置的 `targets` 设成 `-1`，`F.cross_entropy(ignore_index=-1)`
本来就会跳过它们。`model.py` 里早就有这个参数，所以 P2 一行模型代码都没改。

对比另一种常见做法——在模型 forward 里加 `loss_mask` 参数：
那样每个模型实现（含插件）都得跟着改接口，成本高得多。
**用 ignore_index 是把 masking 变成数据问题，不是模型问题。**

> ⚠️ 别忘了一个坑：一条样本如果 mask 全 0（只有 user、没有 assistant），
> 它必须**整条丢掉**。曾经在末尾无条件补一个 `mask=1` 的 `<eos>`，
> 于是这种样本也被当成有效样本，模型学到的是
> 「用户说完话，我应该回一个 EOS」= **被问了就别吭声**。

### 7.4.1 Target 位移（SFT 的第二个坑，已修）

因果语言模型在位置 i 的输出预测的是**第 i+1 个 token**。SFT 的 targets 必须相应左移一位：

```
tgt[i] = ids[i+1]   iff   mask[i+1] == 1     # 只监督回答位置
tgt[i] = -1                               # 其余位置跳过（ignore_index）
```

**旧实现写成 `tgt[i] = ids[i]`** —— 等于让模型学「复制当前 token」的恒等映射，不是学下一个 token，白训。已修复、有测试钉住。有效监督 token 数用 `(tgt != -1).sum()` 统计（第 0 个位置永远没有可预测目标），**不是 mask 之和**。

**评估口径**：
- 预训练验证 loss（`data.eval_loss`）：按**有效 token 数**加权平均，上下文用 `block_size` 整段评估。
- SFT 验证 loss（`sft.eval_sft_loss`）：**只统计回答位置**，通常明显低于预训练 loss —— 它衡量「回答位置预测得准不准」，不能与预训练 loss 或不同模板横向比较。

### 7.5 对话数据格式

标准 jsonl，一行一个样本：

```json
{"messages": [{"role": "user", "content": "中国的首都是哪里？"},
              {"role": "assistant", "content": "北京。"}]}
```

宽松解析，下面几种都认（网上中文数据集的格式五花八门，
逼用户先写转换脚本是很烦的事）：

| 写法 | 例子 |
|---|---|
| `messages` | `[{"role":"user","content":"…"}]`，role 也认中文「用户/助手/系统」 |
| `conversations` | `[{"from":"human","value":"…"},{"from":"gpt","value":"…"}]` |
| `prompt` / `response` | 最简两字段 |
| `instruction` / `input` / `output` | Alpaca 风格 |

认不出来时报错会**指出是第几行**，并列出支持的四种格式和实际拿到的键。

### 7.5.1 模板：为什么默认是 `plain` 而不是 chatml

| 模板 | 形式 | 需要特殊 token |
|---|---|---|
| **`plain`（默认）** | `用户：…\n助手：…` 纯中文标记 | **不需要** |
| `chatml` / `qwen` | `<\|im_start\|>user\n…<\|im_end\|>` | 需要 |

`plain` 的标记本来就是普通汉字，因此：

- **零词表改动**，任何老断点直接能用
- **对小模型更友好**：中文标记在预训练语料里出现过千万次，embedding 是
  训练好的；而 `<|im_start|>` 只在 SFT 阶段见到，13M 的模型根本没足够
  数据把它学明白

chatml 的价值在**互通性**（和 HuggingFace 生态对接），所以两个都留着。

新建的分词器把 `<|im_start|>` / `<|im_end|>` 放进 `SPECIAL_TOKENS`
（`tokenizer.py`），所以 chatml 开箱可用。老分词器用 chatml 会
**报错并给出两条出路**（改用 plain / 重新 prepare），
**不自动降级** —— 悄悄换模板会让用户以为自己在训 ChatML。

### 7.5.2 `SPECIAL_TOKENS` 只能追加，不能插队

词表 ID 就是 token 的身份。往中间插一个，所有既有语料和断点全部错位。
所以加新特殊 token 一律**追加到末尾**。

这条规则配一个自动检查：`BaseTokenizer.fingerprint()`（词表+合并规则的
哈希）会写进 `train.meta.json`，训练时核对。只比 `vocab_size` 和
分词器类型是**不够的** —— 老分词器 4 个特殊 token、新分词器 6 个，
两边都报 8192，可比对大小永远看不出问题，而每个汉字的 ID 都差了 2。
这种不匹配**不报错、loss 照降、输出全是乱码**，是全项目最危险的一类 bug。

### 7.6 诚实的能力预期（写进文档，别让人误解）

| 模型规模 | SFT 之后 |
|---|---|
| 13M | **学会问答的格式和口吻，但答不对事实** —— 会像模像样地编 |
| 100M | 简单问答能对，常识薄弱 |
| 1B+ | 需要像样的指令数据才能有实用价值 |

小模型 SFT 的价值是**看懂完整流程**，不是得到可用的助手。
这一点必须在 README 里说清楚，否则用户会以为训完就能当 ChatGPT 用。

### 7.7 GUI 的聊天界面

- 多轮对话（保持 KV cache，不重算历史）
- 流式输出（逐字显示）
- 可调温度 / top-k / top-p / 重复惩罚
- 显示生成速度（token/s）与显存占用
- **长 prompt 自动裁剪**：历史按 token 预算装（预算按 `block_size - 回答预留` 算）；system 永远保留，超长时从最近往前丢轮次并提示丢了几轮，单个问题太长则从开头截断 —— 不会因为 prompt 超 `block_size` 直接崩
- **非流式模式**：`lingyu chat --no-stream` 等整段生成完再输出（适合复制/接管道）
- **对话历史可导出为 SFT 训练数据**（闭环：聊出来的对话能反哺训练；计划中，P7）

### 7.8 P2 已实现（文件清单）

| 文件 | 作用 |
|---|---|
| `chat_template.py` | 模板渲染 + loss mask + JSONL 读取（四种格式）+ 数据集渲染统计 |
| `sft.py` | SFT 训练循环、`SFTDataset`（动态 padding）、扩词表、找基础断点 |
| `chat.py` | 交互式对话、流式输出、上下文按 token 裁剪、停止词 |
| `configs/sft.ini` | 出厂 SFT 配置（[model] 必须和预训练一致） |
| `chat/sample_chat.jsonl` | 50 条示例对话，用于验证流程 + 当格式模板 |
| CLI | `lingyu sft` / `lingyu chat` |

**三个实测踩到的坑（已修，有测试钉住）**：

1. **SFT 学习率写太大**：预训练那种 1e-3 会在几十步内把模型训崩
   （loss 变低但忘光预训练知识）。引擎在 >2e-3 时直接警告。
2. **流式输出把特殊 token 打给用户看**：模型吐出 `<|im_end|>` 时，
   原来只在 chatml 模板下才停，用 plain 就一路生成下去并把标记
   原样显示。现在**不分模板一律停**，且特殊 token 永不打印。
3. **验证集比 block_size 短时整个训练崩掉**：小语料 + 小 val_ratio
   切出来的 val.bin 常常不够一个窗口，`TokenDataset` 抛 ValueError，
   而调用处只 catch 了 `FileNotFoundError` —— 于是「验证集太小」
   把训练拦死了，报错还看不出问题在验证集。现在**降级并说明原因**。

---

## 8. 开源配套（用户说"看不懂，你自己看着办"）

| 组件 | 为什么必须做 | 优先级 |
|---|---|---|
| **一键 demo（`lingyu demo`）** | 别人 clone 后 30 秒看到效果，决定他会不会继续用。**这条决定项目生死**（计划中，P7，命令尚不存在） | 高 |
| **导出 HuggingFace 格式 + model card** | 不然模型只能在你的引擎里用，项目成孤岛 | 高 |
| **英文 README** | 门面，决定 star 数 | 高 |
| **CONTRIBUTING / CODE_OF_CONDUCT** | 别人敢提 PR 的前提 | 中 |
| **Dockerfile** | 云端/Linux 一键跑，企业用户最先看这个 | 中 |
| **插件开发文档** | 注册表的价值靠它兑现 | 中 |
| **benchmark / 榜单** | 吸引人跑分传播 | 低（后期） |
| **ANNOUNCEMENT / 训练小模型教程** | 中文社区传播 | 低（后期） |

---

## 9. 测试策略（用户要求"不能有 bug"）

当前测试共 **619 条**（32 个收集文件；`pytest tests -q`，其中含 5 条慢速端到端、GUI 84 条）。

| 层次 | 内容 | 何时跑 |
|---|---|---|
| **单元测试** | 配置校验/分词器往返/模型参数量/KV cache/断点存取/注册表/对话模板 | 每次改动 |
| **端到端回归** | 真跑：内置样本 → prepare → 训 65 步 → 生成 → 续训 → SFT → 对话 → 导出 | 每次改动 |
| **数值正确性** | 同一 seed 两次训练 loss 曲线完全一致；FSDP 单卡结果与 local 一致 | 关键改动 |
| **GUI 冒烟** | 启动 → 各页面切换 → 关窗，无异常 | 每次改动 |

**端到端回归**：`tests/test_cli.py`（`pytest -m slow`）。真跑
扫描 → prepare → 训练 12 步 → 生成 → 续训 → 结构变更诊断，
CPU 上十几秒跑完。`pytest -m "not slow"` 跳过它。

**配置完整性**：`tests/test_config_integrity.py`。

### 测试的写法原则（血泪总结）

1. **能用实测就别用公式**，尤其别在测试里手抄一份公式。
   手抄的公式和被测代码会一起错，测试就成了摆设。
2. **断言不变式，不要断言数字**。例如
   「`params` 报的总数 == `sum(p.numel())`」比「总数 == 12,587,904」有用得多：
   改了措辞不会假失败，算错了立刻红。
3. **每个测试对应一个真实踩过的坑**，并在 docstring 里写清坑的样子。
   没有故事的测试，后人不敢删也不敢改。
4. **端到端必须真跑**。子进程 + 真实文件 + 真训练，才验得到
   「入口点、退出码、编码、路径」这类只在真实调用里出现的问题。

### 已用测试钉死的 bug（不许复发）

| bug | 症状 | 钉它的测试 |
|---|---|---|
| 归一化参数量按 `2d` 算 | 屏幕长期挂着假的「公式与实测差 4,224」 | `test_param_counts_agree_with_reality` |
| `ln_f` 写死 `RMSNorm` | `norm=layernorm` 只生效一半 | `test_final_norm_follows_config` |
| 配置缺键抛 `KeyError` | 精简配置直接崩，看不出该补哪项 | `test_resolve_fills_every_spec_param` |
| 组件默认值填具体名 | `use_moe=true` 被静默推翻，MoE 档变稠密档 | `test_component_defaults_do_not_override_use_moe` |
| `--set` 自动纠正无提示 | 纠正发生了但用户看不到 | `test_set_alias_is_autocorrected` |
| `data_dir` 写成 `out_dir` | 训练找不到 train.bin | `test_shipped_config_data_dir_is_corpus_not_output` |
| 语料与分词器不配套 | **不报错、loss 照降、结果是乱码** | `check_data_tokenizer` + 4 条测试 |
| 带 BOM 的配置读不了 | 正确配置被拒，报「第 1 行缺少 = 号」 | `test_config_with_utf8_bom_is_readable` |
| 验证集短于 block_size | 整个训练崩掉，报错指向不到验证集 | e2e + train.py 降级处理 |
| 只有 user 的样本也补 EOS | 模型学到「被问了就别吭声」 | `test_sample_without_assistant_has_empty_mask` |
| 流式输出打印特殊 token | 用户屏幕上出现 `<\|im_end\|>` | `_stop_token_ids` + `_special_ids` |
| 扩词表忘了重绑 tie | 参数量悄悄翻倍 | `test_resize_vocab_keeps_tie_weights` |

> 语料/分词器不配套是全项目最危险的一类：token id 只是词表下标，
> 两次 prepare 的产物形状对得上但语义完全不同。所以引擎**直接拒绝开跑**。

---

## 10. 分阶段实现计划

| 阶段 | 内容 | 交付标志 | 状态 |
|---|---|---|---|
| **P1** | 核心库重构：registry + config spec + backends 骨架 + 标准包结构 | 现有 CLI 行为完全不变，测试全绿 | ✅ 完成 |
| **P2** | 对话链路：chat_template + chat_data + sft + chat 推理 | 13M 模型 SFT 后能按格式问答 | ✅ 完成 |
| **P3** | FSDP2 后端 + 断点版本兼容 + streaming 数据 | 多卡能跑（本机单卡验证不回归） | 🔶 仅适配层；**多卡训练/断点版本/streaming 均未实现** |

### P3 部分完成：后端适配层 + 规模估算（`backends/`）

`backends/plan.py` —— **全项目唯一的显存估算器**（`plan` 命令和
`config/validate.py` 都调它，不再各写一份公式）：

- 每参数 16 字节的来历：bf16 权重 2 + bf16 梯度 2 + AdamW 两个 fp32 动量 8
  + fp32 主权重 4
- ZeRO-1/2/3 与 CPU offload 的分片公式
- 激活值：`batch × seq × d_model × n_layer × factor`，开了重算 factor 从 22 降到 6
- **MoE 按总参数算显存、按激活参数算算力** —— 这是 MoE 的本质，不能含糊
- 拿不到显卡信息时**不给 verdict**（不编一个出来）
- **留余量**：可用显存先扣 15%，再要求 `估算 × 1.2` ≤ 可用才算够（见 §5 的估算口径）

`backends/launch.py` —— 启动脚本生成。三条规矩写进模块 docstring：

1. **诚实**：`backend_info()` 明确标出「已实现 / 只生成配置 / 不实现」
2. **算得出来**：超出本机必须给出「需要几张卡」这个具体数字
3. **不重写别人的东西**：DeepSpeed / Megatron 只做配置翻译 + 脚本生成

| 后端 | 产出 | 真实性 |
|---|---|---|
| `local` | — | 已实现 |
| `fsdp` | `launch_fsdp.sh`（torchrun） | **仅生成脚本，训练循环未实现** |
| `deepspeed` | `launch_deepspeed.sh` + `ds_config.json` | 仅生成脚本（用官方 DeepSpeed） |
| `megatron` | `launch_megatron.sh`（参考模板） | 引擎**不实现**，只翻译参数 |

新命令 `lingyu launch --config <配置>`：算规模 → 生成脚本 → 说明怎么用。

> ⚠️ `fsdp` 后端当前是**实验性**：`launch` 只生成启动脚本，`train` 仍只跑单卡（`local`）；多卡训练循环尚未实现。

> **修掉的一个假承诺**：`validate.py` 原来写着
> 「生成集群启动脚本：`lingyu export --backend deepspeed`」，
> 可那时候**根本没有 `export` 这个子命令**。用户照着敲只会得到
> 「不认识这个命令」，比不给建议更糟 —— 他会以为是自己弄错了。
> 现在指向真实存在的 `lingyu launch`，并且有测试钉住
> 「建议里提到的命令必须真的存在」。

**实测**：small 档（13.08M）估算 2.71 GB / 可用 6.87 GB（实测峰值 2.68 GB）→ 通过；
把 `n_layer=40, n_embd=2048` 后估算 63.46 GB → 诚实报出
「需要约 11 张同等显存的卡」并给出可执行建议。

**还没做**：FSDP2 真正的多卡训练循环（`parallel.backend=fsdp` 走通）、
断点格式版本兼容检查、streaming 数据加载。
| **P4** | 导出 HF + model card + i18n 基础 | 导出能被 transformers 加载 | |
| **P5** | PySide6 GUI（8 个页面，含语料准备与聊天） | 8GB 机器 2 秒内打开，能训完一次 | ✅ 完成 |

### P5 完成记录 —— GUI

```
gui/
  __init__.py        入口，写清「绝不在模块级 import torch」这条铁律
  app.py             主窗口：侧边栏 + 页面堆栈 + 状态栏
  theme.py           配色表 + 整份 QSS（两套主题共用语义色名）
  i18n.py            中英双语文案表
  context.py         全局状态 + 信号总线 + 后台环境探测
  widgets.py         可复用零件：Card / StatTile / Field / LogView / PathPicker…
  chart.py           手绘 loss 曲线（QPainter，不用 matplotlib）
  paramform.py       由 spec.py **自动生成**的参数表单
  runner.py          子进程管理（QProcess 流式读取 + 进度解析）
  shot.py            截图工具（开发时用来看长得对不对）
  pages/             十个页面，一页一个文件
```

十页：概览 / 语料准备 / 训练 / 监控 / 分析 / 断点 / 续写 / 微调 SFT / 聊天 / 设置。
启动命令 `lingyu gui`，快捷键 `Ctrl+1..9` 切前九页、`Ctrl+0` 帮助页。

**视觉层（2026-10 重构后）**：`gui/theme.py` 是唯一配色/间距/圆角来源，
深色为主、浅色可切；`gui/app.py` 顶部有 TopBar（面包屑 + 任务状态 +
主题/语言切换），侧栏分四组。帮助页内置 80 篇百科（`gui/help_content.py`
的教程 + `gui/help_articles_extra.py` 的百科），不联网，搜索支持
智能/精确/模糊三档。

#### 唯一的硬约束：GUI 进程里没有 torch

| 做法 | 效果 |
|---|---|
| 模块级 `import torch` | 窗口要等 3~8 秒才出现 |
| 在 `on_show()` 里 import | 窗口出来了但整界面冻结几秒，像卡死 |
| **后台线程探测**（现在的做法） | 窗口 0.06 秒出来，先显示「检测中…」，结果回来再填 |

实测：`MainWindow()` 构造 **0.064 秒**，此时 `torch` 尚未进入 `sys.modules`。
`tests/test_gui.py::test_gui_does_not_import_torch_at_module_level`
用一个独立子进程守住这条 —— 一旦有人图省事在模块级 import，测试立刻红。

所有要 torch 的活（训练/微调/采样/聊天）都通过 `lingyu` 命令在子进程里跑，
界面只负责显示子进程的输出。聊天用**常驻子进程**（模型只加载一次，
多轮历史由 `chat.py` 自己维护），不是每问一句就重启一次。

#### 监控页的数据从哪来（1.2）

监控页不猜数字。`train.py` / `sft.py` 在各自输出目录写 `run.json`
（`kind` = pretrain/sft、`max_steps`、`tokens_per_step`、参数量、设备等），
SFT 还写 `history.csv`（与预训练同列），所以两个任务的 tok/s、训练 loss、
验证 loss 各看各的，不会串。

- 数据源下拉框列出两个目录：训练页的 `train.out_dir` 与微调页的 `post.out_dir`；
  训练 / SFT 一开始跑就自动切到对应源。
- **速度用 `history.csv` 相邻两行的增量算**（`Δstep × tokens_per_step ÷ Δ秒`）。
  续训时 `elapsed` 会重置，直接拿末尾行除会算出上百万 tok/s 的假速度
  （1.2 截图审计抓到的真 bug）。
- 老目录没有 `run.json` 也能看：按配置估算 `tokens_per_step`，标注「(估算)」。
- SFT 的验证 loss 是「每次」口径，标签写明，避免和预训练按 token 加权混淆。

#### 训练看板的「真实统计」与分析页

训练页的运行看板不是把日志换个皮肤，数字都来自引擎上报的结构化事件
（`--json` 的 `step`）：已用时间用引擎的 `elapsed`（含 eval / 存盘时间），
预计剩余用 `(total-step) × tokens/步 ÷ 速度`，每步耗时 = `elapsed / step`，
epoch = `step × tokens/步 ÷ dataset_tokens`（后两个字段由 `train.py` 写进
`run.json` 和每个 `step` 事件）。演示用的假进度条已删除 —— 界面宁可显示
「—」，也不编数字。

分析页（`gui/pages/analysis.py`）回看跑完的记录：

- 数据源 = `out/` 下**所有**带 `history.csv` 的目录（`ctx.runs()`），不只是当前配置；
- 三张曲线：loss（train + val）、学习率、速度（相邻行增量，续训不重算）；
- 顶部 KPI：总步数 / 最好验证 loss（含步数）/ 末段训练 loss / 平均速度 /
  训练耗时 / 总 token；
- 「导出图表」把三张图拼成一张 PNG。分析页只读文件、不写任何东西。

#### 参数表单是自动生成的

`paramform.py` 读 `config/spec.py`，按类型选控件（int/float → 数字框、
bool → 勾选框、带 choices 的 str → 下拉框、路径 → 输入框 + 浏览按钮），
参数说明直接取 spec 的 `help_zh` / `help_en`，按 `tier` 分成
常用 / 进阶 / 专家三页。

**不手写表单的理由**：手写的话，spec 里加了参数界面忘了加，就出现
「文档说能改、配置文件里能改、界面里找不到」。改三次以后没人信这个界面。
`tests/test_gui.py::test_param_form_covers_every_spec_param` 拿 spec 对账，
一个都不能漏。

#### 视觉规范（写在 theme.py 顶部，页面代码必须遵守）

- 间距只有 4/8/12/16/24/32 几档，圆角只有 6/8/10 几档
- **颜色一律来自配色表**，控件里不许出现硬编码色值
- 两套主题共用同一份**语义色名**（bg / surface / text / accent…），
  所以切主题时页面代码一行都不用改
- `test_theme_has_all_semantic_colors` 保证两套主题的色名完全一致

#### 靠截图发现的三个真实 bug（都已修，有测试钉住）

写完之后用 `gui/shot.py` 截了图来看，一眼就发现三个**读代码发现不了**的问题：

1. **参数表单把学习率存成了 0**。小数位数按参数名硬编码
   （`learning_rate` 用 2 位），而 5e-4 在 2 位小数下就是 0.00 ——
   取回来是 0.0。后果是「打开训练页、什么都不改、点开始训练」，
   保存的学习率是 0，模型一步都学不动，而界面上看着完全正常。
   `norm_eps`（1e-6）同样中招。改成**按数值量级决定小数位**。
2. **标签被截断**。`Field` 的标签宽度写死 132px，
   「一个 token 最多几个字」被从左边截成「token 最多几个字」。
   改成按字体度量算宽度，并让整组表单**等宽对齐**。
3. **左右分栏被压扁**。语料页左边那一栏没包滚动区，
   内容一超过窗口高度，Qt 就把所有控件压到最小尺寸以下 ——
   标签和输入框互相重叠，整页像坏了。给每一栏都包上 `QScrollArea`。

> 教训：**界面的问题只有真的看一眼才能发现**。所以 `shot.py` 是开发流程的
> 一部分，不是可有可无的玩具。

#### 还没做

- 界面里的「英文文案」已备好（`gui/i18n.py` 的 EN 表），
  但英文下没逐页走过一遍
- 大模型规模下的界面（万亿参数）没实测过 —— 本机跑不动
- 打包成 .exe（属 P6）
| **P6** | 启动器 + 打包 + 离线包 | 新机器解压双击即用 | ✅ 启动器/打包完成；**离线包未做** |

### P6 完成记录 —— 打包成「一个文件夹 + 一个 .exe」

用户的原话：「最好做成一个文件夹里有一个 .exe 文件，什么配置文件和运行环境
都在子目录里放着，开箱即用」。

#### 为什么不能真的只有一个 exe

PyTorch + CUDA 有 2.5 GB 以上，塞进 PyInstaller 单文件包的话：

- 打出来的 exe 有 2.6 GB
- **每次启动都要把整包解压到临时目录**，开一次等半分钟

那不叫开箱即用。所以分成两层：

```
dist/LingYu/
  LingYu.exe        27.5 MB —— 启动器，双击这个
  使用说明.txt        UTF-8 with BOM（记事本才不会乱码）
  LICENSE / README*
  runtime/           ★首次双击后才出现
    uv.exe           从 exe 里解出来的，用来下载 Python
    python/          独立 Python（本机没有合适的才下载）
    venv/            装了 torch / numpy / PySide6
    app/             引擎源码（从 exe 内嵌的 zip 解出来）
    configs/         配置（用户可直接改）
    data/ out/ raw/ chat/
    launcher.log
    .installed.json  记录装的是哪一版
```

**删掉整个文件夹 = 卸载干净**，这是「所有东西都在子目录里」的实际价值。

#### 首次运行流程（`packaging/launcher.py`）

1. 找 Python：`runtime/python` → `LINGYU_PYTHON` → PATH → 常见安装位置
2. 都没有就用内置的 **uv** 下载一个（`uv python install 3.12`）
3. 建 venv（`python -m venv`，**不加 `--seed`**，见下）
4. 装依赖，镜像按 清华 → 阿里 → 官方 依次重试
5. 解包引擎源码、复制配置、写界面默认路径
6. 启动 `-m lingyu gui`

之后每次双击直接启动，这些步骤全部跳过。

**为什么 exe 里带 39.6 MB 的 uv**：用户机器上没有任何 Python 时也能一键装好。
不带的话得让用户先去 python.org 下载安装勾 PATH 再回来 —— 那就不是开箱即用了。

**启动器只用标准库 + tkinter**：不能依赖 PySide6，因为「装 PySide6」
正是它要做的事之一，先有鸡还是先有蛋。tkinter 是 CPython 自带的。

#### 实测踩到的四个坑（都已修，有测试钉住）

| bug | 症状 |
|---|---|
| `venv --seed` | `--seed` 是 **uv** 的参数，标准库 `venv` 不认，安装第一步就挂 |
| `✓` 撞上 GBK 控制台 | `UnicodeEncodeError`，**安装全部跑完却崩在最后一行进度输出上** |
| argparse 吃掉透传参数 | `LingYu.exe train --config X` 报 `unrecognized arguments` |
| `lingyu launch` 没做依赖检查 | 缺 torch 时吐一屏 `ModuleNotFoundError` 堆栈而不是人话 |

#### ★ 顺带修掉的一个「开源卫生」问题

`_require_deps` 里写死了开发者自己机器上的路径：

```
utils.info("       & \"C:\\path\\to\\lingyu\\lingyu.ps1\" %s" % cmd_name)
utils.info("  python -m lingyu prepare --raw C:\\path\\to\\lingyu\\raw ...")
```

别人 clone 下来缺依赖时，界面告诉他去跑一个**只存在于开发者机器上**的脚本。
打包版更严重 —— 那条路径压根不存在。

现在一律用 `sys.executable`（正在跑的解释器，永远正确）和
`_project_root()`（认得 `LINGYU_HOME`，所以打包版也算得对）现算。

`tests/test_packaging.py` 用一个 **ast 检查**守住这条：扫描所有
「真正会被执行的字符串常量」，发现 `llm-engine` 就失败。
用 ast 而不是正则扫行，是因为正则既抓不住不带引号的路径
（`--raw C:\path\to\corpus`），又会误伤**文档字符串里对这个坑的解释**。

这个检查立刻就又抓出两处遗漏：`sft.py` 的「没指定数据集」提示、
`train.py` 的 `--set` 示例。

#### 构建

```bash
python packaging/build_exe.py          # 产出 dist/LingYu/
python packaging/build_exe.py --zip    # 顺便压成可上传 Release 的 zip
```

源码表里刻意排除 `torch`/`numpy`/`PySide6`/`matplotlib`/`PIL`/`pytest`，
打出来的启动器只有 27.5 MB。

#### 0.3.1 重打包（2026-10-18）

热修后已重新打包并实测：

- `dist/LingYu/LingYu.exe`（含 Windows 版本资源：FileVersion/ProductVersion/CompanyName=`lingyu-lm`）；
  `dist/LingYu-win64.zip` 同目录；发布文件夹带 **SHA256SUMS**（5 个文件，含 exe 校验和）
- 打包版实测：`--version` → 「启动器 v0.3.1（引擎 v0.3.1）」；`--console doctor` exit 0；
  `plan --config runtime\configs\small.ini` 可用（首次启动后 runtime/app 自动更新为 0.3.1）
- 启动器新增 `--reset-configs`：升级后老用户的 `runtime\configs` 若与新版出厂配置不同，
  会主动提示；`--reset-configs` 恢复出厂版（旧文件先备份成 `.bak-时间戳`）
- `src/lingyu/version.py` 与 `packaging/launcher.py` 的 `LAUNCHER_VERSION` 已一起 bump 到 0.3.1

#### ★★ 用户实测撞上的大坑：装成了 CPU 版 PyTorch

**现象**：双击启动器装完，界面显示 `PyTorch 2.14.0+cpu`、
`CPU（没有可用显卡）`，训练和聊天全废。

**原因**：启动器把 torch 和 numpy 放在一个列表里，用普通 PyPI 镜像装。
而 **PyPI 上 Windows 的 torch 轮子就是 CPU-only 的** ——
CUDA 版在 PyTorch 自己的索引（`download.pytorch.org/whl/cuXXX`）里。

对一个「GPU 训练引擎」来说这是致命的：装出来的东西根本没有 GPU。

**修法**：

1. torch 从 `REQUIREMENTS` 里拿掉，单独走 `TORCH_INDEXES`
2. **装完必须验一遍**（`torch_cuda_ok()`）：真的跑一次
   `torch.cuda.is_available()`，不能只看「装上了没」
3. 如果发现环境里已经是 CPU 版，**自动重装成 CUDA 版**
   （这正是用户遇到的情况）
4. 全都不行才退回 CPU 版，并把修复命令原样打给用户（`CPU_FALLBACK_NOTE`）

#### ★★ 同一类错误的两个变种（第二轮实测）

> **教训一句话：`TORCH_INDEXES` 的「顺序」和「每项的用法」都不能想当然。**

**变种一：顺序错了 —— 「慢」不等于「会兜底」**

原来是「先官方 CDN（想当然认为最快最全），再国内镜像兜底」。
实测同一个 wheel：

| 源 | 速度 |
|---|---|
| 官方 `download.pytorch.org/whl/cu130` | **0.22 MB/s** |
| 官方 `download.pytorch.org/whl/cu128` | 0.27 MB/s |
| 阿里 `mirrors.aliyun.com/pytorch-wheels` | 0.57 MB/s |
| **上交 `mirror.sjtu.edu.cn/pytorch-wheels`** | **10.05 MB/s** |

1990 MB 的 torch：**官方 2.5 小时，上交 3.3 分钟。**

关键在于**兜底根本不会触发**：镜像只在**失败**时才被试到，而慢不是失败。
pip 会以 0.22 MB/s 一直爬，界面上「下载中…」卡着不动，用户以为程序死了
（实测 3 分钟写 0 字节）。

**变种二：用法错了 —— `index` 和 `links` 要按地址的真实形态填**

上交那条一开始写成 `"links"`。pip 用 `--find-links` 去扫，
**一条 `.whl` 都没扫到**，于是退回去用默认 PyPI 索引里的
**CPU 版 torch**（124 MB，无 `+cu130`）—— 变种一修好了，变种二又把
CPU 版放了进来，而且**不报错**。

实测两个镜像的形态完全不同：

| 地址 | 形态 | 正确用法 |
|---|---|---|
| 上交 `/pytorch-wheels/cu130` | PEP 503 索引（118 个子目录，0 条直接 `.whl`） | `index`（依赖齐全） |
| 阿里 `/pytorch-wheels/cu130` | 平铺目录（993 条直接 `.whl`） | `links` |
| 官方 `/whl/cu130` | PEP 503 索引 | `index` |

判断方法：`curl -s <地址> | head`，看有没有 `pypi:repository-version`。

**`lingyu doctor` 也改了**：以前只有一句「CUDA 不可用」，
但「装的是 CPU 版」（重装就好）和「驱动太老 / 卡不支持」（要更新驱动）
是**完全不同的两件事**。现在用 `torch.version.cuda is None` 区分，
CPU 版直接给出带 `sys.executable` 的完整修复命令。

#### ★ 另一个坑：`dist` 根下那个多余的 exe

PyInstaller 把 exe 放在 `dist/` 根下，构建脚本又复制一份进 `dist/LingYu/`。
**两个一模一样**，用户点了根目录那个 —— 它以 `dist/` 为安装根，
环境装到了 `dist/runtime/`，和 `dist/LingYu/runtime/` 完全两套。
用户以为在用发布版，排查时连日志都对不上。

修法：复制完**把根下那个删掉**；发布文件夹里遗留的 `runtime/`
（里面有本机绝对路径）也一并清掉。两条都有测试守着。

#### ★ GUI：把「用户自己点的停止」报成了「进程退出，码 1」

模型加载慢，用户又点了一下「加载模型」，代码 kill 掉上一个进程。
Windows 上被 kill 的退出码是 1 → 界面上出现两行「聊天进程退出，码 1」，
**看起来像崩溃**，用户以为引擎坏了。

修法：停止前打 `_stopping` 标记，回调先看标记再决定报不报错。

#### ★ GUI：子进程的真实报错被日志过滤器吃掉了

原来的过滤器只保留「以 `[` 开头」或含「错误」的行。
而 Python 的 `ModuleNotFoundError: No module named 'lingyu'`
**两条都不满足** —— 关键信息被静默丢弃，用户只看到一句「码 1」，
谁也没法据此排查。

修法：留一份输出尾巴（`_stderr_tail`），失败时原样打出来，
并在什么都没输出时明确说「多半是环境或路径问题」。

#### ★★ 用户第二次实测撞上的：238% 还说「够用」

**现象**（用户截图）：360M 参数、估算 16.37 GB、可用 6.87 GB，
结论写着 **「✓ 显存够用（约占 238%）」** —— 自相矛盾。

**根因**：估算函数收一个「鸭子类型」对象，内部
`free = c.vram_free() if hasattr(c, "vram_free") else 0`。
训练页传进去的是个临时小对象，**没有 `vram_free()`**：

```
free = 0  ->  per_card_bytes=None  ->  估算器不算 cards_needed
          ->  调用方 .get("cards_needed", 1) 拿到默认值 1
          ->  永远「够用」
```

而百分比用的是 `ctx.vram_free()`（真值），于是数字和结论对不上。

**修法（三条一起）**：

1. 签名改成 `estimate_model(cfg, free_bytes)` —— 显存**显式传入**，
   不再靠鸭子类型
2. **结论拿数字现算**（`_fits()`），不信任何「有默认值的字段」：
   字段一旦缺失就会被默认成「一切正常」，这正是 bug 的温床
3. 三态：`True` 够用 / `False` 装不下 / **`None` 显存还不知道**（不给结论）

**这条教训值得单独记**：`dict.get(key, 默认值)` 在「默认值恰好等于
『没问题』」时非常危险 —— 它会把「不知道」静默变成「正常」。

#### 同一次截图还暴露的两个显示问题

| 问题 | 原因 | 修法 |
|---|---|---|
| 「合计」和「可用显存」两行**叠在一起**，建议文字被裁 | 估算卡片内容高度会变（装不下时多出七八行建议），右栏没包滚动区 → Qt 把卡片压到最小尺寸以下 | 卡片加滚动区 |
| 但滚动区一加，**结论被滚到看不见的地方** —— 比压扁更糟 | 结论是这张卡片存在的理由 | 拆开：状态数字 + **结论固定在顶部**，只有「明细 + 建议」可滚 |

顺带修了 `StatTile` 的字号：原来按**字符数**分档，
「360.51 M」只有 8 个字符，在概览页的宽格子里正好，
到训练页的窄格子里就被挤成两行。改成按**实际像素宽度**挑字号，
并在 `resizeEvent` 里重算。

#### ★ 出厂配置里都是开发者机器上的绝对路径

`C:/path/to/lingyu/out/small` 这种 —— 别人 clone 下来或者用打包版，
这些目录**根本不存在**。

改法：仓库里的配置一律用**相对路径**（`out/small`、`data1`），
启动器复制到 `runtime/configs/` 时再**钉成 runtime 下的绝对路径**
（`_absolutize_config()`）。两边都干净：
仓库里没有「某台机器」的信息，用户拿到的配置直接可用。

测试侧现状：`test_config_integrity.py` 校验出厂配置能通过；`test_packaging.py` 用 ast 扫**源码里真正会被执行的字符串常量**（发现 `llm-engine` 就失败）。但**没有**专门的「扫出厂配置盘符」测试 —— 这条 P7 待补。

#### 新增：`gui/audit.py` 界面巡检

光看代码发现不了「238% 还说够用」这种问题 —— 那是截图里一眼看出来的。
所以加了个巡检脚本，把每个页面**在关键状态下**截图：

```bash
python gui/audit.py                 # 全部
python gui/audit.py --only train    # 只看训练页相关的几张
```

关键状态包括：大配置（该说装不下）/ 小配置（该说够用）/
**还没探测到显存**（不该给结论）/ 各页的空状态。
它还会把结论文字直接打到终端，方便不看图也能核对。

> **规矩：改完界面必须跑一遍巡检并挨个看图。**
> 用户原话：「这次你最好在找bug的时候截图挨个看」。

#### 还没做

- 只打了 Windows 版（Linux 用户走源码安装）
- 没做代码签名 —— 用户首次运行会看到 SmartScreen 提示
- 离线包（把 torch 的 wheel 一起打包）没做
- **「全新机器 + 2.5 GB CUDA 下载」仍未跑满过一次**：
  用户机器上已有 cu130 的 torch，实测走的是复用路径。
  cu130 索引本身用 `pip --dry-run` 验证过能解析出版本正确的 wheel。

---

| **P7** | 开源配套：英文 README / demo / Dockerfile / 插件文档 | GitHub 可发布 | |

**每阶段结束都必须**：全套测试通过 + 端到端回归通过 + 文档更新。

### P1 完成记录

- 标准包结构 `src/lingyu/`，`pip install -e .` 装好，命令 `lingyu` 全局可用
- `registry.py` 组件注册表（三层优先级），`components.py` 只做分发、不 import model
- `config/spec.py` 参数单一事实源，`validate.py` 类型/范围/候选/组合 + 规模感知
- 删除 `lingyu_old/`、`tools/`（生成器）、`config_legacy.py`（旧 4 段死代码）

### P2 完成记录

- `lingyu sft` / `lingyu chat` 两个子命令
- 对话模板三选一（plain / chatml / qwen），loss mask 用 `ignore_index=-1` 实现
- 分词器加密钥指纹（`fingerprint()`），语料/分词器不配套时**拒绝开跑**
- 出厂 `configs/sft.ini` + `chat/sample_chat.jsonl`（50 条示例）
- 端到端回归覆盖：预训练 → SFT → 聊天 → 结构不符时拒绝

---

## 9.5 配置系统的两条铁律（P1 血的教训）

### 铁律一：spec 里的默认值必须能被「补全」，不能有副作用

`config.resolve(cfg)` 在入口处把所有缺失参数补成 spec 默认值，这样下游
（train.py / sample.py）可以放心用下标取值 `cfg["weight_decay"]`，
不会因为用户写了精简配置就抛 `KeyError`。

**但默认值本身不能有语义副作用。** 反例：`model.ffn` 的默认值一度写成
`"swiglu"`，于是 `moe.ini`（只写了 `use_moe = true`、没写 `ffn`）
被补成 `ffn = "swiglu"`，解析逻辑看到 `ffn` 非空就按它走、顺手把
`use_moe` 改成 `false` —— **MoE 档静默变成稠密档**，参数量 53.5M 变 28.3M，
而且不报任何错。

所以「空字符串」在配置语言里统一表示**「引擎自己决定」**：

| 参数 | 空的意思 |
|---|---|
| `model.attention` | 按 `n_kv_head` 推断 MHA / GQA / MQA |
| `model.ffn` | 按 `use_moe` 决定 swiglu / moe |
| `model.norm` | rmsnorm |
| `model.pos` | rope |
| `post.method` | 不做后训练 |

校验器对空字符串一律放行（`_check_choices` 显式跳过 `""`）。

### 铁律二：失败要响，绝不静默降级

配置写错的每一种情况都必须**报错并给出正确写法**，不能"忽略这一项继续跑"：

- `--set` 键名写错 → 报错 + 拼写建议（曾经静默忽略）
- 常见笔误（`n_layers` / `lr` / `out.dir`）→ 自动纠正 **且必须提示**
  （`apply_overrides` 用 `on_notice` 回调，CLI 要记得接上，否则纠正了用户也不知道）
- 语料与分词器不配套 → **直接拒绝开跑**（见第 9 节最后一条）

---

## 9.6 路径只有一种算法（P7 血的教训）

用户实测报「找不到分词器和语料」，而**语料其实准备成功了**。
根因不是一个 bug，是四个各自都「看起来合理」的决定叠在一起：

| # | 决定 | 单独看 | 合起来 |
|---|---|---|---|
| 1 | 界面把默认配置存成相对路径 `configs/small.ini` | 可移植 | 解析时要挑一个根 |
| 2 | 界面按 `__file__` 往上数三层求根 → `runtime/app` | 开发版对 | 打包版指向**源码的家** |
| 3 | 出厂配置里 `small.ini` 独自写 `data_dir = data1`（其余五份都是 `data`） | 像笔误 | 被钉到 `<runtime>/data1` |
| 4 | 语料准备页的输出目录取自 `gui.json` 的 `data_dir` | 直观 | 写到 `<runtime>/data` |

结果：语料页刚说「准备完成」，训练页读配置说「还没准备语料」。
**两个目录都合法、都真实存在、代码里没有任何异常** —— 只是不是同一个。

### 铁律三：根目录只有一份实现

`src/lingyu/paths.py` 是唯一答案。GUI、子进程工作目录、命令行
全都调 `paths.home()`，谁都不许自己按 `__file__` 往上数。
`tests/test_paths.py` 用 ast 静态检查钉住这条（出现三层 `dirname` 链就红）。

### 铁律四：相对路径相对 `home()`，**绝不相对当前工作目录**

打包版的工作目录恰好是 `runtime/app`，而它下面**真的有一份**
`configs/small.ini`。任何「先试 cwd」的候选顺序都会稳定命中那份源码副本。
`paths.find_config()` 因此**刻意不把 cwd 放进候选**。

> 教训：`abspath()` 用 cwd 是几乎所有 Python 项目的默认写法，
> 而桌面应用的工作目录是**启动器决定的**，不是用户决定的。

### 铁律五：一个事实只能有一个主人

语料目录以前有两个主人（`gui.json` 和配置文件的 `[data] data_dir`），
谁也不知道对方改了。现在**配置文件是唯一主人**：

- 语料准备页的默认输出目录 = 配置里的 `data_dir`
- 准备成功后顺手把路径 `patch_config` 写回配置

### 铁律六：改值不要重写整份文件

`save_config()` 原本按 spec 重新生成整份 INI，把出厂注释**全抹掉**。
用户只是点一下按钮，回头打开配置文件就发现自己看不懂了。
现在 `patch_config()` **就地改值**，`save_config()` 默认也保注释。

### 顺带修掉：「升级 exe 拿不到修复」

`extract_app()` 原来只看 `.extracted` 标记文件在不在就跳过解包 ——
用户升级了 exe，跑的**还是上一版的源码**，修好的 bug 一个都到不了他手里，
而版本号看着是新的。现在标记里存**源码包的哈希**，变了就重解；
重解前只清 `src/gui/docs`，`plugins/` 和 `chat/` 里可能有用户的东西，**不删**。

---

## 11. 兼容性承诺（开源项目必须的）

- **配置文件向后兼容**：现有 `configs/*.ini` 永久可用
- **断点格式带版本号（待实现）**：`ckpt_format` / `lingyu_version` 常量已预留，但当前保存/加载尚未写入和校验；
  目标是**不兼容时明确报错，绝不静默加载出错的结果**（P3 剩余项）
- **CLI 向后兼容**：现有命令与参数不变，只增不改
- **失败要响**：任何降级/忽略都必须打印警告，绝不静默

---

## 12. 已知风险与对策

| 风险 | 对策 |
|---|---|
| 重构搞崩现有功能 | P1 阶段先建测试基线，重构前后逐项对比 |
| GUI 依赖拖慢启动 | GUI 进程绝不 import torch；重活全在子进程 |
| PySide6 打包 200MB+ | 只打核心 Qt 模块；不用的模块 exclude |
| FSDP 在小卡上没意义 | 后端自动判断：单卡时提示"用 local 更快" |
| 万亿参数吹过头 | 后端明确标注"需要 N 张卡"，永不假装能本地跑 |
| **对话模板训练/推理不一致** | 提供 `chat-template check` 子命令（**计划中，尚未实现**），把拼接结果原样打出来核对 |
| **SFT 全算 loss 导致模型自己生成问题** | loss mask 是默认行为，文档里重点说明 |
| 本地子 agent 失败率高 | 调研类任务自己做；子 agent 只派"自足、具体、可验收"的活 |
