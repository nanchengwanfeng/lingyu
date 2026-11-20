# 灵语引擎 · 项目状态与待办（2026-10-31 整理）

> 这份文件是「会话压缩后的权威状态快照」：把散落在多轮对话里的结论、实测数据、决策与未完成事项
> 收拢到一处。以后上下文被压缩或换新会话，先读这份，再读 `DESIGN.md` 与 `OPTIMIZATION.md`。
>
> 路径约定：引擎根 `C:\path\to\lingyu\`；本文档相对该根目录。

---

## 1. 一句话状态

灵语（LingYu）已从「能训练小模型」走到 **v0.4.0 可分发包 + 完整 GUI + 训练显存优化**：

- 13M～79M 五档配置可训；`train → sft → chat` 全链路可跑
- GUI 10 页（概览/语料/训练/监控/分析/断点/对话/续写/微调/聊天/帮助 共 11 项导航）
- 打包：`dist\LingYu\LingYu.exe` 单 exe 自解压安装器（29 MB）+ `runtime/` 运行时
- 测试：**619 条全绿**（`pytest tests/`，32 个收集文件）

---

## 2. 代码地图（改代码先看这里）

```
src/lingyu/
  model.py            LingYu 模型（RMSNorm/RoPE/GQA/MoE/SwiGLU + grad_checkpoint 分支）
  train.py            预训练循环、断点读写（含压缩容器）、run.json/history.csv 契约
  sft.py              SFT 循环、loss mask、history.csv/run.json
  optim.py            AdamW8bit（m 按 2048 块 absmax int8，v 保留 bf16）
  compression.py      断点无损压缩容器 LYCKPT1（zlib + 分块）
  devices.py          设备探测/描述（cuda/cpu/xpu…）
  backends/
    plan.py           显存估算唯一实现（含 grad_checkpoint / optim_bits 建议）
    launch.py         torchrun / deepspeed / megatron 脚本生成
  plugins.py          插件注册（注意力内核/损失/优化器）
  components.py       组件分发（刻意不 import model，破循环）
  config/{spec.py,loader.py,validate.py}   89+ 参数单一事实源、INI/JSON/YAML、--set
  prepare.py          语料扫描/体检/分词器训练；JSON 支持 raw/extract 两种模式
  chat_template.py    模板渲染 + loss mask + 4 种 JSONL 格式
  chat.py sft.py cli.py utils.py paths.py version.py
gui/                  PySide6；app.py 外壳 + TopBar/侧栏；pages/ 11 页；help_articles*.py 93 篇
packaging/            launcher.py（自解压安装器）、build_exe.py、slim.py（瘦身）
docs/                 DESIGN.md 权威设计、OPTIMIZATION*.md、TOKENIZER.md、KV_LONG_CONTEXT.md、本文
tests/                619 条，按模块分文件
configs/              small/medium/wide/moe/nano65/chat/chat_medium/sft 等出厂配置
external_data/        alpaca_gpt4_zh.jsonl（SFT 中文数据，31 MB）
```

运行环境：`venv\`（torch 2.14.0+cu130）、`dist\LingYu\runtime\`（打包版独立环境）。

---

## 3. 已完成的大块（时间线）

### 3.1 全面审计 + 三批热修（10-17～10-18）
- 6 路并行审计：GUI / 训练 / 对话-SFT / 配置-后端 / 打包 / 测试文档；
  审计报告是开发期内部材料，不随仓库发布；结论已并入本文、`CHANGELOG.md` 与 `docs/PROJECT_STATE.md`。
- 热修三批：SFT target 位移（P0）、BPE 抽样覆盖、断点续训检查、CLI 统一校验入口、
  优雅停止、GUI 任务总线、监控数据分离、JSON/MD 语料、打包启动器修复等。
- 审计结论：引擎「可跑、可交付」但细节粗糙；此后全部进入迭代。

### 3.2 聊天能力（10-18）
- 预训练：`configs/chat.ini`（13.08M）3000 步 → loss 8.78→3.47，best val 4.389；
- SFT：Alpaca-GPT4-zh（4.8 万条，过滤后）+ plain 模板，2.9k～4k 步，loss≈2.9；
  13M 回答质量弱（格式对、知识空），属容量天花板；
- 档位升级：`configs/chat_medium.ini`（47.80M）预训练 3000 步 + SFT 6000 步，
  val 2.62，质量仍受语料/规模限制；
- 结论写进 b15/b17 摘要与 `chat/README.md`：小模型学「怎么回答」，不是知识。

### 3.3 GUI 基线 → 重构（10-24～10-26，多批）
- 任务总线 + 滚轮误改参数 + 监控页预训练/SFT 数据分离（run.json 契约）；
- 训练页「运行看板」：真实 已用/每步/epoch/剩余 + loss/lr/速度 + 进度条；
- 新增**分析页**：回看任意 run 的 loss/学习率/速度曲线 + KPI（6000 步、2.6206、7m28s）；
- **帮助页**：93 篇文章 + 关键词/精确/模糊搜索 + 分组/热门标签筛选；
- **视觉重构**：新设计系统（紫蓝 accent、surface 四层、SPACE/RADIUS/FONT_SIZE 宏）、
  TopBar（面包屑 + 任务 pill + 主题/语言）、侧栏分组、切页淡入；
- 1.4 CLI 结构化：全部命令支持 `--json`（JSONL 事件），GUI 据此解析；
- 1.6 帮助页 + 1.5 参数 `?` 徽标（QSS tooltip）；
- 文档/CHANGELOG/BACKLOG 同步；exe 每次重打并 doctor 验证。

### 3.4 性能与体积优化（10-24～10-31，2.x）
| 项 | 结果 |
|---|---|
| 2.1 瘦身 | 引擎物理占用 12.15 GB → **约 5.6 GB**（硬链接去重 18,321 文件/3.31 GB；删旧 torch 轮子） |
| 2.2 断点无损压缩 | 573.8 MB → 493.6 MB（**1.162×**）；逐元素 bit-exact；**多线程并行：存 25.9→4.15 s、载 1.68→0.68 s**；可选 lzma（487.8 MB，存 17.6 s） |
| Round 2 评估提速 | eval 一次前向多块：47.8M 真实模型实测 20,480 token 0.324→0.122 s（**2.66×**）；与逐块评估数学等价。（此前 3.5× 来自一个 d_ff 配错、实际只有 15M 参数的误建模型，已纠正） |
| Round 3 加速（11-20） | fused AdamW +3.2%；训练 batch 预取 +0.8%（batch 更大/CPU 更慢时收益更大）；语料准备并行读 1.12×（4×480 MB 大文件，瓶颈在磁盘）；`accelerators.py` 统一 CPU 线程/TF32/fp32 精度；`LINGYU_COMPILE` 探测 Triton 后才启用（Windows 跳过）。详见 `docs/OPTIMIZATION_ROUND3.md` |
| Round 4 细项（11-20） | 取批向量化 50×（端到端噪声内）；生成改 `inference_mode`；SFT 验证漏 `no_grad` 已修（GPU 实测 8.78×、−3.9 GB 显存）；SFT 训练复用预取；分词器单字符快路径实测变慢 5%，已回滚。详见 `docs/OPTIMIZATION_ROUND4.md` |
| 2.3 可换内核 | 设备抽象 `devices.py` + 插件注册；`Attention._attend()` 为唯一 SDPA 出口；示例 `examples/plugins/attention_local_window.py`（局部窗口注意力）+ `docs/EXTENDING.md` §5 + 4 条测试 |
| 2.4 训练显存 | 梯度检查点 −64% 峰值；8bit 优化器 −3%；两者 −67%（见下表） |
| 2.7 学术帮助 | 帮助页新增「学术」13 篇（论文速查 42 篇入库） |

**2.4 实测（单卡 8GB 显存，medium 47.8M，batch 24×512，data1，40 步）**

| 配置 | 峰值显存 | 吞吐 | 相对 |
|---|---:|---:|---:|
| 基线 | 5.84 GB | 45,723 tok/s | 100% |
| `grad_checkpoint=true` | 2.12 GB | 37,599 tok/s | 82% |
| `optim_bits=8` | 5.64 GB | 45,653 tok/s | 99.9% |
| 两者 | 1.91 GB | 33,903 tok/s | 74% |

loss 40 步：7.0310（基线）/ 7.0315（重计算），属归约顺序差异。

---

## 4. 模型与实测基线

| 配置 | 总参数 | 训练速度 | 峰值显存 | 生成速度 |
|---|---:|---:|---:|---:|
| nano65 | 12.6 M | — | 1.55 GB | 351 tok/s |
| small（16 层×256） | 13.08 M | — | ~0.9 GB 估算 | — |
| medium | 47.8 M | 25,221 tok/s | 5.24 GB | — |
| moe | 53.5 M（激活 34.6 M） | 46,657 tok/s | 3.93 GB | 114 tok/s |
| wide | 79.4 M | 22,373 tok/s | 5.54 GB | — |

语料：`C:\path\to\corpus`（10 本书 + JSONL）、`data1`（789 万 token，BPE 8192）、
`_训练语料`（15 txt + 7 jsonl，3470 万 token，供最近全流程）。

---

## 5. 打包与环境

- 构建：`python packaging\build_exe.py --zip` → `dist\LingYu\` + `dist\LingYu-win64.zip`
- 单 exe 自解压：首次双击下载 Python+依赖（约 2.5 GB，5~20 分钟），断网可续装；
  安装到 exe 同级 `runtime\`；卸载=删文件夹（另删 `%USERPROFILE%\.lingyu`）
- 只拷 exe 到新设备可用（需 64 位 Win10/11、可写目录、≥8 GB 磁盘、首次联网、允许 SmartScreen）；
  断点/语料/模型不在 exe 内，要单独拷 `runtime\out|data|external_data`
- 复用环境：`LINGYU_VENV=<已有 venv>` 跳过安装
- 瘦身：`python packaging\slim.py`（硬链接去重；venv 升级后需重跑）
- 自检：`LingYu.exe --console doctor`（或源码 `lingyu doctor`）

---

## 6. 未完成 / 下一步优先级

**P0（用户已提但未做）**
1. **2.2 增量断点压缩**：现在只压单文件；多存档场景（ckpt+best+历史）预计还有 2~4× 空间。
   （Round 2 已把单文件存盘压到 4.15 s，增量/差分仍待做。）

**P1（路线图内，收益明确）**
3. **2.5 训练 I/O 与 offload**：DataLoader `pin_memory`/`non_blocking`、优化器状态 CPU offload；
   核显/纯 CPU 路径实测（目前只验证过 CUDA 与 CPU 冒烟）。
4. **2.6 跨厂商验证**：AMD ROCm / Intel XPU / Apple MPS 的 `devices.py` 分支只在代码层支持，未实卡验证。
5. **FSDP2 真训练循环**：目前只有适配层+脚本生成；多卡不可及，保持「诚实标注」。

**P2（论文已读，单卡 8GB 不划算，留路线图）**
6. MLA / FP8 训练 / 序列并行 / ZeRO-3 offload：等有更大显存或多卡再上。
7. HF 格式导出 + model card（P4 遗留）。
8. 代码签名（SmartScreen 提示）、Linux/mac 打包。

**GUI（视用户反馈继续）**
9. 视觉已重构一轮（TopBar/主题/分析页/帮助页）；若用户继续提「不好看」，优先从
   排版密度（卡片高度/留白）、字体字号层级、空状态与图表细节入手。
10. 帮助页内容仍可扩（当前 93 篇；可再补 20~30 篇实战排错/教程）。

---

## 7. 关键决策（别推翻，都有理由）

| 决策 | 选择 | 理由 |
|---|---|---|
| GUI 框架 | PySide6（LGPL） | PyQt6 是 GPL 会传染，企业不敢用 |
| 许可证 | Apache-2.0 | 带专利授权 |
| 打包 | 单 exe 启动器 + runtime/ | torch+CUDA 2.5 GB，塞进单文件每次启动解包半分钟 |
| 图表 | QPainter 手绘，不引 matplotlib | GUI 进程不许 import torch，启动 0.064 s |
| 语言切换 | 重建窗口（`_rebuild_for_lang`） | Qt i18n 运行中换语言易崩；重建最稳 |
| FSDP2 | 适配层 + 脚本生成，不自己实现 | 无多卡可验证，不做假承诺 |
| 分词器指纹 | train.meta.json + 开跑前核对 | 词表不匹配不报错、loss 照降、输出乱码 |
| JSON 语料 | 默认 raw 保留结构，`--json-extract` 抽取正文 | 用户要求「默认别动结构」 |
| 8bit 优化器 | m int8 + v bf16 混合 | v 全 int8 会把小动量量化成 0，实测训练崩 |
| 断点压缩 | zlib + 自描述头 LYCKPT1，默认开启 | 无损、可逆、无第三方依赖 |

---

## 8. 踩过的坑（高频复发，写代码前扫一眼）

1. **`python -m lingyu` 必须在引擎根目录**；激活 venv 不解决目录问题。
2. **GUI 进程里模块级 import torch**：窗口会等 3~8 秒；用后台线程探测（`_EnvProbe`）。
3. **QProcess 环境**：`processEnvironment()` 是新进程的空环境，不是父环境；要 `systemEnvironment()` 再合并。
4. **QT offscreen 加载不了 .ttc**（微软雅黑）→ 截图必须原生平台，否则中文豆腐块。
5. **pytest 缓存目录 WinError 5**：非致命，加 `-p no:cacheprovider` 或忽略。
6. **patch 脚本改文件先截断再校验**：曾把 prepare.py 写成 0 字节；改文件用 `edit` 工具或临时文件 + `os.replace` 原子落盘，改前备份。
7. **numpy 不支持 bfloat16 标量**：`np.asarray(bf16_tensor)` 报 `TypeError: Got unsupported ScalarType BFloat16`；用 `view(torch.int16)` 位级转换。
8. **PySide6 C++ 对象被 GC**：`self.scrollable(body)` 不赋值 → QScrollArea 销毁 → `RuntimeError: Internal C++ object already deleted`；必须 `self.xxx = ...` 持有。
9. **robocopy 从盘根复制会带上 Hidden+System 属性**：资源管理器看不到；`attrib -h -s` 清理。
10. **`dict.get(key, 默认值)` 默认值若是「没问题」**：会把「不知道」变成「正常」（曾报 238% 够用）；结论必须拿数字现算。
11. **torch 下载慢 ≠ 失败**：降级链对「慢」无效；国内镜像（上交 10 MB/s）要排前面。
12. **arXiv 摘要页抽取会串页**：抓 2110.02871 返回了 ClimateGAN 标题；引用前核对编号与主题。
13. **v 动量全 int8 会毁训练**：小 v 被量化成 0 → sqrt=0 → 单步参数跳 2.4。
14. **优化器状态张量必须建在参数所在 device**：默认 CPU 会在 CUDA 上报 device mismatch。

---

## 9. 验证命令速查

```powershell
# 源码侧
cd C:\path\to\lingyu
.\venv\Scripts\python.exe -m pytest tests -q          # 619 条
.\venv\Scripts\python.exe -m pytest tests\test_gui.py -q   # GUI 单测
.\venv\Scripts\python.exe gui\audit.py                # 16 张审计截图
.\venv\Scripts\python.exe -m lingyu doctor

# 打包侧
.\dist\LingYu\LingYu.exe --console doctor
.\dist\LingYu\LingYu.exe --console --version

# 训练（示例）
.\venv\Scripts\python.exe -m lingyu train --config configs\small.ini
.\venv\Scripts\python.exe -m lingyu sft  --config configs\sft.ini
.\venv\Scripts\python.exe -m lingyu chat --ckpt out\sft\best_sft.pt
```

---

## 10. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/DESIGN.md` | 权威设计：定位、决策、目录、注册表、配置、执行后端、GUI、后训练、测试、路线图 |
| `命令手册.md`（仓库根目录） | 全部 CLI 子命令与参数、FAQ、退出码 |
| `读我先看.md`（仓库根目录） | 从零训练教程（四步流水线 + 排错） |
| `docs/OPTIMIZATION.md` | 2.1～2.6 优化：实测表、论文取舍、坑 |
| `docs/PROJECT_STATE.md` | **本文**：压缩后的权威状态快照 |

---

## 11. 关于「上下文压缩」的现状（2026-10-31）

- 本次会话约 84 个压缩块：**~1.68M token 原始内容 → 42K 摘要**；当前可见上下文 ~221K token。
- `acp_status` 显示：**905 个 dead ref**（宿主侧压缩/重写后失效的消息号），少量「未覆盖历史」
  （约 99 条消息、~78K token）无法再折叠；另有 1 条历史冲突记录
  （2026-10-23 的 unannounced-rewrite，属宿主侧改写，非本 agent 产生）。
- 处理方式：以后**重要结论一律落盘**（本文与各 docs），不依赖对话摘要；
  需要细节时用 `search_context` / `decompress` 找回具体块，别凭摘要猜。

---

## 12. P0 开源准备（2026-10-25）

- **模型可插拔**：`[model] type = 名字`；实现 `ModelProtocol`（forward / generate /
  structural_fingerprint）即可替换内置模型；`build_model()` 负责实例化与报错。
  示例：`examples/models/tiny_lm.py` + `examples/configs/example_tiny.ini`；
  文档：`docs/EXTENDING.md`。
- **`estimate_param_count()`**：解析式参数量估算，不实例化模型。
  `lingyu plan/params` 超过 10 亿参数时直接用它，避免为「装不下的大配置」
  真建模型（实测曾吃掉 6.9 GB 内存并卡死）。
- **修复**：注意力 bias 参数量公式每层多算一个 `d`（bias=True 时）。
- **仓库卫生**：`.gitignore`、`CONTRIBUTING.md`、GitHub Actions（lint + CPU 测试）、
  `docs/PROJECT_STATE.md`；本机已装 git 并初始化仓库（默认分支 main）。
- **文档去设备化**：README / docs 不再出现开发者本人的硬件型号，实测统一写成
  「单卡 / 入门级显卡」。

### 下一步（P1）

1. `lingyu doctor` 输出机器可读的插件/模型清单；GUI 里显示已加载组件。
2. FSDP2 单机多卡最小的可运行验证（无卡时用 CPU gloo 跑通流程）。
3. 发布物：CHANGELOG 生成、版本号来源统一（`version.py` 单一真相）。
4. 贡献入口：README 顶部「5 分钟上手」、issue 模板。

## 13. 2026-11-06：分词器 v2 / KV int8 / 长上下文（P1 训练侧）

- 分词器：脚本感知预切分 + 片段内 BPE（`docs/TOKENIZER.md`）；跨标点/中英混合 token 归零，
  压缩比 1.34–1.38 → 1.30（8k 词表），代价是需重新 prepare。旧 tokenizer 走 legacy 编码。
- KV int8：`[model] kv_cache_dtype=int8`，KV 显存 53%、生成 −30%（bf16 155 tok/s → int8 106）；
  32k 上下文单序列 KV：bf16 160 MB → int8 85 MB（`docs/KV_LONG_CONTEXT.md`）。
- 长上下文训练：`[train] loss_chunk`（默认 8192）分块交叉熵，block1024/batch8 实测
  4.11→3.99 GB；与已有 `grad_checkpoint`、`optim_bits=8` 可叠加。
- 版本已 bump 到 0.4.0（2026-11-20 发布整备）；下一步候选：MLA/FP8/序列并行（需重训）、
  StreamingLLM 推理滑窗、16k 中文词表。
