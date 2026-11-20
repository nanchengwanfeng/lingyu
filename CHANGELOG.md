# Changelog

本文件记录用户可见的变化。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)，
版本号遵循语义化版本。项目采用「源码包 + Windows 启动器」分发，每个版本的条目按阶段汇总。

## [0.4.0] - 2026-11-20

首个面向开源发布的版本。相对 0.3.2 的主要变化：

- **可插拔模型契约**：`[model] type` + `build_model()` + `ModelProtocol`，
  第三方模型放 `plugins/` 即可，示例见 `examples/` 与 `docs/EXTENDING.md`。
- **分词器 v2**：按 Unicode 脚本预切分，跨标点/中英混排的「阴间 token」归零。
- **效率**：KV cache int8、分块交叉熵、梯度检查点、8-bit 优化器、
  评估/断点压缩并行、取批预取、预算显存与参数（不实例化巨型模型）。
- **GUI**：11 页 + 93 篇帮助百科 + 训练看板 / 分析页 / 断点续训选择。
- **发布准备**：设备与个人路径去标识、ruff 规则收口、CI、wheel 说明。
- **测试**：620 条。

## [0.4.0-dev] - 2026-11-20（第十三批：Round 4 细项）

### Changed（变化）

- **训练取批向量化**：`TokenDataset.get_batch` 改为一次 fancy indexing 取窗口，
  微基准每批 1.1 ms → 0.02 ms（约 50×）；端到端在噪声内（取批本非瓶颈）。
  新增正确性测试 `test_get_batch_vectorized_matches_windows`。
- **生成改 `torch.inference_mode()`**：防呆、禁止生成张量进入自动求导图。

- **SFT 验证关梯度**：`eval_sft_loss` 补 `torch.no_grad()`（此前每次验证都白建计算图，属 bug 级开销）。
  实测 GPU 验证 3 个 batch：2.875 s / 4,482 MiB → 0.327 s / 582 MiB（**8.78×，−3.9 GB**）；新增测试钉住。

- **SFT 训练批量预取**：`BatchPrefetcher` 支持自定义 `fetch` 回调，SFT 循环复用同一套预取/关闭逻辑（`LINGYU_PREFETCH=0` 仍可关）。

### 尝试后回滚

- 分词器「单字符快路径」实测反而慢约 5%（旧路径已有 piece 缓存），未合并；
  详见 `docs/OPTIMIZATION_ROUND4.md`。

## [未发布] - 2026-11-20（第十二批：Round 3 加速）

### Added（新增）

- **训练 batch 预取**：`data.BatchPrefetcher` 后台线程双缓冲预取 `(x, y)`，与 GPU 计算重叠；
  `LINGYU_PREFETCH=0` 可关。同 seed 下序列与同步取数逐项相等（有测试钉住）。
  实测 100 步 +0.8%（小模型时数据准备只占单步 5~8%；大 batch / 慢 CPU / 慢盘更明显）。
- **语料准备并行读文件**：`prepare.run_prepare` 的读取/解码/JSON 解析走线程池，
  `LINGYU_PREPARE_WORKERS` 可调（=1 强制串行）。实测 4×480 MB 大文件 4.97→4.43 s（1.12×，
  瓶颈在磁盘与解码）；产物与串行逐字符一致。
- **`src/lingyu/accelerators.py`**：统一 CPU 线程（默认 min(8, cpu_count)）、
  `set_float32_matmul_precision("high")`、CUDA TF32 + cuDNN benchmark；
  `LINGYU_NO_ACCEL=1` 全关，`LINGYU_CPU_THREADS` 可调。
- **`LINGYU_COMPILE=1`（默认关）**：探测到 Triton 才启用 `torch.compile`；
  Windows 无 Triton 会提示并跳过（强行 compile 曾让进程直接以退出码 1 死掉）。
- `docs/OPTIMIZATION_ROUND3.md`：本轮基准、取舍与复现命令。

### Changed（变更）

- 新增测试 8 条：`test_accelerators.py` 4、`test_data.py` 预取 2、`test_prepare.py` 并行读 2；
  全套 **616 条全绿**。
- **纠正 Round 2 的评估数字**：此前写的 3.5×（0.21→0.06 s）来自一个 `d_ff` 配错、
  实际只有 15M 参数的误建模型；47.8M 真实模型重测为 0.324→0.122 s（**2.66×**）。

## [未发布] - 2026-11-20（第十一批：Round 2 优化）

### Added（新增）

- **评估批量并行**：`data.eval_loss` 支持一次前向多块（`eval_batch`，默认按
  `min(8, 8192/(block_size+1))` 自适应）；短尾块用 target=-1 补齐、按有效 token 加权，
  与逐块评估数学等价。实测 20,480 token 评估 0.21 s → 0.06 s（**3.5×**）。
  > 2026-11-20 更正：该数字来自一个 `d_ff` 配错、实际只有 15M 参数的误建模型；
  > 47.8M 真实模型重测为 0.324→0.122 s（**2.66×**）。
- **断点压缩并行化**：`compression.py` 的字节平面压缩改多线程（zlib/lzma 释放 GIL），
  读档解压同样并行，容器格式与老断点完全兼容。实测 573.8 MB 断点存盘
  25.91 s → **4.15 s（6.2×）**、读档 1.68 s → **0.68 s（2.5×）**，体积不变且逐 bit 相等。
- **`LINGYU_PACK_METHOD=lzma`**（可选）：体积 494.3 → 487.8 MB（多省 1.3%），
  但存盘 17.6 s；只作为存储吃紧时的显式选项，不默认。
- **`LINGYU_PACK_THREADS`**：并行压缩线程数（默认 min(8, CPU 数)）。
- **注意力内核插件示例**：`model.py` 的 SDPA 调用收敛为 `Attention._attend()`
  单一出口；`examples/plugins/attention_local_window.py`（局部窗口注意力，约 40 行核心）
  + `examples/configs/local_window.ini` + `tests/test_plugin_attention.py`（4 条）
  + `docs/EXTENDING.md` §5「写一个注意力内核」。
- `docs/OPTIMIZATION_ROUND2.md`：本轮计划、论文取舍与验收记录。

### Changed（变更）

- `tests/test_data.py` +2 条（批量评估等价性、iters 上限不受批量影响）；
  `tests/test_compression.py` +3 条（并行字节一致、lzma 往返、v1 旧格式兼容）。
- 测试基线 **599 → 608 条**（31 个收集文件）。

### Fixed（修复）

- 无（本轮无行为破坏性改动；老断点、老分词器、老配置文件均保持可读）。

## [未发布] - 2026-11-06（第十批：分词器 v2 / KV int8 / 长上下文）

### Added（新增）

- **分词器 v2（脚本感知预切分）**：`src/lingyu/tokenizer.py` 先按 CJK / 拉丁 / 标点 / 空白
  切片段，片段内做 BPE、跨片段禁止合并；拉丁片段上限 24 字符、空白粘后词。实测跨标点
  token 180→0、中英混合 token 20→0，英文不再退化成字节。旧 tokenizer 按 meta 里的
  `pretokenize` 标记走 legacy 编码，旧断点仍可用；新分词器指纹不同，需重新 prepare。
  详见 `docs/TOKENIZER.md`。
- **`[model] kv_cache_dtype`**（auto/f32/bf16/int8）：int8 按 token/head 的 absmax 量化
  K/V（fp32 scale），KV 显存降到 bf16 的 53%；生成速度约 −30%（量化开销）。
- **`[train] loss_chunk`**（默认 8192）：分块交叉熵，logits/softmax 中间量不再随整批
  token 数膨胀；实测 4.11→3.99 GB（−3%，模型/词表越大收益越大）。
- `docs/KV_LONG_CONTEXT.md`：KV/长上下文论文取舍表与实测。

### Changed（变更）

- `kv_cache_bytes()` / `kv_cache_per_token()` 支持按 `kv_cache_dtype` 计算，plan 的显存估算
  与实际分配一致。
- 分词器训练把词表扩到 4k+ 时，`min_pair_count`、`max_token_chars` 的行为与 v1 一致（向后兼容）。

### Fixed（修复）

- 分词器 v2 首版漏了 `import unicodedata`（`char_kind` 依赖 Unicode 类别判断）。

## [未发布] - 2026-10-25（第九批：开源准备 P0）

### Added（新增）

- **[model] type**：模型现在是注册表里的一类组件，可用 `[model] type = 名字` 换成自定义实现。
  内置实现仍是 `lingyu`；自定义模型实现 `ModelProtocol`（forward / generate /
  structural_fingerprint）即可，`build_model()` 负责实例化与报错提示。
- **docs/EXTENDING.md**：模型契约、注册方式、示例与四条硬约束。
- **examples/models/tiny_lm.py** + `examples/configs/example_tiny.ini`：完整可跑的示例模型。
- **CONTRIBUTING.md** 与 GitHub Actions（lint + CPU 测试）。
- **docs/PROJECT_STATE.md**：项目当前状态、实测数据与待办路线图。
- **`estimate_param_count()`**：按结构解析式估算参数量，不需要实例化模型。

### Fixed（修复）

- **`lingyu plan` / `lingyu params` 不再为超大规模配置实例化模型**。以前 80 层 × 8192 维
  的配置会真的去建模型，实测吃掉 6.9 GB 内存并卡死测试；现在超过 10 亿参数就直接用
  解析式估算并在输出里标注「以下为结构估算值」。
- **注意力 bias 参数量公式**：旧公式每层多算一个 `d`（bias=True 时参数报表比实测多
  1 个 `d`/层）。`attn_param_count()` 与估算函数已同时修正。

### Changed（变更）

- 仓库文档去掉针对开发者本人设备的描述，实测数据统一表述为「单卡 / 入门级显卡」。

## [未发布] - 2026-10-24（第八批：训练显存优化 2.4）

### Added（新增）

- **[train] grad_checkpoint**：梯度检查点（激活重计算）。实测 medium 47.8M
  batch 24×512 峰值显存 **5.84 → 2.12 GB（−64%）**，吞吐 45.7k → 37.6k tok/s。
- **[train] optim_bits**：8-bit 优化器状态（一阶动量 int8 分块量化 + 二阶动量
  bf16）。47.8M 省 0.2 GB；模型越大省得越多。`configure_optimizer` 支持
  `optim_bits=8`，非法值直接报错。
- `lingyu plan` 的显存建议新增这两条开关（都是真的会被训练循环读的）。
- `src/lingyu/optim.py`：纯 PyTorch 的 `AdamW8bit`（约 100 行，CPU 可测）。
- 新增测试：`tests/test_optim.py` 4 条 + `test_model.py` 梯度检查点等价性 1 条。

### Notes（踩坑）

- 二阶动量不能直接 int8：小 v 被量化到 0 会让 sqrt(v) 为 0、单参数一步跳 2.4；
  改为 v 保留 bf16。
- 期间修正了 8-bit 优化器状态张量建在 CPU 上导致 CUDA 训练报
  `Expected all tensors to be on the same device` 的 bug。

## [未发布] - 2026-10-24（第七批：体积瘦身 / 断点无损压缩 / 跨厂商设备）

> 用户反馈：引擎 12 GB 太大、13M 权重占 100+ MB、希望无损压缩、架构可换、
> 降低训练开销、兼容非 NVIDIA 卡、继续扩充百科（学术向）。
> 本批新增 12 条测试（test_compression 7 + test_devices 5）。

### Added（新增）

- **断点无损压缩**（`src/lingyu/compression.py`）：字节平面 + zlib，自有容器
  `LYCKPT1`，实测 573.8 MB → 493.6 MB（**1.162×**），读回逐 bit 相等；
  `LINGYU_PACK_CKPT=0` 可关。`train/sft` 的断点保存/读取统一走它。
- **跨厂商设备探测**（`src/lingyu/devices.py`）：cuda（含 AMD ROCm）、mps、
  xpu、npu（torch_npu）、musa；`doctor` 新增「设备后端」段与 auto 选择。
- `packaging/slim.py`：runtime venv 与开发 venv 硬链接去重（3.31 GB）、
  可选删除离线 torch 轮子（1.99 GB）。
- 帮助百科 80 → **93 篇**，新增「学术」组 13 篇（公式 + 论文 arXiv 号）。
- `docs/OPTIMIZATION.md`：六项优化的实测数字、未做事项与论文依据。

### Changed（改动）

- 开发机引擎目录物理占用 12.15 GB → 约 **5.6 GB**（D 盘已用 −4.9 GB）。
- 说明：硬链接去重后，升级开发 venv 会影响打包版 —— 升级后重跑
  `python packaging/slim.py --apply` 即可。
## [未发布] - 2026-10-24（第六批：GUI 视觉重构 + 帮助百科扩充）

### Fixed（修复）

- **语料体检/准备的文件清单被杂文件淹没**：目录里混进几千个 md 时，
  逐文件日志刷屏，JSON/JSONL 的行（在顶部）被日志滚走。现在：
  `scan` 先打印「按类型」汇总（md N 个 / txt N 个 / jsonl N 个）、
  「读入/跳过」计数，并把 JSON/JSONL 文件**逐个列出、永不淹没**；
  `prepare` 同样先给类型汇总与 JSON 文件清单。日志明细改为
  前 10 个 + 每 500 个一行（scan 之前是全量打印）。


> 用户反馈：「百科知识远远不够多」「开始重构整个 GUI，充分利用前端审美」。
> 本批是**视觉与内容**更新，不改训练行为；测试基线保持 **556**。

### Added（新增）

- **新设计系统**（`gui/theme.py` 重写）：紫蓝主色 + 四层表面 + 语义色表，
  深/浅两套主题共用同一套色名；圆角/间距/字号全部走宏，页面代码不再出现色值。
- **顶栏（TopBar）**：面包屑（分组 / 页面）、当前任务状态、主题与语言切换；
  原来散落在设置页的切换动作现在在任何页面都能一键完成。
- **帮助百科 14 篇 → 80 篇**：新增概念（Transformer/注意力/RoPE/SwiGLU…）、
  参数（n_layer/n_embd/GQA/batch…）、训练技巧、排查（OOM/乱码/指纹不匹配…）、
  引擎用法（CLI/配置/GUI/插件）与术语表；帮助页加「热门标签」与「分组筛选」。
- 帮助页搜索保留三档：智能（分词）、精确（原文）、模糊（容错）。

### Changed（改动）

- 侧栏结构收敛为「工作台 / 训练 / 资产 / 其它」四组；帮助页固定在「其它」。
- 训练页右侧把估算、运行看板、启动卡分区固定，滚动只发生在数据区 ——
  按钮与结论永远可见。
- 卡片/输入/下拉/按钮/进度条/滚动条/表格/聊天气泡的样式统一在主题层，
  页面不再各自写样式。

## [未发布] - 2026-10-24（第五批：训练看板真实统计 + 训练分析页）

> 用户实测反馈：进度行/看板数字不像真的，训练看着像「假的」；要求补已用/预计/
> 每步耗时/进度%/epoch，并新增一个能回看跑过训练的「分析」页。
> 测试 549 → **556**（GUI 83 → 84）。

### Added（新增）

- **训练页运行看板补全**：已用时间改读引擎真实 `elapsed`（含验证/存盘），
  新增「每步耗时」「epoch 当前/整轮」、进度条百分比；第一次进度没有历史可比时
  用引擎上报的 `tok_s`，不再停在 0。
- **训练分析页（侧栏「分析」）**：列出 `out/` 下所有跑过的训练（按 `history.csv`
  识别），画 loss（含验证点）、学习率、速度三条曲线；KPI 总步数 / 最好验证 loss /
  末段训练 loss / 平均速度 / 训练耗时 / 总 token；支持导出三图拼成一张 PNG。
- `train.py` 的 `run.json` 与 `step` 事件补 `dataset_tokens`、`tokens_per_step`，
  epoch 进度据此精确计算。

### Changed（改动）

- 删除假进度演示数据（`_demo_*` 脚本与截图），界面不显示编造的数字。
- 快捷键扩展：`Ctrl+1..9` 切前九页，`Ctrl+0` 打开帮助页。
- 测试基线 **556 条**（23 个收集文件）：新增 `tests/test_analysis.py` 6 条 +
  GUI 看板 1 条。

## [未发布] - 2026-10-24（第三批：CLI 结构化 + 帮助页 + 视觉/显存）

> 按用户「继续按计划」：先 1.4（CLI 结构化输出），再 1.6（帮助页），顺手统一视觉；
> 同时修打包版实测反馈的「JSON 抽不出正文」「显存测不准 / 重新检测无效」。
> 测试 526 → **549**（GUI 79 → 83）。用户要求本轮不跑重负载测试，`-m slow` 未跑。

### Added（新增）

- **1.4 收尾（全命令）**：`doctor / scan / prepare / params / plan / launch / chat` 也支持 `--json`；JSON 模式下人类日志全部走 stderr，stdout 只剩一行一个的结构化事件。
- **1.4 CLI 结构化输出（train/sft/sample）**：`train / sft / sample` 加 `--json`，stdout 输出逐行 JSONL
  事件（`start/step/eval/ckpt/sample/done/error`）；GUI 的 Runner 直接消费事件，
  原正则解析保留作兜底；`cmd_train / cmd_sft` 包一层异常，失败也会发 `event:error`。
- **1.6 帮助页**：侧栏新增「帮助」（`nav.help`）。首页 14 篇文章（从零训练 / 关键参数 /
  语料准备 / SFT / CLI / 常见报错 / 术语表），顶部搜索栏支持智能、精确、模糊三种模式；
  内容全部内置在 `gui/help_content.py`，不联网。
- **1.1 视觉第一刀**：调色板整体提亮一档；卡片改极淡竖向渐变（`surface_hi/surface_lo`）；
  主按钮渐变、侧栏导航胶囊化、输入框/下拉统一 8px 圆角；训练页右栏从 5:4 改 3:2，
  估算卡 + 看板放进可滚动舞台，窗口矮时不再把数字压到互相叠住。

### Fixed（修复）

- **JSON 语料「抽不出正文」**：顶层同时有 `name/version` 与 `content` 的 JSON 不再被
  误判成配置整份丢弃；只有按文件名能确定的真配置（`package.json` / `mcp.json` …）才跳过；
  没有正文字段的 JSON 回退按原文收录，体检日志不再打误导性的「跳过」。
- **显存检测慢 / 「重新检测」没反应**：环境探针先用 `nvidia-smi` 拿数字（0.1–0.2s），
  torch 再补设备详情；实测点一次 0.15s 返回，游戏占用后能反映真实可用显存。
- **帮助页/训练页布局**：估算卡最小高度 + 右栏滚动舞台，窗口变矮不再叠字。

### Changed（变更）

- 新增 `utils.emit_event` 作为唯一事件出口；JSON 模式下 `utils.info/ok/warn` 等人类日志改走 stderr，保证 stdout 可被逐行解析。
- GUI 三个运行任务（train / sft / sample）统一带 `--json`；`gui/runner.py` 增加
  JSON 事件解析（`_handle_event`），人类可读日志由事件字段拼装。

## [未发布] - 2026-10-24（第二批：实测反馈修复）

> 用户拿打包版实测后报的 9 个问题（记录见 `docs/PROJECT_STATE.md`），本批修完
> B-01…B-09、B-11；B-10（进度行口径）留下批。测试 515 → **526**（GUI 73 → 79）。

### Fixed（修复）

- **「?」悬停无帮助**：`HelpBadge` 改 `enterEvent` 显式 `QToolTip.showText()`，
  离开自动收起；`QToolTip` 两套主题补 QSS。
- **「续训」盲训 / 不能选断点**：训练页新增「续训起点」下拉（自动 + 各 run/断点页全部
  断点）；跨结构/跨 run 自动 `--set model.*` 对齐，跨类型改用新引擎参数 `--init-from`；
  按钮上方常显实际命令。**SFT 断点可直接当预训练起点**（回答 B-03）。
- **必须选配置文件**：配置下拉末尾加「自定义（默认参数）」；新增「另存为配置」把当前
  参数存成新 ini。
- **JSON 语料抽不出内容**：`extract_json_text` 改通用递归（深层 `meta.rows[*]` 也能抽）；
  `mcp.json / package.json / tsconfig.json` 等配置文件按文件名+顶层键整份跳过；
  `prepare`/`scan` 日志限行（前 10 + 每 500 个一行 + 跳过汇总）。
- **切英文不生效**：`app._rebuild_for_lang` 先 `set_lang()` 再建窗口。
- **断点页没有 SFT 断点**：「来源」下拉 = 全部 / 当前项目，行前缀「预训练 ·」「SFT ·」。
- **选中断点→续训报结构不一致**：断点页把断点路径+结构带回训练页，自动对齐后启动。
- **可用显存显示偏大**：概览页加「刷新」+「检测于 HH:MM:SS」，`ctx.refresh_env()` 重探。
- **断点元数据**：新断点写同名 `.json` sidecar（格式/step/best_val/结构/分词器指纹/配置
  路径），断点页优先读 sidecar，旧断点回退异步加载。

## [未发布] - 2026-10-24

> GUI 大改第一刀：把「界面」从「终端转储」变成可读的运行面板，并把参数说明
> 收进 `?` 徽标。改动集中在训练页 + 全局控件/侧栏；不改引擎行为。
> 测试 509 → 515（GUI 67 → 73），全套 `pytest tests/` 通过。

### Added（新增）

- **参数说明 `?` 徽标**：每个可调参数后面一个灰色小圆圈问号，悬停显示详细说明
  （富文本、显式宽度，Qt 纯文本 tooltip 不换行）。训练页的 spec 参数自动取
  `help_zh/help_en`，SFT 页的 8 个手写字段补 `help.sft.*` 文案。常驻灰色说明行
  不再重复同一段话。
- **训练页「运行看板」**：训练中显示「状态 + 进度条 + 步数 / 当前 loss / 学习率 /
  速度」四块数字，原始日志收进可折叠的「详细日志」（默认收起，只有点开才渲染）。
  数据走 `ctx.task_progress` 任务总线，页面重建后仍能收到；速度用相邻 progress
  的增量算，续训重置 `elapsed` 也不会出现假速度。
- **侧栏品牌区与版本号**：品牌加色块，下面直接显示当前版本（`0.3.2`，从
  `lingyu.version` 读）；当前导航项左侧 2px 强调条。
- 需求与分批计划登记在 `docs/PROJECT_STATE.md`（内部讨论文档不随仓库发布）。

### Changed（变更）

- 切页加 140ms 淡入（`gui/app.py`），可用 `LINGYU_GUI_NO_ANIM=1` 关掉
  （测试与 `gui/audit.py` 截图用）；不做飞入/缩放这类打断阅读的动效。
- `StatTile` 新增 `compact` 档（训练页右栏在 1440×900 下只有约 250 逻辑像素，
  20pt 数字会把格子撑高、把估算卡挤扁）；单位放到数字下方、标题最后一行。
- 测试总数 509 → 515，GUI 67 → 73（README / DESIGN / 命令手册 / 读我先看同步）。

## [未发布] - 2026-10-23

> GUI 全面改进的两批地基：第一批修用户当场报的滚轮 bug、审计 1.7 的
> 「停止按钮挂掉」根因，并顺手修掉一个只在测试里暴露的进程崩溃；
> 第二批做审计 1.2 —— 监控页把「预训练」与「微调 SFT」的数据分开统计。
> 测试 499 → 509（GUI 61 → 67），全套 `pytest tests/` 通过。

### Added（新增）

- **监控页把「预训练」与「微调 SFT」的数据分开统计（审计 1.2）**：以前两个任务
  共用同一套步骤 / loss / 速度数字，切任务会串。现在引擎在输出目录写
  `run.json`（`kind`、`max_steps`、`tokens_per_step`、参数量、设备等），
  SFT 也写 `history.csv`（与预训练同列）。GUI 监控页多了一个**数据源下拉框**
  （「预训练 · 目录」/「微调 SFT · 目录」），两个任务的 tok/s、训练 loss、
  验证 loss 各看各的；训练 / SFT 一开始跑就自动切到对应源。
  SFT 的验证 loss 标签改成「验证 loss（每次）」，避免与预训练按 token 加权
  的口径混淆。
- **速度改用 `history.csv` 相邻行增量计算**（`Δstep × tokens_per_step ÷ Δ秒`），
  续训后不再把 `elapsed` 重置算成上百万 tok/s 的假速度；老目录读不到
  `run.json` 时按配置估算并标「(估算)」。

### Fixed（修复）

- **滚轮路过输入框会改掉训练参数**（用户上报）：以前鼠标停在数字框/下拉框上
  滚页面，滚轮事件被控件吃掉、参数被悄悄改掉。现在 `gui/widgets.py` 给每个
  数字框和下拉框装了 `_WheelGuard`：**没有键盘焦点时滚轮一律转交外层滚动区**，
  点进输入框后才用滚轮微调（`guard_wheel()`；焦点策略设 `StrongFocus`，
  避免 Qt 的 `WheelFocus` 又靠滚轮抢焦点）。
- **窗口重建后「停止」按钮消失、训练停不下来**（审计 1.7 根因）：任务状态原来
  只挂在页面实例的 `Runner` 上，换语言/主题会重建页面，新页面拿不到旧
  `Runner` —— 表现是按钮回到「开始训练」，后台任务却还在跑。
  现在 `ctx` 增加**任务总线**：`attach_runner()` / `stop_current_task()` +
  信号 `task_line / task_progress / task_validation`；训练、微调、续写三页
  都从总线收输出和停止请求。切回页面会恢复「停止」按钮；关窗口时也会停掉
  重建前启动的任务（`app.closeEvent` 兜底）。新增 3 条回归测试。
- **GUI 测试套件退出码 `0xC0000005`（-1073741819）**：窗口构造 50ms 后会在
  后台线程 `import torch` 探测显卡，要 3~8 秒；pytest 常在这之前就退出，
  Qt 在解释器收尾阶段回调会直接让进程崩掉（测试全绿、退出码却是崩溃码）。
  现在测试里不自动起探测（`LINGYU_GUI_NO_PROBE=1`），并在
  `app.aboutToQuit` 里调 `ctx.shutdown()`（`QThreadPool.waitForDone(5000)`）；
  探查线程 `emit` 也捕获「信号对象已删」的 `RuntimeError`。

### Changed（变更）

- 测试总数 499 → 509，GUI 61 → 67（README / DESIGN / 命令手册 / 读我先看同步）。

## [0.3.2]
- **修复：打包版 GUI 里所有需要启动引擎子进程的页面**（聊天页「加载模型」、训练/微调/语料准备走的那条路）—— 子进程环境原来从**空的** `processEnvironment()` 起底，insert UTF-8 变量后等于替换掉整个父环境，启动器设的 `PYTHONPATH` 丢失，报 `No module named lingyu`。现在统一用 `gui.runner.child_environment()`（`systemEnvironment()` + 覆盖 `os.environ`），并加了 2 条回归测试。
 - 2026-10-18

> 0.3.2 把「聊天能力」变成开箱可复现的一条流水线：
> 一份配置跑完 预训练 → SFT → chat，并附带中文指令数据的一键获取脚本。

### Added（新增）

- **聊天流水线配置**：`configs/chat.ini`（13M，流程验证档）与
  `configs/chat_medium.ini`（47.8M，真正能聊出内容的档位），
  都是「train → sft → chat」三条命令走完，SFT 数据默认读
  `external_data/alpaca_gpt4_zh.jsonl`（中文 `plain` 模板，不依赖特殊 token）。
- **语料准备支持 JSON**：`.json` / `.jsonl` / `.ndjson` 默认**原样进语料**，
  花括号、键名、缩进全部保留（适合教模型学 JSON 格式）；加 `--json-extract`
  （GUI 语料页勾「抽 JSON 正文」）才只取 `text` / `content` / `instruction` /
  `output` / `messages` 等字段并跳过 id、url 元数据。`.md` 原来就支持。
- **`chat/fetch_sft_data.py`**：纯标准库从 ModelScope 下载
  `AI-ModelScope/alpaca-gpt4-data-zh`（`CC BY-NC 4.0`，仅限非商用），
  过滤超长样本后转成引擎的 JSONL；支持 `--skip-download / --keep-csv / --max-chars / --seed`。
- `chat/README.md` 增加数据集获取与转换说明。
- **GUI 训练页「配置文件」下拉框**：以前界面只认 `gui.json` 里钉死的那一份配置，
  换 `chat.ini` / `chat_medium.ini` 得手改 JSON；现在训练页顶部可直接切换，
  切换后表单与显存估算立即刷新，并写回 `gui.json`（任务运行中禁止切换）。

### Changed（变更）

- `configs/chat.ini` 的 `[data] data_dir` 统一为 `data`（与其余出厂配置一致，
  符合「配置里一个事实只能有一个主人」）；老实验目录 `data1/` 用
  `--set data.data_dir=data1` 覆盖。README / 命令手册里的同类说明同步更新。
- 引擎版本 0.3.1 → 0.3.2（启动器版本同号）。

### 实测（13M 小档，单卡 8GB 显存）

- 预训练 3000 步 / 263.2 秒，loss 8.7782 → 3.4669，最好验证 loss 4.3891（ppl 80.57），
  峰值显存 2.68 GB，断点 `out/chat_pretrain/{best,ckpt}.pt`。
- SFT 4000 步 / 170.5 秒（续训两段：1500→2000→4000），最后 20 步平均 loss 2.8827，
  最好验证 loss 2.9049，断点 `out/chat_sft/{best_sft,ckpt_sft}.pt`。
- 13M 的效果边界如实说明：**格式学会了（能按中文问答回话），内容仍会跑偏**；
  要像样的回答请用 `configs/chat_medium.ini`（或更大档位 + 更多语料）。

## [0.3.1] - 2026-10-18

> 0.3.1 是一次**热修**：专治「看起来正常、实际在错」的问题 ——
> SFT 学错目标、验证 loss 口径错、BPE 跨步抽样、CLI 完全不校验配置、
> chat 长 prompt 生成 0 token。同时把文档里与代码不符的承诺全部改成实话。
> 10-18 收尾：SFT 断点续训、显存估算实测标定、`pos`/`vocab_size` 从装饰品变真开关。
> 全套测试 484 passed（19 个收集文件；本版本累计新增约 124 条回归测试）。

### Added（新增）

- **优雅停止**：训练 / SFT 每步轮询环境变量 `LINGYU_STOP_FILE` 指向的文件，
  收到停止请求就在当前这一步结束时保存最终断点、正常退出；
  GUI 的「停止」按钮走同一协议（先请求保存，4 秒未退出才兜底强杀，且先校验目标进程身份）。
  实测 SFT 第 60 步停止成功写盘。
- **CLI 统一配置校验**：所有子命令载入配置后走 `_validate_cfg()`（`--set` 覆盖后重新校验），
  非法值退出码 2、人话报错；`validate.py` 补 31 处硬限制。
- **新测试**：`test_chat.py`(22)、`test_data.py`(4)、`test_model.py`(8)、`test_prepare.py`(3)、
  `test_train.py`(5)；GUI 回归扩到 53 条。
- `plan` 显存估算按实测标定（激活系数 29 + 0.7 GB 固定开销 + MoE ×1.4），
  实测 5 份配置误差 −7%~+1%；判定改为 `usable = 可用显存 × 0.92`、`估算 × 1.03 ≤ usable`。
- **SFT 断点续训**：`lingyu sft --resume [路径]`（不给路径用 `post.out_dir/ckpt_sft.pt`），
  校验结构/分词器指纹后恢复模型、优化器动量、步数、best_val 与 RNG 状态。
- `[model] pos` 成为真开关：`rope / learned / none` 真正决定用哪种位置编码
  （以前写了也没用，仍是 RoPE）；`use_rope` 与 `use_learned_pos` 同时为 true 会报错。
- `[model] vocab_size` 不再被静默覆盖：写成 0（默认）跟随分词器，写别的值但和分词器
  不一致时明确报错，而不是假装生效。
- GUI 概览页的显存估算改为复用 `backends.plan` 与 `validate.estimate_params`
  （删掉自带的第三份公式和重复代码），不再和 `lingyu plan` 给出两个数字。
- **发布物**：exe 加 Windows 版本资源（文件版本/产品版本/公司名），发布文件夹带 `SHA256SUMS`；
  启动器 `--version` 同时打印启动器与引擎版本。
- **`--reset-configs`**：升级后若 `runtime\configs` 与新版出厂配置不同会主动提示，
  该参数可恢复出厂配置（旧文件先备份 `.bak-时间戳`）。
- qwen 模板无 system 时补 Qwen 官方默认 system 提示。
- 新增本 CHANGELOG（`pyproject.toml` 的 Changelog 链接此前指向不存在的文件）。

### Fixed（修复）

- **SFT targets 未左移**：原实现 `tgt[i] = ids[i]`，让模型学「复制当前 token」的恒等映射。
  改为 next-token `tgt[i] = ids[i+1]`（仅回答位置参与 loss），并加位移回归测试。
- **验证 loss 口径错**：上下文只取 `batch_size`、每轮固定 token、评估后无条件切回训练态。
  改为 `block_size` 整段滑窗 + token 加权，返回 `(loss, tokens)`，BPB 用真实验证 token 数。
- **BPE 跨步抽样**：`corpus[::step]` 把原文相隔很远的字拼成邻居（压缩比 1.160 → 连续块 1.438）。
  改为 16 个等距连续块。
- **chat 长 prompt 生成 0 token / `--no-stream` 不打印回答**：预算改用真实 `block_size`，
  保留 system + 最后一问，按轮次裁剪并提示；`prompt >= block_size` 时不再直接 break，
  用滚动窗口重建 cache；非流式分支正常打印。
- **CLI 不校验配置**：`--set model.n_head=0` 曾 `ZeroDivisionError`、`n_layer=-5` 能建负层数模型。
- **GUI**：点停止被谎报成「进程退出，码 1」；切语言误杀训练；续训忽略所选断点。
- **`rope_base`** 配了但没传给 `RotaryEmbedding`；`lr_decay_iters=0` 没按 spec 跟随 `max_steps`。
- **FSDP 假实现**：`backends.IMPLEMENTED` 改为 `("local",)`，文案「真的能跑」→「实验性：仅生成脚本，
  引擎尚未实现分布式训练」；测试改为断言 fsdp **不在** IMPLEMENTED。
- **文档假承诺真值化**：FSDP2 多卡 / `lingyu export` / `lingyu demo` / `chat-template check` /
  离线安装包 / 断点格式版本兼容，全部标「计划中 / 未实现」并注明命令不存在；
  测试数、五档参数、旧目录结构（v0.2 平铺包、生成器）按实际更新。
- **打包启动器**：子进程工作目录、`--help` 乱码、子进程失败不可见
  （stderr 落 `runtime\logs\engine-<date>.log` 并打印尾部）、pip 300 秒空闲看门狗。

### Changed（变更）

- GUI 停止按钮语义：从「强杀」改为「请求保存后退出」。
- `lingyu launch` 对 fsdp 打三条实验性警告；`--help` 去掉「真的能跑」。
- 出厂配置一律用相对路径（`small.ini` 的 `data_dir` 统一为 `data`），
  打包版由启动器钉成 `runtime` 下的绝对路径。
- Windows 启动器重新打包并实测：`dist/LingYu/LingYu.exe` = 28,891,719 B（zip 28,597,466 B）；
  打包版里 `plan --config runtime\configs\small.ini` 可用。

### 未实现（如实列出）

- FSDP2 真正的多卡训练循环（`fully_shard`）；HF 导出与 model card；`lingyu demo`；
  `chat-template check`；断点格式版本兼容检查；streaming 数据加载；离线安装包。
- 已知性能遗留：KV cache 预分配、chunked prefill 掩码、批内重复惩罚 / EOS 多行处理、张量并行。

## [0.3.0] - 2026-09

初版 v0.3 架构（P1–P6）：`src/lingyu` 标准包化、7 段配置系统 + 组件注册表、
SFT / chat 链路、执行后端估算与启动脚本生成、PySide6 九页 GUI、Windows 启动器打包。
