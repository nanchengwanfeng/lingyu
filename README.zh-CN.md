# 灵语（LingYu）

**一个引擎，从 13M 到万亿参数。**

中文优先 · 对新手友好 · 对规模诚实 · 开源到可以放心用

[English](README.md) · [设计文档](docs/DESIGN.md) · [命令手册](命令手册.md) · [新手上路](读我先看.md)

> **诚实说明（当前能力边界）**：多机多卡**训练**尚未实现（P3 剩余项）；
> `lingyu launch` 只负责生成启动脚本，`[parallel]` 里的 fsdp/deepspeed/megatron
> 后端目前**不会真正跑分布式训练**。`lingyu export` / `lingyu demo` /
> `chat-template check` 属于**计划中**（P4/P7），命令尚不存在。

---

## 这是什么

一句话：**把「训练一个中文语言模型」这件事，从头到尾做完整。**

同一个配置文件、同一个界面、同一条代码路径，从你显卡上跑得动的 13M，
一直到大企业才玩得起的万亿参数 —— 配置 schema 与显存估算器是同一份；
真正的多卡训练还在计划中（见上面的诚实说明）。

| 你是谁 | 模型规模 | 怎么用 |
|---|---|---|
| 完全没接触过 | 13M | 双击启动器 → 点几下 → 看它写中文 |
| 学生 / 爱好者 | 10M ~ 100M | 图形界面 + 配置文件，单张显卡 |
| 研究者 / 企业 | 1B ~ 1T+ | **同一份配置**；换后端（FSDP2 / DeepSpeed / Megatron）——*规划中，当前仅生成启动脚本* |
| 想要个能聊天的模型 | 任意 | 预训练 → SFT → 聊天界面 |

## 三个「不一样」

**① 它会对你说实话。** 配置超出你显卡能力时，它算给你看「需要约 10 张卡」，
并列出省显存的具体办法（带能直接抄的 `--set`）。**绝不假装能在本地跑万亿参数。**

**② 它默认说中文。** 文档、界面、报错全是中文；分词器按中文优化（字符级 BPE，
不用为中文跑不动的字节级方案）；连对话模板默认都是「用户：/助手：」这种
中文标记 —— 因为对小模型来说，用本来就是汉字的标记比 `<|im_start|>` 学得快得多。

**③ 报错是给人看的。** 不说 `size mismatch for blocks.0.attn...`，而是：

```
模型结构和断点对不上，没法续训：
  层数 n_layer     断点里是 8      ，现在配置是 16
三选一：
  ① 把配置改回去，再 --resume
  ② 不改配置，用 --set 临时对上
  ③ 就想要新结构 -> 删档重训：加 --force
```

## 快速开始（三种方式）

### 方式一：下载启动器（最简单）

下载 `LingYu.exe` → 双击 → 首次运行会自动准备环境（约 2.5 GB，只做一次）
→ 然后就是一个桌面界面。

所有东西都装在 exe 旁边的 `runtime\` 里，**删文件夹就等于卸载干净**。

### 方式二：从源码装

```bash
git clone https://github.com/nanchengwanfeng/lingyu
cd lingyu
python -m venv venv

# Windows
venv\Scripts\pip install -e ".[gui]"
# 注意：PyTorch 要按你的显卡单独装（见下）

venv\Scripts\lingyu gui
```

**PyTorch 刻意没有写进依赖里** —— CUDA 版本必须和你的显卡匹配，
自动装很容易装错。较新的 NVIDIA 显卡（较新的 GPU 架构）需要 cu128 及以上的 wheel：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

详细的环境搭建步骤见 [读我先看.md](读我先看.md)。

### 方式三：命令行跑通「能聊天」的全流程

```bash
python chat/fetch_sft_data.py                 # 下载 SFT 数据（一次；30MB，CC BY-NC 4.0）
lingyu train --config configs/chat.ini        # 1) 预训练（data/ 的语料）
lingyu sft   --config configs/chat.ini        # 2) 监督微调（学会回答问题）
lingyu chat  --config configs/chat.ini        # 3) 直接提问
```

`configs/chat.ini` 把「预训练 → SFT → 聊天」装进同一份配置。它是 13M 的
小模型：学会的是**怎么回答**（格式、口吻、该停就停），不是知识。想要像样的
聊天，换成 `configs/chat_medium.ini`（47.8M，同样三条命令），或把
`n_layer`/`n_embd` 继续放大并喂更多语料。

### 方式四：只想看看命令行

```bash
lingyu doctor                                    # 体检环境
lingyu scan   --raw D:\我的小说                    # 看看语料情况
lingyu prepare --raw D:\我的小说 --data D:\data     # 准备数据
lingyu plan   --config configs/small.ini          # 看显存够不够
lingyu train  --config configs/small.ini          # 预训练
lingyu sft    --config configs/sft.ini            # 教它回答问题
lingyu chat   --config configs/sft.ini            # 聊天
```

## 它会做什么

```
语料准备 → 预训练 → 监督微调(SFT) → 聊天
   ↑
   └── 对话导出为 SFT 训练数据（计划中，尚未实现）
```

| 命令 | 干什么 | 界面里在哪 |
|---|---|---|
| `lingyu doctor` | 环境体检（Python/torch/CUDA/核心模块） | 概览 |
| `lingyu scan` | 语料体检，算建议词表大小 | 语料准备 |
| `lingyu prepare` | 训练分词器 + 编码成 token | 语料准备 |
| `lingyu params` | 算参数量和 KV cache | 概览 |
| `lingyu plan` | 显存与耗时估算 | 训练 |
| `lingyu train` | 预训练 | 训练 / 监控 |
| `lingyu sft` | 教它「回答问题」 | 微调 SFT |
| `lingyu chat` | 直接提问，不用给开头词 | 聊天 |
| `lingyu sample` | 给个开头往下写 | 续写 |
| `lingyu launch` | 本机跑不动时，生成分布式启动脚本 | — |
| `lingyu gui` | 打开桌面界面 | 就是它 |

## 图形界面

九个页面：概览 / 语料准备 / 训练 / 监控 / 断点 / 续写 / 微调 SFT / 聊天 / 设置。

- **界面进程里没有 PyTorch** —— 窗口 0.06 秒就出来，训练在子进程里跑。
  8GB 显存的老机器也能流畅打开。
- **所有参数都能在界面上调**，改一个数字右边的显存估算立刻跟着变。
- 参数按「常用 / 进阶 / 专家」分三层，新手只需要看第一层。
- 深色 / 浅色两套主题，中英双语可切换。

## 它不做什么（说在前面）

- **不会替你搞到好数据。** 数据质量决定一切，这是引擎解决不了的。
- **不会让 13M 的模型变成 ChatGPT。** 小模型 SFT 之后能学会「怎么回答」
  的格式和口吻，但答不对事实 —— 它没读过互联网，只是把那几本书背下来了。
- **不自己重写 DeepSpeed / Megatron。** 那些是成熟的大型项目，
  灵语只做「把你的配置翻译成它们需要的参数」，真正的训练交给它们跑。
  自己造一个半成品对使用者是有害的。
- **多卡训练尚未实现。** 单机多卡（FSDP2）是**计划中**的能力，当前 `train`
  只跑单卡（`local`）；`lingyu launch` 生成的启动脚本里标了 `[TODO]` 的地方
  需要你自己确认，脚本也不会替你完成分布式训练循环。
- **不支持多机训练开箱即用。**

## 硬件要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| 显卡 | 支持 CUDA 的卡，4GB 显存 | 8GB 以上 |
| 内存 | 8 GB | 16 GB 以上 |
| 磁盘 | 5 GB（含依赖） | 20 GB 以上 |
| 系统 | Windows 10 / Linux | — |

**实测（单卡 8GB 显存，可用约 6.9 GB）**：

| 配置 | 参数量 | 训练速度 | 峰值显存 |
|---|---|---|---|
| nano65 | 12.59 M | — | 1.55 GB |
| small（16 层×256） | 13.08 M | 待重测 | ~1.6 GB |
| medium | 47.80 M | 25,221 tok/s | 5.24 GB |
| moe | 53.52 M（激活 34.64 M） | 46,657 tok/s | 3.93 GB |
| wide | 79.45 M | 22,373 tok/s | 5.54 GB |

> medium/moe/wide 的速度是旧配置的历史实测；small 已从 8 层×512 改为
> 16 层×256（13.08M），速度与显存待重测。

8GB 的卡训练 79M 很舒服。再大就得动 `batch_size` 或上并行（并行训练待实现）。

## 怎么改架构（不用碰核心代码）

组件走注册表。往 `plugins/` 里丢一个 `.py` 文件就能替换注意力 / 前馈 /
归一化 / 位置编码，然后配置里点名用它：

```ini
[model]
attention = 我的注意力
ffn       = 我的前馈
```

```python
# plugins/我的组件.py
from lingyu.registry import register
import torch.nn as nn

class 我的注意力(nn.Module):
    #: 估算参数量用的一维系数（可选）
    PARAM_DIM_MULT = 1
    def __init__(self, cfg): ...
    def forward(self, x, cache=None, cache_pos=0): ...

register("attention", "我的注意力", override=True)(我的注意力)
```

插件加载失败**不会让引擎起不来** —— 错误会被记录下来，内置组件继续使用。
（`lingyu doctor` 目前只报告 Python/torch/CUDA/核心模块健康，**还不列举**
已加载的插件。）

## 测试

```bash
pytest tests/ -q            # 全部 619 条（含真跑一遍 train→sft→chat）
pytest tests/ -q -m "not slow"   # 跳过端到端，快得多
```

端到端回归会真的建分词器、真的训练 12 步、真的生成、真的聊天 ——
不是 mock。测试写法有两条硬规矩写在 `docs/DESIGN.md` 第 9 节：
**能用实测就别用公式**、**每个测试对应一个真实踩过的坑**。

## 文档

| 文件 | 给谁看 |
|---|---|
| [读我先看.md](读我先看.md) | 完全没接触过，从装环境开始 |
| [命令手册.md](命令手册.md) | 每条命令的每个参数是什么意思 |
| [引擎原理.md](引擎原理.md) | 想搞懂内部怎么工作的 |
| [docs/DESIGN.md](docs/DESIGN.md) | 想改代码 / 提 PR 的，**权威设计文档** |

## 参与

提 Issue、提 PR 都欢迎。开发环境的搭建见 [读我先看.md](读我先看.md)。

有一条规矩请遵守：**改动设计之前先改 `docs/DESIGN.md`**。
那份文档记录了每个决策和它的理由 —— 不写理由的决定，后来的人会推翻它。

## 许可

Apache-2.0，见 [LICENSE](LICENSE)。带专利授权，商用没有额外顾虑。

界面用 PySide6（LGPL），不是 PyQt6（GPL）—— 就是为了让企业敢用。
