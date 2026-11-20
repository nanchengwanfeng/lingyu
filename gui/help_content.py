# -*- coding: utf-8 -*-
"""帮助页的文章库（内置，不联网）。

每篇 = {id, title, group, keywords, body}。body 是 Markdown，QTextBrowser
直接渲染。写的时候守三条：

1. **只写能验证的东西**：命令、参数、数字都来自本引擎（`docs/` 和代码）。
2. **说人话**：面向零基础，但别灌水；不写「作为一个 AI」这种。
3. **短**：一篇讲清一件事，长文拆开。

搜索（`search()`）支持三种模式：
- smart：空格分词，每个词都要命中（标题/关键词权重更高），默认；
- exact：整串必须原样出现；
- fuzzy：标题/关键词做 difflib 相似度，容错错别字。
"""

import difflib

ARTICLES = [
    dict(id="first", title="从零训练第一个模型（总览）", group="入门",
         keywords=["入门", "教程", "第一次", "流程", "quickstart", "开始"],
         body="""## 四步，一条流水线

1. **准备语料**：把 txt / md / json / jsonl 放进一个文件夹。
   在「语料准备」页选目录 → 体检 → 开始准备。
   产物是 `tokenizer.txt` + `train.bin` / `val.bin`，默认放在 `data/`。
2. **预训练**：在「训练」页选配置文件（`configs/small.ini` 起步）→
   右侧看显存估算 → 点「开始训练」。
   13M 的小模型在 8GB 显存显卡 上约 27 分钟跑 3000 步，loss 会从 8 左右降到 3.5。
3. **微调 SFT**：预训练只学会「接着写」，SFT 才教它「回答问题」。
   需要对话数据（JSONL，一行一条），默认用 `external_data/alpaca_gpt4_zh.jsonl`。
4. **聊天**：「聊天」页选 SFT 断点 → 加载模型 → 提问。

## 先跑通再调大

第一遍别改参数。用 `small` 跑完四步、确认能聊，再去动层数/宽度/语料。
小模型答不好是正常的：它学的是「怎么回答」的格式，不是知识。

## 命令行等价写法

```
lingyu prepare --raw D:/my_corpus --data data
lingyu train --config configs/small.ini
lingyu sft   --config configs/sft.ini
lingyu chat  --config configs/sft.ini
```
"""),
    dict(id="corpus", title="语料怎么准备", group="入门",
         keywords=["语料", "corpus", "prepare", "txt", "md", "json", "jsonl", "分词器", "tokenizer"],
         body="""## 支持什么

`txt` / `md` / `json` / `jsonl`（`.ndjson` 也行），支持子文件夹。
`json` 默认**保留原始结构**当文本学；勾「抽 JSON 正文」才只取正文字段
（text/content/instruction/output/messages… 这些），丢掉键名和括号。

## 体检先看三行

「语料准备」页点「体检一下」：

- **可读文件数**：0 说明目录选错或格式不认识；
- **总字符数 / 建议词表**：词表默认 8192，字符少时会建议 4096；
- **每个文件的字符数**：某个文件特别小/特别大，通常是路径或编码问题。

## 分词器指纹（最容易踩的坑）

`prepare` 会写 `train.meta.json`，里面记了分词器指纹。**重新 prepare 之后
分词器变了，旧断点就作废**（每个汉字 ID 都可能变），续训会被引擎拦下。

换语料 = 换一次断点。这不是 bug，是保护。

## 常用参数

- `--vocab-size 8192`：词表越大压缩比越高、embedding 越大；
- `--val-ratio 0.02`：验证集比例，2% 够用；
- `--tokenizer-sample-mb 3`：训 BPE 时抽多少 MB，语料大时不用调大；
- `--strip-ads`：网文下载版建议开，去网址和广告串。
"""),
    dict(id="pretrain", title="预训练在学什么", group="训练",
         keywords=["预训练", "train", "training", "loss", "困惑度", "perplexity", "next token"],
         body="""## 一句话：学「下一个 token 是什么」

预训练只做这件事。数据里每个位置都是答案，所以不需要人工标注 ——
十几本书就能让 13M 模型学会中文的字词搭配和基本句式。

## 看 loss，不看「像不像」

- 8.0 左右 = 随机猜（词表 8192 的 ln(8192)≈9，模型一开始比随机还差）；
- 4.0 左右 = 学会常用字词组合；
- 3.5 左右 = 13M / 3000 步的正常水平；
- 再往下要看语料规模和模型容量，硬压会过拟合。

困惑度（perplexity）= e^loss，表示「平均在多少个候选里犹豫」。

## 什么时候停

小模型 3000~5000 步足够。日志里 `[试写]` 行会周期性给出样例：
从乱码 → 有标点的片段 → 通顺短句，就可以停了。
继续加步数收益很小，不如换更大的模型档位。
"""),
    dict(id="sft", title="微调 SFT：教它回答问题", group="对话",
         keywords=["sft", "微调", "微调", "问答", "对话", "alpaca", "instruction", "template", "plain"],
         body="""## 为什么还要 SFT

预训练模型只会「接着写」。你问「李白是谁」，它接着写的是另一句诗，
不是回答。SFT 用「问题 → 回答」成对的数据，教它**看到问题就切到回答模式**。

## 数据格式

JSONL，一行一条，四种写法都认：

```json
{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}
{"conversations": [{"from": "human", "value": "…"}, {"from": "gpt", "value": "…"}]}
{"prompt": "…", "response": "…"}
{"instruction": "…", "input": "…", "output": "…"}
```

空行、坏行会被跳过并在日志里计数。清洗在 `lingyu chat_template` 里，
模板用 `plain`（用户：/助手：）不需要特殊 token，任何断点都能用。

## 只学回答

loss mask 只把「助手」部分算进 loss，问题部分不计分。
所以 SFT 的 loss 数字**不能**和预训练比（口径不同），看 `[验证]` 和
`[试问]` 的实际回答更可靠。

## 参数经验

- 步数 2000~6000，数据少就少跑；
- 学习率 2e-4 左右，比预训练小一个量级；
- `--resume` 接着训，`--init-from` 换预训练起点。
"""),
    dict(id="chat", title="聊天页怎么用", group="对话",
         keywords=["聊天", "chat", "对话", "温度", "temperature", "回答"],
         body="""## 先有 SFT 断点

聊天页只认 SFT 之后保存的 `best_sft.pt` / `ckpt_sft.pt`。
如果列表是空的，先去「微调 SFT」跑一遍。

## 参数

- **温度**：0.6~0.8 稳，1.0 更发散。回答绕圈就调低；
- **最长回复**：给答案留的空间，小模型 200 够用；
- 回车发送，Shift+回车换行；生成中可以点停止。

## 回答不好怎么办

先确认三件事：用的是不是 SFT 断点、问题是否超出训练分布、模型是不是太小。
13M 模型能学会「回答的格式」，但知识极少 —— 这是容量上限，不是参数没调好。
"""),
    dict(id="vram", title="显存够不够：估算器怎么算", group="训练",
         keywords=["显存", "vram", "oom", "估算", "估算器", "8GB", "5060", "batch", "梯度"],
         body="""## 估算公式（引擎里就这一份）

训练显存 ≈ 参数 × 16 字节 + 激活 ≈ batch × block_size × n_embd × n_layer × 22。

- 16 字节/参数 = bf16 权重 2 + 梯度 2 + fp32 主权重 4 + AdamW 两个动量 8；
- 激活那项是经验系数，实际会随实现浮动；
- 估算结果再乘 **1.2 安全余量**、并先扣掉 15% 系统占用，才和可用显存比。

所以「估算 5.84 GB / 可用 6.87 GB」显示够用，实际峰值可能到 6.5~7 GB，
边训练边开浏览器就可能 OOM。

## 显存不够，按这个顺序降

1. **batch_size 减半**（最有效，激活项线性下降）；
2. **block_size 512 → 256**（激活项再减半，但长文本被截）；
3. **n_layer / n_embd 换小档**（medium → small → nano65）；
4. `grad_accum` 加大，用时间换显存（tokens/步不变）。

## 可用显存显示不准？

桌面程序（浏览器/游戏/录屏）会占显存，探测只代表**那一刻**。
点「重新检测」会重新问一次驱动；游戏没退干净时数字仍会偏小。
"""),
    dict(id="batch", title="batch_size 与梯度累积", group="训练",
         keywords=["batch", "批大小", "grad_accum", "梯度累积", "tokens", "吞吐"],
         body="""## tokens/步 = batch_size × grad_accum × block_size

一次 optimizer.step() 看到的 token 数。它决定两件事：

- **显存**：只有 batch_size 影响激活占用，grad_accum 不额外吃显存；
- **速度**：tokens/步 越大，同样步数看到的数据越多，但每步更慢。

显存不够时的标准做法：batch_size 减半、grad_accum 翻倍，
tokens/步 不变，只是每一步拆成两次前向+反向。

## 小模型别贪大

13M 的模型用 batch 16~32（block 512）就够。batch 太大反而学得慢：
每步梯度更稳，但同样时间走的步数少了，小语料上还会反复看同几批数据。
"""),
    dict(id="lr", title="学习率、预热与余弦衰减", group="训练",
         keywords=["学习率", "lr", "learning rate", "warmup", "预热", "余弦", "cosine", "nan"],
         body="""## 常用值

- 预训练：`1e-3` 起步，`min_lr 1e-4`，warmup 100~400 步；
- SFT：`2e-4` 左右（比预训练小一个量级），warmup 50~100 步。

## 三条曲线

- **warmup**：从 0 线性升到峰值，避免一开始就把权重带偏；
- **cosine**：从峰值按余弦降到 `min_lr`，越到后面步子越小；
- **grad clip**：默认 1.0，把梯度范数压到 1 以内，防止炸。

## loss 变 NaN

99% 是学习率太大或数据里有空样本。先把学习率调小一个数量级，
再检查语料是否有大量空行/乱码。NaN 一旦进梯度，所有参数都会变 NaN，
必须从断点重来。
"""),
    dict(id="moe", title="MoE 是什么，什么时候用", group="进阶",
         keywords=["moe", "专家", "expert", "路由", "router", "稀疏", "激活参数"],
         body="""## 一个模型里放多个「专家」

MoE 把 FFN 拆成 N 个专家，每个 token 只走其中 top-k 个（另有共享专家）。
好处：**总参数变大，但每个 token 只算激活的那部分**，训练吞吐高。

本引擎的 MoE 在 `configs/moe.ini`：8 专家 + 每 token 走 2 个 + 1 个共享专家。
实测（8GB 显存显卡，8GB）：训练 46,657 tok/s，峰值 3.93 GB；
但**生成慢一倍**（114 tok/s，8 个专家要各走一次矩阵乘）。

## 什么时候别用

- 只想要小模型对话：用地道的 dense（small/medium）更省心；
- 显存紧：MoE 省的是计算，不是参数显存。

## 路由平衡

训练时有两个辅助 loss：`aux_loss`（专家负载均衡）+ `z_loss`（稳定路由）。
日志里 `aux=0.01 z=0.001` 是正常的，突然变大会拖慢收敛。
"""),
    dict(id="gqa", title="n_head 与 n_kv_head（GQA）", group="进阶",
         keywords=["gqa", "mqa", "头数", "n_head", "n_kv_head", "kv cache", "注意力"],
         body="""## 两套头数

`n_head` = query 头数；`n_kv_head` = key/value 头数。
`n_kv_head < n_head` 就是 GQA：多个 query 头共享一组 KV 头。

## 省的是 KV cache

KV cache / token = 2 × n_layer × n_kv_head × head_dim × 2 字节。
`n_kv_head` 减半，长上下文的缓存就减半 —— 推理时要产出的 token 越多，
这个差距越明显。

## 约束

`n_head` 必须能被 `n_kv_head` 整除，且 `head_dim`（n_embd/n_head）通常保持
64 或 128。配置校验会在启动前拦下不整除的组合。
"""),
    dict(id="block", title="block_size（上下文长度）", group="训练",
         keywords=["block_size", "上下文", "context", "序列长度", "长文本"],
         body="""## 它是什么

训练时每条样本截成 `block_size` 个 token。512 是 8GB 卡上的常用值：
够学句子和段落，显存也扛得住。

## 影响

- 显存：激活项与 block_size 近似线性；
- 生成：`sample` / `chat` 一次最多看这么多 token，答长文会被截；
- loss：太小（<128）学不到跨句关系；太大（>1024）8GB 卡基本 OOM。

## 和续训的关系

block_size 记在断点里，续训时如果改了会被结构检查拦下 ——
它是模型结构的一部分，不是纯训练参数。
"""),
    dict(id="cli", title="命令行怎么用", group="工具",
         keywords=["cli", "命令行", "终端", "命令", "--set", "--resume", "--init-from", "--json", "doctor"],
         body="""## 常用命令

```
lingyu doctor                 # 体检：torch/CUDA/语料是否就绪
lingyu scan --raw D:/corpus   # 只统计语料，不写文件
lingyu plan   --config configs/small.ini   # 显存/耗时估算
lingyu train  --config configs/small.ini
lingyu sft    --config configs/sft.ini
lingyu chat   --config configs/sft.ini
lingyu sample --config configs/small.ini --prompt "他抬起头"
lingyu launch --config configs/xxx.ini     # 本机跑不动时生成分布式脚本
```

## 三个救命参数

- `--set model.n_layer=12`：临时改配置，不动文件，可写多次；
- `--resume [路径]`：续训（用原优化器状态）；`--force` 从头重训；
- `--init-from 路径`：拿别的断点权重作起点（例如 SFT 后继续预训练）。

## --json（给程序用）

`train` / `sft` / `sample` 支持 `--json`：stdout 只输出一行一个 JSON 事件
（start / step / eval / ckpt / sample / probe / done / error），
人类日志改走 stderr。GUI 就是这么读进度的，写脚本也方便。
"""),
    dict(id="errors", title="常见报错与处理", group="工具",
         keywords=["报错", "错误", "error", "oom", "找不到", "不一致", "分词器", "断点", "指纹"],
         body="""## 「模型结构和断点对不上」

续训要求层数/宽度/头数/block_size/词表完全一致。要么把配置改回去
（报错会给具体值），要么用 `--set` 临时对上，要么 `--force` 删档重训。

## 「分词器与断点不配套」

重新 `prepare` 之后分词器变了，旧断点作废。换回旧 data 目录，
或者用新数据从头练。

## 「找不到断点」

`chat` / `sample` 不给 `--ckpt` 时会找 `post.out_dir`、`train.out_dir`。
先确认这些目录里真有 `*.pt`；断点页也能看到所有存档。

## CUDA out of memory

按这个顺序：batch_size 减半 → block_size 减半 → 换小档模型。
估算页显示「够用」只代表估算值，开浏览器/游戏会额外吃显存。

## loss 变 NaN

学习率太大或数据里有空样本。调小学习率，检查语料。
"""),
    dict(id="glossary", title="术语表", group="工具",
         keywords=["术语", "词表", "token", "loss", "困惑度", "epoch", "step", "bf16", "fsdp", "checkpoint", "断点"],
         body="""## 训练

- **token**：模型眼里的最小单位。中文常用词可能是一个 token，罕见字是 2~3 个。
- **step**：一次参数更新。1 step 看 `batch_size × block_size` 个 token（预训练）。
- **epoch**：把整个数据集过一遍。小语料常用步数而不是 epoch 来计。
- **loss**：交叉熵，越小越好；困惑度 = e^loss。
- **lr**：学习率；**gnorm**：梯度范数，用来判断训练是否稳。
- **bf16**：半精度格式，省显存且数值范围够；训练默认混合精度。

## 模型/部署

- **ckpt（断点）**：权重 + 优化器状态 + 步数 + 配置快照，续训要用整套；
  `best.pt` 是验证 loss 最低的那次。
- **GQA**：多个 query 头共享 KV 头，省 KV cache。
- **MoE**：稀疏专家，总参数大、每 token 只激活一部分。
- **FSDP**：多卡参数分片（本版本只生成启动脚本，单机多卡训练未实现）。
- **`--json`**：把进度输出成结构化事件，给 GUI / 脚本消费。
"""),
]

# 百科正文单独维护（gui/help_articles_extra.py），这里只负责拼进来
try:                                  # GUI 内是包导入
    from .help_articles_extra import EXTRA as _EXTRA
except ImportError:                   # 测试/脚本把 gui/ 当顶层目录导入
    from help_articles_extra import EXTRA as _EXTRA

# 学术向百科（第二批：公式 + 论文对应）
try:
    from .help_articles_academic import ACADEMIC as _ACADEMIC
except ImportError:
    from help_articles_academic import ACADEMIC as _ACADEMIC

_known = {a["id"] for a in ARTICLES}
for _art in list(_EXTRA) + list(_ACADEMIC):
    if _art["id"] not in _known:
        ARTICLES.append(_art)
        _known.add(_art["id"])

_GROUP_ORDER = ["入门", "概念", "参数", "训练", "后训练", "排查", "进阶", "引擎", "术语"]

# 旧文章里还有「对话 / 工具」两组：新组排前面，缺的补到末尾，
# 否则 by_group() 会把它们静默吞掉（帮助页看不到）。
for _g in dict.fromkeys(a["group"] for a in ARTICLES):
    if _g not in _GROUP_ORDER:
        _GROUP_ORDER.append(_g)


def by_group():
    """按「入门 → 训练 → 对话 → 进阶 → 工具」返回文章，首页用。"""
    out = []
    for group in _GROUP_ORDER:
        for art in ARTICLES:
            if art["group"] == group:
                out.append(art)
    return out


def get(article_id):
    for art in ARTICLES:
        if art["id"] == article_id:
            return art
    return None


def _score(article, query, mode):
    ql = query.lower()
    title = article["title"].lower()
    keywords = [k.lower() for k in article["keywords"]]
    hay = title + " " + " ".join(keywords) + " " + article["body"].lower()
    if mode == "exact":
        return 100.0 if ql in hay else 0.0
    if mode == "fuzzy":
        best = max([difflib.SequenceMatcher(None, ql, title).ratio()]
                   + [difflib.SequenceMatcher(None, ql, k).ratio() for k in keywords])
        sub = 1.0 if ql in hay else 0.0
        if best < 0.45 and not sub:
            return 0.0
        return best * 100 + sub * 20
    # smart：空格分词，每个词都要命中
    terms = [w for w in ql.replace("，", " ").replace(",", " ").split() if w]
    if not terms:
        terms = [ql]
    for term in terms:
        if term not in hay:
            return 0.0
    score = 0.0
    for term in terms:
        if term in title or any(term in k for k in keywords):
            score += 6
        else:
            score += 2
    if ql in title:
        score += 8
    if ql in hay:
        score += 3
    return score


def search(query, mode="smart", limit=40):
    """按查询返回文章列表；空查询返回全部（按首页顺序）。"""
    q = (query or "").strip()
    if not q:
        return by_group()
    scored = []
    for art in ARTICLES:
        score = _score(art, q, mode)
        if score > 0:
            scored.append((score, art))
    scored.sort(key=lambda item: (-item[0], item[1]["title"]))
    return [art for _score, art in scored[:limit]]
