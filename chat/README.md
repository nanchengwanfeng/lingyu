# 对话数据集（SFT 用）

这个目录放**对话数据集**，用来教模型「回答问题」而不只是「续写」。

## 格式：JSONL（一行一条样本）

一行就是一条完整的样本，**换行必须是真正的换行符分隔，不能整个文件是一行**。

支持下面四种写法，挑一种用就行：

```jsonl
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
{"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好！"}]}
{"prompt": "你好", "response": "你好！"}
{"instruction": "介绍一下李白", "input": "", "output": "李白是唐代诗人。"}
```

- `role` 可以是 `system` / `user` / `assistant`，也认中文的「系统 / 用户 / 助手」
- 多轮对话就写多条 message，顺序别乱
- 空行和 `//` `#` 开头的行会被跳过，可以拿来写注释

## 现成的示例

`sample_chat.jsonl` 里有 50 条示例，覆盖了闲聊、简单常识、算术、
系统提示词、多轮对话。它的用途是：

1. **验证流程能跑通** —— 先拿它跑一遍 `lingyu sft`，确认没问题
2. **当作格式模板** —— 照着它的样子写自己的数据

⚠️ **50 条数据练不出有用的模型**。这点数据只够让模型学会「该它说话了」
这个格式，学不到任何知识。13M 的模型至少要几千条才能像个样子。

## 现成的中文指令数据：Alpaca-GPT4 中文（推荐先用它跑通）

`configs/chat.ini` 默认用 ModelScope 的 `AI-ModelScope/alpaca-gpt4-data-zh`
—— 4.9 万条 GPT-4 生成的中文问答，覆盖常识、写作、代码、日常建议等。
许可证 **CC BY-NC 4.0（非商用）**：自己玩/研究没问题，商用请换数据集或
去数据页确认最新条款。

仓库不直接带这份数据（原始 CSV 30MB）。第一次跑聊天流程前，先执行一次：

```powershell
python chat/fetch_sft_data.py                # 下载 → 过滤 → 打乱 → external_data/alpaca_gpt4_zh.jsonl
python chat/fetch_sft_data.py --help         # 换输出目录 / 超长过滤 / 最多保留条数
```

产物在 `external_data/`（不进发布包，删了重跑即可）。过滤规则是「总长度
≤ 700 字符」——为了适配 `configs/chat.ini` 的 max_seq_len=512；把
`max_seq_len` 调大后，可以用 `--max-chars` 放宽。

## 上哪儿找更多数据

| 来源 | 说明 |
|---|---|
| 自己写 | 质量最高。写 500 条你自己的问答，效果比网上随便扒 5 万条好 |
| 把自己的文档改写成问答 | 有 PDF/笔记的话，用大模型批量改写成问答对，再人工过一遍 |
| ModelScope / HuggingFace | 搜「中文 对话 数据集」，注意看许可证能不能商用 |
| 用大模型蒸馏 | 拿本地 Qwen 之类的模型批量生成问答，再筛 |

## 为什么数据质量比数量重要

SFT 是在**教格式和风格**，不是在灌知识。几条错得离谱的样本，
模型会学得一模一样地离谱。所以：

- **宁少勿滥**：500 条干净的比 5 万条脏的好
- **回答长度尽量一致**：混着「嗯」和三百字长文，模型会学得忽长忽短
- **别有重复**：同一条出现几百遍，模型会只会回那一句

## 下一步

**推荐：直接用 `configs/chat.ini` 走完整聊天流水线**

```powershell
python chat/fetch_sft_data.py               # 1. 下载 SFT 数据（只需一次）
lingyu train --config configs/chat.ini      # 2. 预训练（学会说人话）
lingyu sft   --config configs/chat.ini      # 3. SFT（学会回答问题）
lingyu chat  --config configs/chat.ini      # 4. 对话
```

13M 的 `chat.ini` 是流程验证档（学会格式、内容会跑偏）；**想要像样的回答，把
上面四条命令里的 `chat.ini` 换成 `configs/chat_medium.ini`（47.8M）**，
预训练约 9 分钟、SFT 约 12 分钟（单卡 8GB 显存实测）。

**想先验证格式对不对**：拿 50 条示例跑个短 SFT

```powershell
lingyu sft --config configs/sft.ini --data chat/sample_chat.jsonl --set post.max_steps=200
lingyu chat --config configs/sft.ini
```
