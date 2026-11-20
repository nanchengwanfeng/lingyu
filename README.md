# lingyu-lm

**灵语（LingYu）** — an engine for training language models **from 13M to
trillions of parameters**. Chinese-first. Beginner-friendly. Honest about scale.

[中文说明](README.zh-CN.md) · [Design doc](docs/DESIGN.md) · Apache-2.0

**Status:** v0.4 — GUI, post-training, distributed *planning* and Windows
packaging are in place. 620 tests passing.

> **Honest scope note:** multi-GPU/cluster **training is not implemented yet**
> (planned, P3 remaining). `lingyu launch` only generates launch scripts;
> `[parallel] backend=fsdp/deepspeed/megatron` does not run a distributed
> training loop today. `lingyu export` / `lingyu demo` / `chat-template check`
> are **planned** (P4/P7), not available.

## Install

### Option A — Windows launcher (easiest)

Download `LingYu-win64.zip` from Releases, unpack it, double-click `LingYu.exe`.

The first run prepares everything automatically (~2.5 GB, once). Everything
lives in a `runtime\` folder next to the exe — **deleting that folder
uninstalls it completely**: no registry entries, no leftovers elsewhere.

If anything goes wrong, the launcher keeps a log at `runtime\launcher.log`,
and `LingYu.exe --console --reinstall` shows exactly what it is doing.

### Option B — from source

```bash
pip install -e ".[gui]"                 # core + desktop GUI
pip install torch --index-url https://download.pytorch.org/whl/cu128
lingyu gui
```

PyTorch is deliberately **not** a dependency — the CUDA build must match your
GPU. Recent NVIDIA GPUs may need cu128 or newer wheels. Chinese PyPI mirrors are
configured by default in the launcher, because pulling 2.5 GB from pypi.org
directly is not realistic for most users.

## What it does

```
prepare corpus → pretrain → SFT → chat
```

| Command | What it does |
|---|---|
| `lingyu gui` | Desktop GUI — 11 pages, every parameter editable |
| `lingyu doctor` | Environment check (torch/CUDA/bf16/core modules) |
| `lingyu scan` / `prepare` | Corpus check, tokenizer training, tokenize |
| `lingyu params` / `plan` | Parameter count, VRAM & time estimate |
| `lingyu train` | Pretraining (dense or MoE) |
| `lingyu sft` | Supervised fine-tuning — teaches it to *answer* |
| `lingyu chat` | Ask it questions directly (no prompt continuation needed) |
| `lingyu sample` | Continue from a prompt (pre-training style generation) |
| `lingyu launch` | Generate distributed launch scripts when it doesn't fit |

> Exporting to HuggingFace format (`lingyu export`) is **planned (P4)** and not
> implemented yet.

## Train a model that can chat (one config)

`configs/chat.ini` is a complete chat pipeline in a single file:
pretrain on your corpus, then SFT on Chinese instruction data, then talk.

```bash
python chat/fetch_sft_data.py            # once: Alpaca-GPT4 zh (CC BY-NC 4.0)
lingyu train --config configs/chat.ini   # 1) pretrain on data/
lingyu sft   --config configs/chat.ini   # 2) SFT on the downloaded data
lingyu chat  --config configs/chat.ini   # 3) ask it questions
```

It is a 13M-parameter model: it learns the **format** of answering (and some
of the corpus), not knowledge. Want visibly better answers? Use
`configs/chat_medium.ini` (47.8M) — same three commands. For a serious model,
scale `n_layer`/`n_embd` up and give it more data. Details: [chat/README.md](chat/README.md).

## One engine, 13M to 1T

The same config file is *designed* to work at every scale; the config schema and
the VRAM estimator already do. **Actual multi-GPU training is not implemented
yet** — `lingyu launch` generates `torchrun` / DeepSpeed launch scripts for you
to inspect and run, but the engine's own training loop only runs on a single
device (`local` backend) today.

**It tells you the truth about scale.** When a config exceeds your GPU it
computes *how many cards you would need* and lists concrete ways to shrink it,
instead of silently OOM-ing or refusing without explaining:

```
[注意] 这个规模在本机**跑不动** —— 需要约 10 张同等显存的卡。
  ① 换数据并行分片（FSDP）…  --set parallel.backend=fsdp --set parallel.zero_stage=2
  ② 开激活重算，激活值约降到 1/4 …  --set parallel.activation_checkpoint=true
  ⑥ 真要用这个规模：lingyu launch --config <配置>  生成启动脚本
```

It is equally honest about what it does **not** do: it does not reimplement
DeepSpeed or Megatron. At those scales it only *translates your config* into
launch scripts, with everything you must confirm yourself marked `[TODO]`.

## Pluggable architecture

Replace attention / FFN / normalization / positional encoding without touching
core code — drop a `.py` file into `plugins/` and name it in the config.
Full contract, registration and a runnable example: [docs/EXTENDING.md](docs/EXTENDING.md).

```ini
[model]
attention = my_attention
ffn       = my_ffn
```

A failing plugin never takes the engine down; the error is logged and the
built-in component stays in use. (`lingyu doctor` currently reports Python /
torch / CUDA / core-module health — it does not list loaded plugins yet.)

## Why Chinese-first

Character-level BPE tuned for Chinese (not byte-level, which compresses
badly on CJK), Chinese error messages, Chinese docs, and a chat template that
defaults to `用户：/助手：` markers rather than `<|im_start|>` — because for a
small model, markers that are *already ordinary Chinese characters* are far
easier to learn.

## Docs

| File | Audience |
|---|---|
| [README.zh-CN.md](README.zh-CN.md) | Chinese README |
| [读我先看.md](读我先看.md) | Complete beginner, from environment setup |
| [命令手册.md](命令手册.md) | Every command and every parameter |
| [引擎原理.md](引擎原理.md) | How it works inside |
| [docs/DESIGN.md](docs/DESIGN.md) | **Authoritative design doc** — read before changing design |

## License

Apache-2.0 — see [LICENSE](LICENSE). Includes a patent grant.

The GUI uses PySide6 (LGPL), deliberately *not* PyQt6 (GPL), so companies can
adopt it without licensing concerns.
