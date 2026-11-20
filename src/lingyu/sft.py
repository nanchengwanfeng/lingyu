# -*- coding: utf-8 -*-
r"""SFT（监督微调）—— 教模型「回答问题」，而不只是「续写」。

## 预训练模型和能对话的模型差在哪

预训练做完，你手上是一个**续写器**：给「今天天气真」，它接「好，我们去公园吧」。
这已经很厉害了，但你没法**问**它东西 —— 因为它从没见过「问题」和「回答」
这两种角色的区别。语料里所有文字都是平铺的。

SFT 就是拿几万条「问题 -> 回答」的样本再训一会儿，教它这个格式：

    用户：李白是哪个朝代的？
    助手：唐代。

样本里「用户：」「助手：」这些标记把两种文字分了工。学会之后，
你只要写到「助手：」，模型就知道该输出**回答**了。

## 三件容易做错的事（这个文件主要在处理它们）

### 1. loss 不能算在问题上（loss mask）

新手最容易漏的一步。如果对「用户：李白是哪个朝代的？」这几个字也算 loss，
模型学的就是「复述问题」—— 可问题是用户给的，它不需要学。而且问题通常
比回答长，不算 mask 的话大部分梯度都花在「抄题」上。

做法：把不该学的位置的 target 设成 `-1`，PyTorch 的
`cross_entropy(ignore_index=-1)` 会自动跳过它们。**不用改模型**。

### 2. 学习率必须比预训练小得多

预训练用 1e-3 甚至更大，SFT 通常用 1e-5 ~ 5e-4。用预训练那个学习率
会在几十步内把模型「训崩」：loss 看着很低，但它忘光了预训练学到的一切，
只会复读训练集里的回答。这叫**灾难性遗忘**，是小模型 SFT 最常见的翻车方式。

### 3. 得看着它到底会不会答

SFT 的 loss 意义不大（几百条样本就能背下来，loss 会很低但模型很蠢）。
所以每过一段就**真的问它一句**，把回答打出来看。这个文件会这么做。
"""

import os
import time

import torch

from . import accelerators, utils
from .data import BatchPrefetcher
from .chat_template import (check_ckpt_tokenizer, check_template_usable,
                            load_jsonl, render_dataset, render_for_inference)
from .config import get
from .model import build_model, config_from_section
from .tokenizer import PAD, load_tokenizer
from .train import (ckpt_extra, configure_optimizer, load_checkpoint, lr_at,
                    set_config_path,
                    save_checkpoint, structure_diffs, _restore_rng)

# SFT 期间固定拿这几句去问模型，用来肉眼观察有没有学坏。
# 选的是**训练集里没有**的问题，这样看到的是泛化而不是背诵。
PROBE_QUESTIONS = [
    "你好，请介绍一下你自己。",
    "李白是哪个朝代的诗人？",
    "什么是光合作用？",
]


# ==========================================================================
# 数据集
# ==========================================================================
class SFTDataset:
    """SFT 数据集：每条样本是 (ids, 监督位置)。

    和预训练的 TokenDataset 有两个关键区别：

    1. **不是定长切窗口**，而是「一条问答一条样本」—— 一条样本里
       问题、回答、角色标记的边界是有意义的，不能像预训练那样
       随便切一刀。
    2. **带监督掩码**：只有回答部分的 token 参与 loss。

    批次内做**动态 padding**：同批里最长的样本决定长度，
    短样本补 PAD。比全局定长省很多算力（短样本不用白填到最长）。
    """

    def __init__(self, samples, block_size, device="cpu", seed=1234):
        if not samples:
            raise ValueError("SFT 数据集是空的")
        self.samples = samples
        self.block_size = int(block_size)
        self.device = device
        self.rng = torch.Generator().manual_seed(int(seed))
        # 超过模型上下文长度的样本在这里就报出来，别等到训练时
        # 才发现被静默截断 —— 截断会砍掉回答的后半截，模型学半句话。
        too_long = [s for s in samples if len(s["ids"]) > self.block_size]
        if too_long:
            raise ValueError(
                "有 %d 条样本比模型的上下文长度还长（最长 %d > block_size %d）。\n"
                "三条路：\n"
                "  ① 把 [post] max_seq_len 调小，让 prepare 阶段就过滤掉长样本\n"
                "  ② 把 [model] block_size 调大（注意显存）\n"
                "  ③ 把长样本自己裁短\n"
                "现在这样直接开跑的话，长样本会被默默截断，"
                "回答只学一半 —— 比报错更糟。"
                % (len(too_long), max(len(s["ids"]) for s in too_long),
                   self.block_size))
        self.n_supervised = sum(s["n_supervised"] for s in samples)

    def __len__(self):
        return len(self.samples)

    def batch(self, batch_size, indices=None):
        """抽 batch_size 条，动态 padding 成一个批次。

        返回 (idx, targets, n_supervised)：
            idx       (B, T) 输入 token
            targets   (B, T) 要预测的 token，**不该学的位置是 -1**
            n_supervised  这一批里有多少个 token 真的参与 loss

        `indices` 给定时按它取样本（不随机）。评估和测试要用 ——
        否则「同一个验证集跑两次结果不一样」，没法比较两次训练谁更好。
        """
        n = len(self.samples)
        if indices is None:
            picks = torch.randint(0, n, (batch_size,), generator=self.rng).tolist()
        else:
            picks = list(indices)
            batch_size = len(picks)
        rows = [self.samples[i] for i in picks]
        T = max(len(r["ids"]) for r in rows)

        idx = torch.full((batch_size, T), PAD, dtype=torch.long)
        tgt = torch.full((batch_size, T), -1, dtype=torch.long)
        n_sup = 0
        for b, r in enumerate(rows):
            ids = r["ids"]
            L = len(ids)
            idx[b, :L] = torch.tensor(ids, dtype=torch.long)
            # mask[i] = 1 表示「第 i 个 token 属于回答」。
            # 因果 LM 在位置 i 的输出预测的是第 i+1 个 token，
            # 所以要学第 j 个 token，就把 ids[j] 放到位置 j-1 当 target：
            #   tgt[i] = ids[i+1]  iff  mask[i+1] == 1
            # 之前的写法 tgt[i]=ids[i] 等于让模型「复制当前 token」，
            # 学的是恒等映射，不是下一个 token —— 白训。
            m = torch.tensor(r["mask"], dtype=torch.long)
            row = torch.full((L,), -1, dtype=torch.long)
            if L > 1:
                row[:-1] = torch.where(m[1:].bool(),
                                       torch.tensor(ids[1:], dtype=torch.long),
                                       torch.full((L - 1,), -1, dtype=torch.long))
            tgt[b, :L] = row
            # 用真实参与 loss 的位置数，而不是 mask 总和：
            # 第 0 个 token 永远没有「上一个位置」可预测。
            n_sup += int((row != -1).sum())
        return idx.to(self.device), tgt.to(self.device), n_sup


# ==========================================================================
# 评估
# ==========================================================================
@torch.no_grad()
def eval_sft_loss(model, ds, batch_size, iters, device, amp_dtype=None):
    """在验证集上算「只看回答部分」的 loss。

    注意这个 loss 通常**远低于**预训练 loss，别因此以为自己训得特别好 ——
    它衡量的是「在回答位置上预测得准不准」，而不是「回答得好不好」。
    一个小模型完全可以 loss 很低但答非所问（它在背训练集的措辞）。
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    # 验证不需要梯度：以前这里没包 no_grad，每次验证都在建一整张计算图
    # （显存白涨、速度白慢），属于纯 bug 级别的开销。
    with torch.no_grad():
        for _ in range(iters):
            idx, tgt, _ = ds.batch(batch_size)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss, _ = model(idx, targets=tgt, collect_aux=False)
            if loss is not None and torch.isfinite(loss):
                total += float(loss.detach())
                count += 1
    if was_training:
        model.train()
    return total / max(1, count)


# ==========================================================================
# 试问
# ==========================================================================
@torch.no_grad()
def ask(model, tok, question, template, device, max_new_tokens=80,
        temperature=0.7, top_k=40, top_p=0.9, system=None):
    """真的问模型一句，返回它的回答（纯文本）。

    推理时用 `add_generation_prompt=True`：结尾写上「助手：」，
    让模型从「回答」的位置开始写。
    """
    was_training = model.training
    model.eval()
    msgs = [{"role": "user", "content": question}]
    ids = render_for_inference(msgs, tok, template=template, system=system)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k,
                         top_p=top_p, eos_id=None, use_cache=True)
    new = out[0].tolist()[len(ids):]
    if was_training:
        model.train()
    return _clean_reply(tok.decode(new))


def _clean_reply(text):
    """把生成的文字收拾干净：切掉它开始编下一轮对话的部分。

    模型常常写着写着开始「自问自答」（用户：… 助手：…），
    因为训练数据里就是这种一串。截到第一个角色标记为止。
    """
    for marker in ("\n用户：", "\n助手：", "用户：", "助手：",
                   "<|im_start|>", "<|im_end|>", "\n系统："):
        i = text.find(marker)
        if i != -1:
            text = text[:i]
    return text.strip()


# ==========================================================================
# 找基础断点
# ==========================================================================
def find_base_checkpoint(cfg):
    """找要用哪个预训练断点做起点。

    按优先级找，全找不到就报一个**列出所有找过的地方**的错 ——
    「找不到文件」却不说是哪儿找过，是新手最抓狂的报错。
    """
    tried = []
    init = get(cfg, "post", "init_from", "") or ""
    if init:
        tried.append(("post.init_from", init))
        if os.path.isfile(init):
            return init, tried
        return None, tried

    out_dir = get(cfg, "train", "out_dir", "") or ""
    for name in ("best.pt", "ckpt.pt"):
        if out_dir:
            p = os.path.join(out_dir, name)
            tried.append(("[train] out_dir/%s" % name, p))
            if os.path.isfile(p):
                return p, tried
    for cand in ("out/best.pt", "out/ckpt.pt"):
        tried.append(("默认位置 " + cand, os.path.abspath(cand)))
        if os.path.isfile(cand):
            return os.path.abspath(cand), tried
    return None, tried


def _load_base_weights(model, ckpt_path, device, tok=None):
    """把预训练权重灌进模型。结构不一致时报人话。"""
    ck = load_checkpoint(ckpt_path, map_location=device)
    # 先确认「这份断点认识当前这份分词器」—— 错配时形状可能完全一样，
    # 加载不报错、loss 照降，但 token id 全对不上，训出来只会是乱码。
    if tok is not None:
        check_ckpt_tokenizer(ck, tok, where="预训练断点")
    old_cfg = ck.get("config") or {}
    mc = model.cfg
    diffs = []
    for k, label in (("n_layer", "层数 n_layer"), ("d_model", "宽度 n_embd"),
                     ("n_head", "头数 n_head"), ("d_ff", "前馈 d_ff"),
                     ("use_moe", "是否用 MoE")):
        a, b = old_cfg.get(k), mc.get(k)
        if a is not None and b is not None and a != b:
            diffs.append((label, a, b))
    if diffs:
        utils.err("预训练断点的模型结构和当前配置对不上，没法在它基础上做 SFT：")
        for label, a, b in diffs:
            utils.info("  %-16s 断点里是 %-8s，现在配置是 %s" % (label, a, b))
        print()
        utils.info("SFT 是「在已有模型上接着教」，所以结构必须完全一样。")
        utils.info("检查一下 [model] 段是不是被改过；或者用 --set 对回来。")
        raise ValueError("模型结构与预训练断点不一致，已停下")

    sd = ck["model"]
    # 词表大小可能不同（比如 SFT 时给分词器加了新 token）—— 那就扩 embedding
    want = mc["vocab_size"]
    got_emb = sd.get("tok_emb.weight")
    if got_emb is not None and got_emb.shape[0] != want:
        utils.warn("预训练词表 %d，当前分词器词表 %d —— 会扩充 embedding。"
                   % (got_emb.shape[0], want))
        utils.info("新增的那些行是随机初始化的，需要多训一会儿才能学好。")
        model.resize_vocab(want, sd)
        sd = {k: v for k, v in sd.items()
              if k not in ("tok_emb.weight", "lm_head.weight")}
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        utils.err("加载预训练权重失败：%s" % e)
        raise ValueError("预训练权重加载失败")
    # strict=False 是为了容忍扩充出来的 embedding；但**其它**缺失/多余
    # 必须报出来 —— 静默少加载几个权重，模型会变成半个随机模型，
    # 训练照样跑、loss 照样降，你根本看不出来。
    real_missing = [k for k in missing
                    if not k.startswith(("tok_emb", "lm_head"))]
    if real_missing or unexpected:
        utils.err("预训练权重和模型对不上：")
        for k in real_missing[:8]:
            utils.info("  模型里有、断点里没有: %s" % k)
        for k in list(unexpected)[:8]:
            utils.info("  断点里有、模型里没有: %s" % k)
        raise ValueError("预训练权重与模型不匹配，已停下")
    return ck


# ==========================================================================
# 主流程
# ==========================================================================
def sft(cfg, device=None, force=False, resume=None, config_path=None):
    """跑 SFT。返回统计信息字典。

    `resume`：
        None           从预训练断点开始（默认）
        \"auto\"        从 post.out_dir/ckpt_sft.pt 续训
        一个路径       从指定 SFT 断点续训
    """
    # ------------------------------------------------------------ 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    accel = accelerators.apply(device)
    if use_cuda:
        utils.info("设备: %s（加速: %s）" % (device,
                                            accelerators.describe(accel)))

    # ------------------------------------------------------------ 分词器
    tok_path = get(cfg, "data", "tokenizer_path", None)
    data_dir = get(cfg, "data", "data_dir", None)
    if not tok_path and data_dir:
        tok_path = os.path.join(data_dir, "tokenizer.txt")
    if not tok_path or not os.path.isfile(tok_path):
        raise FileNotFoundError(
            "找不到分词器文件：%s\n"
            "SFT 用的是**预训练时那份**分词器，先在 [data] 里指对 tokenizer_path。"
            % tok_path)
    tok = load_tokenizer(tok_path)

    template = (get(cfg, "post", "template", "plain") or "plain").lower()
    check_template_usable(template, tok)

    # 系统提示必须**训练和推理用同一份**：训练时把 post.system_prompt
    # 渲染进每条样本，chat 时 run_chat 也读同一个配置项。只训不推（或只推
    # 不训）会让模型看到没见过的前缀，表现为「答非所问」。
    system = get(cfg, "post", "system_prompt", "") or None
    loss_on_prompt = bool(get(cfg, "post", "loss_on_prompt", False))

    # ------------------------------------------------------------ 数据集
    ds_path = get(cfg, "post", "data_path", "") or ""
    if not ds_path:
        # 这里的示例路径用占位符，**不写开发者自己机器上的路径** ——
        # 别人照着抄会得到一个不存在的目录，然后以为是引擎的问题。
        # 顺便给出仓库里自带的那份示例数据（它是真实存在的，可以直接用）。
        sample = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "chat", "sample_chat.jsonl")
        raise ValueError(
            "没指定 SFT 数据集。在配置里加一行：\n"
            "  [post]\n"
            "  method    = sft\n"
            "  data_path = <你的数据集.jsonl>\n\n"
            + (("仓库里带了一份示例，想先跑通流程可以直接用：\n"
                "  --data \"%s\"\n\n" % sample) if os.path.isfile(sample) else "")
            + "数据集每行一条样本（JSONL），支持这几种写法：\n"
            '  {"messages": [{"role":"user","content":"问"},'
            '{"role":"assistant","content":"答"}]}\n'
            '  {"prompt": "问", "response": "答"}\n'
            '  {"instruction": "问", "input": "", "output": "答"}')
    rows = load_jsonl(ds_path)

    # ------------------------------------------------------------ 模型配置
    model_cfg = config_from_section(cfg.get("model", {}), tok.vocab_size)
    block_size = int(model_cfg["block_size"])

    max_seq_len = int(get(cfg, "post", "max_seq_len", 512) or 512)
    if max_seq_len > block_size:
        utils.warn("post.max_seq_len=%d 比 model.block_size=%d 还大，"
                   "按 block_size 处理（超过模型就看不见了）"
                   % (max_seq_len, block_size))
        max_seq_len = block_size

    utils.title("SFT 数据")
    utils.info("数据集        : %s" % ds_path)
    utils.info("样本条数      : %s" % "{:,}".format(len(rows)))
    utils.info("对话模板      : %s" % template)
    if system:
        utils.info("系统提示      : %s" % system.replace("\n", " ")[:60])
    if loss_on_prompt:
        utils.warn("loss_on_prompt=true：问题/格式标记也参与训练 —— "
                   "小数据下模型容易学会复述问题，确认这是你要的。")
    rendered = render_dataset(rows, tok, template=template,
                              max_seq_len=max_seq_len, system=system,
                              loss_on_prompt=loss_on_prompt)
    st = rendered["stats"]
    if st["n_samples"] == 0:
        raise ValueError(
            "渲染之后一条可用样本都不剩（原始 %d 条，丢掉了 %d 条）。\n原因：\n  %s\n"
            "最常见的两种：样本里没有 assistant 回复；或者所有样本都超过了 "
            "post.max_seq_len=%d。"
            % (len(rows), st["n_skipped"], "\n  ".join(rendered["skip_reasons"]),
               max_seq_len))
    utils.info("可用样本      : %s（丢掉 %s）"
               % ("{:,}".format(st["n_samples"]), "{:,}".format(st["n_skipped"])))
    for why in rendered["skip_reasons"]:
        utils.info("    - %s" % why)
    utils.info("平均长度      : %.1f token" % st["avg_len"])
    utils.info("回答占比      : %.0f%% 的 token 要学（其余是问题和格式标记）"
               % (st["supervised_ratio"] * 100))

    # 训练/验证切分。SFT 数据本来就少，留 2% 看有没有背题。
    all_s = rendered["samples"]
    n_val = max(1, int(len(all_s) * 0.02)) if len(all_s) >= 50 else 0
    val_s = all_s[:n_val]
    train_s = all_s[n_val:]
    utils.info("训练/验证切分 : %s / %s" % ("{:,}".format(len(train_s)),
                                            "{:,}".format(len(val_s))))

    ds_train = SFTDataset(train_s, block_size, device=device,
                          seed=int(get(cfg, "train", "seed", 1234)))
    ds_val = SFTDataset(val_s, block_size, device=device, seed=999) if val_s else None

    # ------------------------------------------------------------ 基础断点 / 续训
    utils.set_seed(int(get(cfg, "train", "seed", 1234)))
    model = build_model(model_cfg, device)
    model = accelerators.maybe_compile(model, device)
    _tcfg = cfg.get("train", {})
    model.grad_checkpoint = bool(_tcfg.get("grad_checkpoint", False))
    model.loss_chunk = int(_tcfg.get("loss_chunk", 0) or 0)
    if model.grad_checkpoint:
        utils.info("梯度检查点    : 开（用约 1.3 倍算力换 30~60% 激活显存）")
    if int(_tcfg.get("optim_bits", 32) or 32) == 8:
        utils.info("优化器        : AdamW 8-bit（动量分块量化，约省 4 倍）")

    resume_path = None
    if resume:
        out_dir_hint = get(cfg, "post", "out_dir", "out_sft") or "out_sft"
        if str(resume).lower() == "auto":
            cand = os.path.join(out_dir_hint, "ckpt_sft.pt")
            if not os.path.isfile(cand):
                raise FileNotFoundError(
                    "post.out_dir 下没有 ckpt_sft.pt，没法续训：%s\n"
                    "  想从头开始就加 --force；想续别的断点就 --resume <路径>"
                    % os.path.abspath(cand))
            resume_path = cand
        else:
            resume_path = os.path.abspath(str(resume))
            if not os.path.isfile(resume_path):
                raise FileNotFoundError("找不到要续训的 SFT 断点：%s" % resume_path)

    resume_ck = None
    if resume_path:
        utils.title("从 SFT 断点续训")
        resume_ck = load_checkpoint(resume_path, map_location=device)
        check_ckpt_tokenizer(resume_ck, tok, where="SFT 断点")
        old_cfg = resume_ck.get("config") or {}
        diffs, unchecked = structure_diffs(old_cfg, model_cfg)
        if diffs:
            utils.err("模型结构和 SFT 断点对不上，没法续训：")
            for label, a, b in diffs:
                utils.info("  %-16s 断点里是 %-8s，现在配置是 %s" % (label, a, b))
            raise ValueError("模型结构与 SFT 断点不一致")
        if unchecked:
            utils.warn("旧断点里有 %d 个字段无法核对（只能按同结构加载）：%s"
                       % (len(unchecked), "、".join(unchecked)))
        try:
            model.load_state_dict(resume_ck["model"])
        except RuntimeError as e:
            utils.err("加载 SFT 断点失败：%s" % e)
            utils.info("断点里记录的模型配置：")
            for k in ("n_layer", "d_model", "n_head", "d_ff", "vocab_size"):
                if old_cfg.get(k) is not None:
                    utils.info("  %-12s = %s" % (k, old_cfg[k]))
            raise ValueError("SFT 断点加载失败")
        utils.ok("已载入 SFT 断点：%s（第 %s 步）"
                 % (resume_path, "{:,}".format(int(resume_ck.get("step", 0)))))
    else:
        base, tried = find_base_checkpoint(cfg)
        utils.title("从哪里接着教")
        if base is None:
            utils.err("找不到预训练断点 —— SFT 必须有个预训练模型做起点。")
            utils.info("找过这些地方：")
            for label, p in tried:
                utils.info("  %-28s %s" % (label, p))
            print()
            utils.info("三个选择：")
            utils.info("  ① 先把预训练跑完： lingyu train --config configs/small.ini")
            utils.info("  ② 在 [post] 里明确指定： init_from = D:/.../ckpt.pt")
            utils.info("  ③ 临时指定： --set post.init_from=D:/.../best.pt")
            raise FileNotFoundError("找不到预训练断点")
        ck = _load_base_weights(model, base, device, tok)
        utils.ok("已载入 %s（第 %s 步的预训练权重）"
                 % (base, "{:,}".format(int(ck.get("step", 0)))))
    total_params = sum(p.numel() for p in model.parameters())
    utils.info("模型参数      : %s (%.2f M)" % ("{:,}".format(total_params),
                                                total_params / 1e6))

    # ------------------------------------------------------------ 优化器
    opt_cfg = {
        "learning_rate": float(get(cfg, "post", "learning_rate", 5e-4) or 5e-4),
        "min_lr": float(get(cfg, "post", "learning_rate", 5e-4) or 5e-4) * 0.1,
        "warmup_iters": max(1, int(get(cfg, "post", "max_steps", 2000) * 0.03)),
        "lr_decay_iters": int(get(cfg, "post", "max_steps", 2000)),
        "weight_decay": float(get(cfg, "train", "weight_decay", 0.1) or 0.1),
        "beta1": float(get(cfg, "train", "beta1", 0.9) or 0.9),
        "beta2": float(get(cfg, "train", "beta2", 0.95) or 0.95),
    }
    if opt_cfg["learning_rate"] > 2e-3:
        utils.warn("SFT 学习率 %.1e 偏大。预训练那种大学习率会把模型训崩"
                   "（loss 变低但忘光预训练知识），建议 1e-5 ~ 5e-4。"
                   % opt_cfg["learning_rate"])
    opt = configure_optimizer(model, opt_cfg)

    # 续训：把优化器动量 / 步数 / 最佳验证 loss / RNG 接上（否则等于换个种子重训）
    start_step = 0
    best_val = float("inf")
    if resume_ck is not None:
        if resume_ck.get("optimizer") is not None:
            try:
                opt.load_state_dict(resume_ck["optimizer"])
            except Exception as e:
                utils.warn("优化器状态没接上（%s），动量会重新开始" % e)
        start_step = int(resume_ck.get("step", 0))
        best_val = float(resume_ck.get("best_val", float("inf")))
        if _restore_rng(resume_ck.get("rng")):
            utils.info("已恢复随机数状态（续训可复现）")
        else:
            utils.warn("断点里没有 RNG 状态（旧格式），续训的数据顺序会和上次不同")

    batch_size = max(1, int(get(cfg, "post", "batch_size", 16) or 16))
    max_steps = max(1, int(get(cfg, "post", "max_steps", 2000) or 2000))
    grad_clip = float(get(cfg, "train", "grad_clip", 1.0) or 1.0)
    log_interval = max(1, int(get(cfg, "post", "log_interval", 10) or
                              get(cfg, "train", "log_interval", 10) or 10))
    eval_interval = max(1, int(get(cfg, "post", "eval_interval", 100) or
                               get(cfg, "train", "eval_interval", 100) or 100))
    ask_interval = max(1, int(get(cfg, "post", "ask_interval", 200) or 200))
    ckpt_interval = max(1, int(get(cfg, "post", "ckpt_interval", 200) or 200))

    set_config_path(config_path)
    out_dir = get(cfg, "post", "out_dir", "out_sft") or "out_sft"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt_sft.pt")
    if resume_ck is None and os.path.isfile(ckpt_path) and not force:
        utils.err("%s 里已经有 SFT 断点了。" % os.path.abspath(out_dir))
        utils.info("想接着训就加 --resume；想推倒重来就加 --force。")
        raise FileExistsError("已存在 SFT 断点")

    amp_dtype = torch.bfloat16 if use_cuda else None
    utils.title("开始 SFT")
    utils.info("输出目录      : %s" % os.path.abspath(out_dir))
    utils.info("批大小        : %d" % batch_size)
    utils.info("总步数        : %s" % "{:,}".format(max_steps))
    utils.info("学习率        : %.1e（预训练用的是 [train] learning_rate）"
               % opt_cfg["learning_rate"])
    utils.info("每 %d 步问它一句，看看到底会不会答" % ask_interval)
    if start_step:
        utils.info("续训起点      : 第 %s 步（总步数 %s）"
                   % ("{:,}".format(start_step), "{:,}".format(max_steps)))

    step, t_start = start_step, time.time()
    hist = []
    stop_seen = False
    model.train()

    # ---------------------------------------------------------------- 记录
    # GUI 监控页据此把「预训练 / SFT」两套数据分开（审计 1.2）。
    # ★ SFT 的 1 步 = batch_size 条问答样本，token/步 = batch_size × max_seq_len。
    #   以前没写 run.json / history.csv，监控页只能拿预训练的数字套上来，
    #   结果把 93k tok/s 配到回答部分 loss 上（用户实测撞到过）。
    utils.write_run_meta(
        out_dir, "sft",
        max_steps=int(max_steps), start_step=int(start_step),
        batch_size=int(batch_size), block_size=int(block_size),
        max_seq_len=int(max_seq_len),
        tokens_per_step=int(batch_size * max_seq_len),
        n_params=int(sum(p.numel() for p in model.parameters())),
        device=str(device), template=str(template), resume=start_step > 0,
        loss_on_prompt=bool(loss_on_prompt),
    )
    hist_path = os.path.join(out_dir, "history.csv")
    if start_step == 0 or not os.path.isfile(hist_path):
        with open(hist_path, "w", encoding="utf-8") as fp:
            fp.write("step,train_loss,val_loss,lr,grad_norm,elapsed_sec\n")
    utils.emit_event(
        "start", task="sft", config=config_path,
        max_steps=int(max_steps), start_step=int(start_step),
        batch_size=int(batch_size), block_size=int(block_size),
        max_seq_len=int(max_seq_len), template=str(template),
        device=str(device), resume=start_step > 0)
    last_hist_step = start_step
    lr, gnorm = float("nan"), 0.0   # 循环一次都没跑时收尾行也要能写

    try:
        # SFT 的取数方法叫 batch() 且返回 (idx, tgt, n_sup)，用 fetch 回调复用预取
        _prefetch_on = os.environ.get("LINGYU_PREFETCH", "1").strip().lower() \
            not in ("0", "false", "no", "off")
        batch_stream = (BatchPrefetcher(ds_train, batch_size,
                                        fetch=ds_train.batch)
                        if _prefetch_on else None)
        while step < max_steps:
            if utils.stop_requested():
                # 外部优雅停止协议：LINGYU_STOP_FILE 指向的文件出现 → 存断点后正常退出
                stop_seen = True
                utils.warn("[停止] 收到停止请求（%s），保存断点后正常退出……"
                           % utils.stop_file())
                break
            step += 1
            lr = lr_at(step, opt_cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            if batch_stream is not None:
                idx, tgt, n_sup = next(batch_stream)
            else:
                idx, tgt, n_sup = ds_train.batch(batch_size)
            if n_sup == 0:
                # 理论上到不了这儿（render_dataset 已经丢掉全 0 的样本），
                # 但万一到了：跳过这一步，否则 cross_entropy 会返回 NaN，
                # 一个 NaN 进梯度之后**所有**参数都会变成 NaN，训练彻底废掉。
                utils.warn("第 %d 步这一批一个要学的 token 都没有，已跳过" % step)
                continue
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss, _ = model(idx, targets=tgt)
            if not torch.isfinite(loss):
                utils.err("第 %d 步 loss 变成了 %s —— 训练已经坏了，停下。"
                          % (step, loss.detach().item()))
                utils.info("常见原因：学习率太大、数据里有空样本、或者数值溢出。")
                utils.info("先试着把 [post] learning_rate 调小一个数量级。")
                raise FloatingPointError("loss 非有限值")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            opt.step()
            hist.append(float(loss.detach()))

            if step % log_interval == 0:
                recent = hist[-log_interval:]
                el = time.time() - t_start
                avg_loss = sum(recent) / len(recent)
                utils.info("  第 %5d/%d 步  loss %.4f  lr %.2e  |g| %.2f  %s"
                           % (step, max_steps, avg_loss, lr,
                              gnorm, utils.fmt_duration(el)))
                utils.emit_event("step", task="sft", step=int(step),
                                 total=int(max_steps), loss=avg_loss,
                                 lr=float(lr), grad_norm=float(gnorm),
                                 elapsed=float(el))

            # --- 真的问它一句 ---
            if step % ask_interval == 0 or step == max_steps:
                print()
                utils.info("  ── 试问（第 %d 步）──" % step)
                for q in PROBE_QUESTIONS[:2]:
                    a = ask(model, tok, q, template, device,
                            max_new_tokens=48, system=system)
                    utils.info("    问：%s" % q)
                    utils.info("    答：%s" % (a.replace("\n", " ")[:120] or "（空）"))
                    utils.emit_event("probe", task="sft", step=int(step),
                                     question=str(q),
                                     answer=a.replace("\n", " ")[:400])
                print()

            # --- 验证 + 落 history.csv（监控页画曲线用）---
            if step % eval_interval == 0 or step == max_steps:
                v = float("nan")
                if ds_val is not None:
                    v = eval_sft_loss(model, ds_val, batch_size, 5, device, amp_dtype)
                    utils.info("  [验证] 第 %d 步  回答部分 loss %.4f" % (step, v))
                    utils.emit_event("eval", task="sft", step=int(step),
                                     val_loss=float(v),
                                     best=bool(v < best_val))
                    if v < best_val:
                        best_val = v
                        save_checkpoint(os.path.join(out_dir, "best_sft.pt"), model,
                                        opt, step, best_val, model_cfg,
                                        extra=ckpt_extra(tok, tok_path, template))
                        utils.emit_event("ckpt", task="sft", step=int(step),
                                         path=os.path.abspath(
                                             os.path.join(out_dir, "best_sft.pt")),
                                         best=True)
                        utils.info("           ^ 目前最好，已存 best_sft.pt")
                recent = hist[-log_interval:] if hist else []
                recent_mean = (sum(recent) / len(recent)
                               if recent else float("nan"))
                try:
                    with open(hist_path, "a", encoding="utf-8") as fp:
                        fp.write("%d,%.4f,%.4f,%.6f,%.4f,%.1f\n"
                                 % (step, recent_mean, v, lr, gnorm,
                                    time.time() - t_start))
                    last_hist_step = step
                except OSError:
                    pass

            if step % ckpt_interval == 0:
                save_checkpoint(ckpt_path, model, opt, step, best_val, model_cfg,
                                extra=ckpt_extra(tok, tok_path, template))
                utils.info("  [存档] 第 %d 步 → %s" % (step, ckpt_path))
                utils.emit_event("ckpt", task="sft", step=int(step),
                                 path=os.path.abspath(ckpt_path), best=False)

    except KeyboardInterrupt:
        utils.warn("收到 Ctrl+C，正在安全收尾……")
    finally:
        if batch_stream is not None:
            batch_stream.close()

    # 停止/中断可能发生在两次验证之间：补一行，别让监控页停在旧步数
    if step > last_hist_step:
        recent = hist[-log_interval:] if hist else []
        recent_mean = sum(recent) / len(recent) if recent else float("nan")
        try:
            with open(hist_path, "a", encoding="utf-8") as fp:
                fp.write("%d,%.4f,%.4f,%.6f,%.4f,%.1f\n"
                         % (step, recent_mean, float("nan"), lr, gnorm,
                            time.time() - t_start))
        except OSError:
            pass

    save_checkpoint(ckpt_path, model, opt, step, best_val, model_cfg,
                    extra=ckpt_extra(tok, tok_path, template))
    if stop_seen:
        utils.ok("[停止] 收到停止请求，已保存断点 %s" % os.path.abspath(ckpt_path))
    elapsed = time.time() - t_start
    utils.title("SFT 结束")
    utils.info("总步数        : %s" % "{:,}".format(step))
    utils.info("总耗时        : %s" % utils.fmt_duration(elapsed))
    if hist:
        utils.info("最后 20 步平均 loss: %.4f"
                   % (sum(hist[-20:]) / len(hist[-20:])))
    utils.info("断点          : %s" % os.path.abspath(ckpt_path))
    print()
    utils.info("下一步 —— 和它聊天：")
    utils.info("  lingyu chat --config <你刚才用的配置文件> --ckpt %s" % ckpt_path)
    utils.emit_event(
        "done", task="sft", steps=int(step), max_steps=int(max_steps),
        best_val=None if best_val == float("inf") else float(best_val),
        out_dir=os.path.abspath(out_dir), stopped=bool(stop_seen),
        elapsed=float(elapsed), ckpt=os.path.abspath(ckpt_path),
        n_samples=int(st["n_samples"]), template=str(template))
    return {"steps": step, "elapsed": elapsed, "ckpt": ckpt_path,
            "n_samples": st["n_samples"], "template": template}
