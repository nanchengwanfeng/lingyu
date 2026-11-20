# -*- coding: utf-8 -*-
'''数据加载：把 token 序列存成二进制文件，训练时用内存映射随机取片段。

为什么要存成二进制而不是每次现编码：
    编码 11 MB 中文要几十秒，训练要跑几万步，每次现编完全不可接受。
    预先把 token id 写成 .bin，训练时 mmap 直接按偏移读，速度接近内存访问。

文件格式（比 nanoGPT 多一个 meta 文件，用来记住 dtype 和长度）：
    train.bin        token id 数组，uint16 或 uint32
    train.meta.json  {"dtype": "uint16", "n_tokens": 1234567, ...}
不写 meta 也能用：nanoGPT 的 train.bin/val.bin 会自动按 uint16 处理。
'''

import json
import os

import numpy as np
import torch

DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


def pick_dtype(vocab_size):
    '''按词表大小挑最小的存储类型，省内存也省磁盘。'''
    if vocab_size <= 65535:
        return "uint16"
    return "uint32"


def _meta_path(bin_path):
    return os.path.splitext(bin_path)[0] + ".meta.json"


def write_tokens(path, ids, vocab_size, extra=None):
    '''把 token 序列写成二进制 + meta。返回写出的字节数。'''
    dtype_name = pick_dtype(vocab_size)
    arr = np.asarray(ids, dtype=DTYPES[dtype_name])
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    arr.tofile(path)
    meta = {
        "dtype": dtype_name,
        "n_tokens": int(arr.size),
        "vocab_size": int(vocab_size),
    }
    if extra:
        meta.update(extra)
    with open(_meta_path(path), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    return int(arr.nbytes)


def read_meta(bin_path):
    '''读 meta；没有就按老格式（uint16）推断。'''
    mp = _meta_path(bin_path)
    if os.path.isfile(mp):
        with open(mp, "r", encoding="utf-8") as fp:
            return json.load(fp)
    if not os.path.isfile(bin_path):
        raise FileNotFoundError("找不到数据文件: " + bin_path)
    n = os.path.getsize(bin_path) // 2
    return {"dtype": "uint16", "n_tokens": int(n), "vocab_size": None,
            "assumed": "无 meta 文件，按 nanoGPT 老格式 uint16 处理"}


def check_data_tokenizer(data_dir, tok, strict=True):
    """检查「语料 bin」和「分词器」是不是同一次 prepare 出来的。

    ## 为什么必须查（这是会静默毁掉训练的坑）

    token id 只是**词表里的下标**。语料 bin 和分词器如果不是同一次
    prepare 的产物，同一串 id 解码出来就是完全不同的字：

        data/  的 bin 里的 id 1234 -> "的"
        data1/ 的词表里 id 1234    -> "剑"

    而两边的 vocab_size 都是 8192，**形状完全对得上**。于是：

        - 不报任何错
        - loss 照常下降（模型在学一套自洽但错误的映射）
        - 生成的文字是乱码，你还以为是模型没训好

    这是本项目里最危险的一类 bug，因为它不崩、只是悄悄毁掉你几小时的电费。
    所以这里直接**拒绝开跑**，而不是打个警告。

    meta 文件缺失（老数据目录）时只警告 —— 那种情况下没法判断，
    不能因为查不了就不让人训练。
    """
    meta_path = os.path.join(data_dir, "train.meta.json")
    if not os.path.isfile(meta_path):
        return ("没有 %s，无法核对语料与分词器是否配套。"
                "如果你换过分词器，请重新跑 prepare。" % os.path.basename(meta_path))
    try:
        with open(meta_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except Exception as e:
        return "读不了 %s（%s），跳过配套检查" % (meta_path, e)

    problems = []
    mv = meta.get("vocab_size")
    if mv is not None and int(mv) != int(tok.vocab_size):
        problems.append("语料词表大小 %s，分词器词表大小 %s"
                        % (mv, tok.vocab_size))
    mt = meta.get("tokenizer_type")
    if mt and mt != tok.kind:
        problems.append("语料是 %s 分词器，你给的是 %s 分词器"
                        % (mt, tok.kind))

    # ★ 指纹核对：这是唯一能发现「词表大小一样、但 ID 分配不同」的检查。
    #   例子：老分词器 4 个特殊 token、新分词器 6 个 —— 两边都报 8192，
    #   可比对大小永远看不出问题，而每个汉字的 ID 都差了 2。
    mf = meta.get("tokenizer_fingerprint")
    if mf:
        tf = tok.fingerprint()
        if mf != tf:
            problems.append(
                "词表指纹不一致：语料是 %s，当前分词器是 %s\n"
                "    （这说明语料 bin 和分词器不是同一次 prepare 的产物，"
                "两边的 token ID 对不上）" % (mf, tf))

    if problems:
        msg = ("语料和分词器不配套：\n    " + "\n    ".join(problems) +
               "\n\n这两个文件必须来自**同一次** prepare。用同一个 --data 目录重新准备：\n"
               "    python -m lingyu prepare --raw <语料目录> --data %s\n\n"
               "为什么必须拦下来：token id 只是词表下标，两边形状对得上但不配套时\n"
               "训练不会报错、loss 照样下降，可解码出来全是乱码。" % data_dir)
        if strict:
            raise ValueError(msg)
        return msg
    return None


class TokenDataset:
    '''内存映射的 token 数据集。

    它不真的「装」数据，只是记住文件在哪；每次取样本时按随机偏移读
    block_size+1 个 token（多要一个用来当「下一个 token」的答案）。
    '''

    def __init__(self, bin_path, block_size, device="cpu", seed=1234):
        self.path = bin_path
        self.meta = read_meta(bin_path)
        self.dtype = DTYPES[self.meta["dtype"]]
        self.n_tokens = int(self.meta["n_tokens"])
        self.block_size = block_size
        self.device = device
        if self.n_tokens < block_size + 2:
            raise ValueError(
                "数据太短了：只有 %d 个 token，但 block_size=%d，至少需要 %d 个。"
                "要么加语料，要么把 block_size 调小。"
                % (self.n_tokens, block_size, block_size + 2))
        self.data = np.memmap(bin_path, dtype=self.dtype, mode="r")
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_tokens

    def get_batch(self, batch_size):
        '''随机取一批训练样本，返回 (x, y)。

        x 和 y 错开一位：x 是「看到的」，y 是「要预测的下一个」。
        '''
        ix = self.rng.integers(0, self.n_tokens - self.block_size - 1, size=batch_size)
        # 向量化取窗口：旧写法逐条切片再 np.stack，batch 大时纯 Python 开销明显。
        offs = np.arange(self.block_size + 1, dtype=np.int64)
        win = np.asarray(self.data[ix[:, None] + offs])
        x = win[:, :-1].astype(np.int64)
        y = win[:, 1:].astype(np.int64)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.device != "cpu":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        return x, y

    def first_tokens(self, n):
        '''取开头 n 个 token（固定片段，用于可复现的验证）。'''
        n = min(n, self.n_tokens)
        return np.asarray(self.data[:n]).astype(np.int64)

    def describe(self):
        lines = []
        lines.append("  文件          : %s" % self.path)
        lines.append("  大小          : %.2f MB" % (os.path.getsize(self.path) / 1024 / 1024))
        lines.append("  token 数      : %s" % "{:,}".format(self.n_tokens))
        lines.append("  存储类型      : %s" % self.meta["dtype"])
        if self.meta.get("assumed"):
            lines.append("  注意          : %s" % self.meta["assumed"])
        if self.meta.get("bytes_per_token"):
            lines.append("  平均字节/token: %.3f" % self.meta["bytes_per_token"])
        return "\n".join(lines)


class BatchPrefetcher:
    '''后台线程预取训练 batch，让「取数据」和「GPU 算」重叠。

    为什么值得做：`get_batch` 要随机取 block、np.stack、astype、pin_memory、
    再拷到 GPU。对 47.8M 小模型，一批 batch=16×512 的前向反向约 0.19s，
    而取数据约 8~15ms（约 5%）；大 batch 或 CPU 弱时占比更高。这个类用
    `queue.Queue(maxsize=prefetch)` 做双缓冲：GPU 在算第 n 步时，worker
    线程已经把第 n+1 批准备好，主线程 `next()` 直接拿到（通常不阻塞）。

    线程/流安全：worker 每次只调 `dataset.get_batch`（内部用数据集自己的
    RNG，且只在 worker 线程里调），非阻塞拷贝与训练都在默认 CUDA 流上排队，
    顺序天然有保证。关闭时 `close()` 幂等，可放在 finally 里。

    关掉：环境变量 LINGYU_PREFETCH=0（train 侧判断），或别用它。
    '''

    def __init__(self, dataset, batch_size, prefetch=2, fetch=None):
        import queue as _queue
        import threading as _threading
        self.dataset = dataset
        self.batch_size = batch_size
        # fetch 可选：SFT 数据集的取数方法是 batch() 且返回三元组，
        # 不叫 get_batch；传一个可调用对象即可复用同一套预取逻辑。
        self._fetch = fetch
        self._q = _queue.Queue(maxsize=max(1, int(prefetch)))
        self._stop = _threading.Event()
        self._err = None
        self._closed = False
        self._thread = _threading.Thread(
            target=self._run, name="lingyu-batch-prefetch", daemon=True)
        self._thread.start()

    def _run(self):
        import queue as _queue
        fetch = self._fetch if self._fetch is not None else self.dataset.get_batch
        try:
            while not self._stop.is_set():
                batch = fetch(self.batch_size)
                while not self._stop.is_set():
                    try:
                        self._q.put(batch, timeout=0.2)
                        break
                    except _queue.Full:
                        continue
        except Exception as exc:           # 把 worker 里的错带回主线程
            self._err = exc

    def __iter__(self):
        return self

    def __next__(self):
        import queue as _queue
        while True:
            if self._err is not None:
                raise RuntimeError("预取线程出错：%s" % self._err) from self._err
            if self._closed and self._q.empty():
                raise StopIteration
            try:
                return self._q.get(timeout=0.5)
            except _queue.Empty:
                if self._stop.is_set():
                    if self._q.empty():
                        raise StopIteration
                    continue
                continue

    next = __next__

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self._thread.join(timeout=3.0)
        except Exception:
            pass
        # 丢掉队列里剩余 batch，避免持有显存
        while True:
            try:
                self._q.get_nowait()
            except Exception:
                break

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def eval_loss(model, dataset, batch_size, iters, device, amp_dtype=None,
              block_size=None, eval_batch=None):
    '''在验证集上算平均 loss（不更新参数），返回 (平均 loss, 参与评估的 token 数)。

    - 上下文长度用 block_size（不是 batch_size），评估整段完整上下文，
      否则 loss 会被「只看 8 个 token」严重高估。
    - 每块取 block_size+1 个 token、滑窗不重叠（最后一块可短），
      按 token 数加权平均；iters>0 时最多评估 iters 块。
    - **一次前向并行 eval_batch 块**：块之间互不重叠，但注意力是因果的，
      每块只看自己块内的 token，所以批量推理与逐块推理在数学上等价；
      短块用 target=-1 补齐（模型交叉熵本来就按有效 token 求均值），
      按有效 token 数加权后与逐块结果一致。GPU 上这是评估提速的主要来源。
    - 结束时恢复进入前的 model.training 状态（不再无条件 model.train()）。
    '''
    was_training = model.training
    model.eval()
    n = dataset.n_tokens
    if n < 3:
        if was_training:
            model.train()
        return float("nan"), 0
    ctx = int(block_size or getattr(dataset, "block_size", 0) or batch_size or 8)
    ctx = max(2, min(ctx, n - 1))
    max_chunks = int(iters) if iters and int(iters) > 0 else 0
    # 序列越长，单批能塞的块越少；8 是实测最优点（block 512 时 8 块最快，
    # 16 块反而变慢：SDPA 的 batch 维到一定大小后收益饱和，调度开销上升）。
    if eval_batch is None:
        eval_batch = max(1, min(8, 8192 // max(ctx + 1, 1)))
    eval_batch = max(1, int(eval_batch))
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    loss_sum = 0.0
    total = 0
    chunks = 0
    start = 0
    with torch.no_grad():
        while start + 2 <= n:
            rows = []
            weights = []
            # 攒一批块（最后一块可能短，统一补齐）
            while len(rows) < eval_batch and start + 2 <= n:
                end = min(start + ctx + 1, n)
                chunk = np.asarray(dataset.data[start:end]).astype(np.int64)
                if chunk.size < 3:
                    break
                rows.append(chunk)
                weights.append(int(chunk.size - 1))
                start += ctx
                chunks += 1
                if max_chunks and chunks >= max_chunks:
                    break
            if not rows:
                break
            max_len = max(r.size for r in rows) - 1
            xb = np.zeros((len(rows), max_len), dtype=np.int64)
            yb = np.full((len(rows), max_len), -1, dtype=np.int64)
            for i, row in enumerate(rows):
                ln = row.size - 1
                xb[i, :ln] = row[:-1]
                yb[i, :ln] = row[1:]
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)
            with torch.autocast(device_type=dev_type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                _, loss, _ = model(x, y, collect_aux=False)
            w = sum(weights)                 # 这一批实际贡献的预测 token 数
            loss_sum += float(loss.item()) * w
            total += w
            if max_chunks and chunks >= max_chunks:
                break
    if was_training:
        model.train()
    if total <= 0:
        return float("nan"), 0
    return loss_sum / float(total), total


def perplexity(loss):
    import math
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def bits_per_byte(loss, n_tokens, n_bytes):
    '''每字节的比特数 —— 跨分词器比较模型好坏的正确指标。

    困惑度（PPL）依赖分词器：同一个模型换个词表，PPL 就变了。
    BPB 把 loss 折算成「原文每个字节需要多少 bit」，和分词器无关，可以公平比较。
    '''
    import math
    if n_tokens <= 0 or n_bytes <= 0:
        return float("nan")
    return loss / math.log(2) * (float(n_tokens) / float(n_bytes))
