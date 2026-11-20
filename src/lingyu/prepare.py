# -*- coding: utf-8 -*-
'''数据准备：把一堆中文 txt 变成模型能直接读的二进制 token 文件。

完整流程（对应命令 `python -m lingyu prepare`）：

    1. 扫描目录里的文本文件
    2. 嗅探编码（UTF-8 / GBK / GB18030 / UTF-16）并读成统一文本
    3. 清洗：统一换行、去 BOM、去零宽字符、合并多余空行
    4. 语料体检：数一数有多少字符、多少种字、建议词表多大
    5. 训练分词器（默认抽取代表性子集，省时间）
    6. 用分词器把全部语料编码成 token，按比例切训练/验证
    7. 写成 train.bin / val.bin + meta

为什么编码要「流式」写盘：
    语料大了以后，「先全部编码到内存再写」会同时占住「文本 + token 列表」两份内存。
    这里一次只编码一块，边编边写，内存峰值只有一块的大小。
'''

import json
import os
import sys

from . import utils
from .tokenizer import train_tokenizer, inspect_corpus

TEXT_EXTS = (".txt", ".md", ".text", ".json", ".jsonl", ".ndjson")

# 这些扩展名不是「纯文本」，而是 JSON —— 读取时要先抽出正文再进语料
JSON_EXTS = (".json", ".jsonl", ".ndjson")

# 这些文件是我们自己放的说明文件，不是语料，扫描时要跳过
IGNORE_NAMES = ("放小说到这里.txt", "readme.txt", "readme.md", "说明.txt")


def find_text_files(raw_dir, extra_ignore=()):
    """找出目录里所有文本文件（支持子文件夹）。

    会跳过我们自己放的说明文件 —— 否则几十个字的说明会混进语料，
    把分词器的字符统计彻底带偏（实测：只放说明文件时统计出"122 种字符"）。
    """
    ignore = set(IGNORE_NAMES) | set(extra_ignore or ())
    out = []
    for root, _dirs, files in os.walk(raw_dir):
        for fn in sorted(files):
            if fn in ignore:
                continue
            if fn.startswith("."):
                continue
            if fn.lower().endswith(TEXT_EXTS):
                out.append(os.path.join(root, fn))
    return sorted(out)


# ---------------------------------------------------------------- 编码嗅探
def sniff_and_read(path):
    '''读文件并返回 (文本, 用的编码, 原始字节数)。

    中文小说最常见的坑：老的 GBK/GB18030 文件被按 UTF-8 读会乱码或直接报错。
    这里按「UTF-8 BOM → UTF-8 → GB18030 → UTF-16」的顺序试，
    GB18030 是 GBK 的超集，覆盖绝大多数中文老文件。
    '''
    with open(path, "rb") as fp:
        raw = fp.read()
    n_bytes = len(raw)

    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "UTF-8 (带BOM)", n_bytes
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16"), "UTF-16", n_bytes
        except UnicodeError:
            pass
    for enc in ("utf-8", "gb18030"):
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        bad = text.count("\ufffd")
        if bad > max(3, len(text) * 0.001):
            continue
        return text, enc, n_bytes
    return raw.decode("utf-8", errors="replace"), "UTF-8 (有损坏字节，已替换)", n_bytes


# ---------------------------------------------------------------- 清洗
_ZW = ["\u200b", "\u200c", "\u200d", "\ufeff"]



# ---------------------------------------------------------------- JSON 语料
# 常见的开源数据集是 JSON / JSONL：字段名五花八门，但正文基本落在下面这些键里。
# 只认这些键、并且跳过 id/url 之类的元数据 —— 否则 id 和 URL 会混进语料，
# 把分词器教成一堆随机字母。
_TEXT_KEYS = (
    "text", "content", "body", "article", "story", "novel", "chapter",
    "paragraph", "paragraphs", "instruction", "input", "output", "response",
    "answer", "question", "prompt", "completion", "title", "description",
    "transcript", "dialogue", "conversation", "messages", "sentence",
    "sentences", "summary", "value", "completion_text",
)
_SKIP_KEYS = (
    "id", "uuid", "url", "uri", "image", "img", "images", "audio", "video",
    "source", "license", "lang", "language", "timestamp", "date", "author",
    "name", "file", "filename", "path", "hash", "sha", "score", "label",
    "meta", "metadata", "config", "category", "role", "from", "turn", "index",
)


def _collect_all_strings(node, out, depth=0):
    """不挑键名，收集所有字符串 —— 前面挑正文字段的规则一条都没命中时用它兜底。

    现实里的 JSON 千奇百怪：有的把正文放在 `payload.body`，有的叫
    `article`，还有的干脆就是 {"a": "第一段", "b": "第二段"}。宁可把
    正文和一点元数据一起收进来，也好过因为「键名不认识」整份文件被跳过。
    """
    if depth > 12:
        return
    if isinstance(node, str):
        text = node.strip()
        if text:
            out.append(text)
    elif isinstance(node, dict):
        for k in node:
            _collect_all_strings(node[k], out, depth + 1)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _collect_all_strings(v, out, depth + 1)


def _collect_json_strings(node, out, depth=0):
    if depth > 8:
        return
    if isinstance(node, str):
        text = node.strip()
        if text:
            out.append(text)
    elif isinstance(node, dict):
        preferred = [v for k, v in node.items()
                     if isinstance(k, str) and k.lower() in _TEXT_KEYS]
        for v in preferred:
            _collect_json_strings(v, out, depth + 1)
        # 同一条记录里的其它字符串也收（例如 {"title": ..., "desc": ...}）。
        # 只过滤键名明显是元数据的**短标量**；容器里的内容一律往下走 ——
        # 正文可能藏在 {"meta": {"rows": [{"inner": {"paragraph": ...}}]}}。
        for k, v in node.items():
            if any(v is pv for pv in preferred):
                continue
            skip = isinstance(k, str) and k.lower() in _SKIP_KEYS
            if not isinstance(v, (dict, list, tuple)) and skip:
                continue
            _collect_json_strings(v, out, depth + 1)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _collect_json_strings(v, out, depth + 1)


#: 配置文件类 JSON 的顶层键 —— 命中就整份跳过，别把命令/依赖名当语料。
_CONFIG_JSON_KEYS = frozenset((
    "mcpservers", "dependencies", "devdependencies", "scripts",
    "compileroptions", "engines", "peerdependencies", "build", "contributes",
))
_CONFIG_JSON_NAMES = frozenset((
    "mcp.json", "package.json", "package-lock.json", "tsconfig.json",
    "settings.json", "launch.json", "tasks.json", "manifest.json",
    "composer.json", "pyproject.json",
))


def extract_json_text(text):
    """从 JSON / JSONL 文本里抽出可训练的正文。

    返回 (正文, 记录数)。坏行不报错：以 `{`/`[` 开头的行当损坏 JSON 跳过，
    其余行（用户可能把 txt 混进 jsonl）原样保留 —— 语料准备不该因为
    一个坏字节就整份失败。
    """
    raw = text.strip()
    if not raw:
        return "", 0

    # 整个文件是一个 JSON 对象/数组（普通 .json 与单行 jsonl 都走这里）
    try:
        node = json.loads(raw)
    except ValueError:
        node = None
    if node is not None:
        if isinstance(node, dict):
            top = {str(k).lower() for k in node}
            has_text = any(str(k).lower() in _TEXT_KEYS for k in node)
            # 顶层是 mcpServers/dependencies/scripts 这类配置键、且一个正文
            # 字段都没有 → 配置文件，整份跳过（B-07）。
            if (top & _CONFIG_JSON_KEYS) and not has_text:
                return "", 0
        out = []
        _collect_json_strings(node, out)
        if not out:
            _collect_all_strings(node, out)     # 挑字段没命中：全收兜底
        return "\n".join(out), 1

    # 逐行 JSONL
    out, records = [], 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            node = json.loads(line)
        except ValueError:
            head = line.lstrip()[:1]
            if head not in ("{", "["):
                out.append(line)      # 混进来的纯文本行，保留
            continue
        _collect_json_strings(node, out)
        records += 1
    return "\n".join(out), records


def _is_config_json(path, text):
    """像配置文件的 JSON：按文件名或顶层键判定。"""
    import os
    if os.path.basename(path).lower() in _CONFIG_JSON_NAMES:
        return True
    try:
        node = json.loads(text)
    except ValueError:
        return False
    if isinstance(node, dict):
        return bool({str(k).lower() for k in node} & _CONFIG_JSON_KEYS)
    return False


def read_corpus_file(path, json_mode="raw"):
    """读一个语料文件，返回 (正文, 编码, JSON 记录数或 None)。

    json_mode：
      - "raw"（默认）：.json/.jsonl 和 .txt/.md 一样**原样**进语料，
        保留 `{}`、键名、缩进等结构 —— 适合想教模型学 JSON 格式的语料。
      - "extract"：只抽 text/content/instruction/output/messages 等正文字段，
        跳过 id/url 等元数据（老行为）。
    """
    text, enc, _n_bytes = sniff_and_read(path)
    if path.lower().endswith(JSON_EXTS) and json_mode == "extract":
        import os as _os
        # 按文件名就能确定是配置文件的（package.json / mcp.json…）直接跳过，
        # 不看内容 —— 这类文件里也常混着零散字符串，抽出来全是噪声。
        if _os.path.basename(path).lower() in _CONFIG_JSON_NAMES:
            return "", enc + " / JSON 配置文件（不是语料）", 0
        extracted, records = extract_json_text(text)
        if not extracted.strip():
            # 其余没有正文字段的 JSON 不静默丢，按原文收（用户看到
            # 「跳过：没从 JSON 里抽出正文」会以为功能坏了）。
            return text, enc + " / JSON（无正文字段，按原文）", None
        return extracted, enc + " / JSON 提取", records
        return extracted, enc + " / JSON 提取", records
    return text, enc, None


def clean_text(text, do_clean=True):
    '''基础清洗。默认只做安全操作（不删正文）。'''
    if not do_clean:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for z in _ZW:
        text = text.replace(z, "")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text


AD_WORDS = ("请收藏本站", "最快更新", "无弹窗", "手机版阅读网址", "记住本站",
            "章节错误", "内容严重缺失", "下载本站", "笔趣阁", "首发地址")


def strip_ads(text):
    '''可选：粗暴去掉网址和常见广告串。默认不开。'''
    import re
    text = re.sub(r"(https?://|www\.)\S+", "", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "", text)
    for kw in AD_WORDS:
        text = text.replace(kw, "")
    return text


# ---------------------------------------------------------------- 流式编码
class SplitWriter:
    '''边编码边按比例写 train/val，内存里只放一块。'''

    def __init__(self, data_dir, vocab_size, total_tokens, val_ratio, meta_extra=None):
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        self.total_tokens = total_tokens
        self.split_at = int(total_tokens * (1.0 - val_ratio))
        self.seen = 0
        self.n_bytes = 0
        self.n_train = 0
        self.n_val = 0
        self.meta_extra = meta_extra or {}
        os.makedirs(data_dir, exist_ok=True)
        self.ftr = open(os.path.join(data_dir, "train.bin"), "wb")
        self.fva = open(os.path.join(data_dir, "val.bin"), "wb")
        self.dtype_name = "uint16" if vocab_size <= 65535 else "uint32"
        import numpy as np
        self.np_dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    def _write(self, is_train, ids):
        if not ids:
            return
        import numpy as np
        arr = np.asarray(ids, dtype=self.np_dtype)
        if is_train:
            self.ftr.write(arr.tobytes())
            self.n_train += arr.size
        else:
            self.fva.write(arr.tobytes())
            self.n_val += arr.size

    def add(self, text, tok):
        ids = tok.encode(text)
        end = self.seen + len(ids)
        if self.seen < self.split_at < end:
            cut = self.split_at - self.seen
            self._write(True, ids[:cut])
            self._write(False, ids[cut:])
        elif end <= self.split_at:
            self._write(True, ids)
        else:
            self._write(False, ids)
        self.seen = end
        self.n_bytes += len(text.encode("utf-8"))

    def close(self):
        self.ftr.close()
        self.fva.close()
        bpt = self.n_bytes / max(1, self.seen)
        for name, n in (("train", self.n_train), ("val", self.n_val)):
            meta = {"dtype": self.dtype_name, "n_tokens": int(n),
                    "vocab_size": int(self.vocab_size),
                    "n_bytes": int(round(bpt * n)),
                    "bytes_per_token": round(bpt, 4)}
            meta.update(self.meta_extra)
            with open(os.path.join(self.data_dir, name + ".meta.json"), "w",
                      encoding="utf-8") as fp:
                json.dump(meta, fp, ensure_ascii=False, indent=2)
        return self.n_train, self.n_val, bpt


def _contiguous_sample(corpus, limit_chars, n_blocks=16):
    '''从长语料里取「均匀分布的连续块」拼成子集。

    为什么不能跨步抽（corpus[::step]）：
      BPE 学的是「相邻字对」。跨步抽样会把原文里相隔很远的两个字
      变成邻居，学到的合并规则是伪的。实测同一份 966 万字语料：
      连续块子集在 held-out 文本上压缩比 1.438，跨步只有 1.160（-24%）
      ——「分词器好像训过了、实际没学到东西」。
    首块贴住语料开头，末块贴住结尾，块与块等距铺满全文，
    避免只学到开头几本书。
    '''
    n = len(corpus)
    if limit_chars <= 0 or limit_chars >= n:
        return corpus
    n_blocks = max(1, min(int(n_blocks), limit_chars))
    if n_blocks == 1:
        return corpus[:limit_chars]
    block_len = max(1, limit_chars // n_blocks)
    span = max(0, n - block_len)
    starts = [int(round(i * span / (n_blocks - 1))) for i in range(n_blocks)]
    return "".join(corpus[s:s + block_len] for s in starts)


# ---------------------------------------------------------------- 主流程
def _prepare_workers(n_files):
    '''并行读文件用几个线程。LINGYU_PREPARE_WORKERS=1 可强制串行（排查用）。'''
    import os as _os
    raw = _os.environ.get("LINGYU_PREPARE_WORKERS", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n > 0:
            return max(1, min(32, n, max(1, n_files)))
    return max(1, min(8, _os.cpu_count() or 1, max(1, n_files)))


def run_prepare(raw_dir, data_dir, tokenizer_type="bpe", vocab_size=None,
                val_ratio=0.02, tokenizer_sample_mb=3.0, do_clean=True,
                do_strip_ads=False, seed=1234, min_pair_count=2, max_token_chars=4,
                verbose=True, exclude=None, json_mode="raw"):
    '''完整的数据准备。'''
    utils.title("第 1 步 / 扫描文本文件")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError("语料目录不存在: %s" % raw_dir)
    files = find_text_files(raw_dir, extra_ignore=exclude)
    if not files:
        raise FileNotFoundError(
            "在 %s 里没找到任何文本文件（支持 %s）。\n把小说/语料放进这个目录再运行。"
            % (raw_dir, "、".join(TEXT_EXTS)))
    total_bytes = sum(os.path.getsize(f) for f in files)
    utils.ok("找到 %d 个文件，合计 %.2f MB" % (len(files), total_bytes / 1024 / 1024))
    # ★ 先给「按类型」的汇总：目录里混进几千个 md 时，用户最关心的是
    #   「我的 jsonl 到底算不算在内」，而不是逐文件清单的前 10 行。
    from collections import Counter as _Counter
    ext_counts = _Counter(os.path.splitext(f)[1].lower() or "(无扩展名)"
                          for f in files)
    utils.info("  按类型：" + "  ".join(
        "%s %d 个" % (e, n) for e, n in ext_counts.most_common()))
    json_files = [f for f in files
                  if os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".ndjson")]
    if json_files:
        utils.info("  JSON/JSONL（都会读）：")
        for f in json_files[:10]:
            utils.info("    %-46s %8.2f MB"
                       % (os.path.basename(f)[:44],
                          os.path.getsize(f) / 1024 / 1024))
        if len(json_files) > 10:
            utils.info("    …… 其余 %d 个" % (len(json_files) - 10))
    utils.info("  逐文件明细（前 10 个）：")
    for f in files[:10]:
        utils.info("    %-46s %8.2f MB"
                   % (os.path.basename(f)[:44], os.path.getsize(f) / 1024 / 1024))
    if len(files) > 10:
        utils.info("    …… 其余 %d 个文件不再逐个列出"
                   % (len(files) - 10))

    utils.title("第 2 步 / 读取与清洗（自动嗅探编码）")
    texts = []
    encodings = {}
    shown = 0
    skipped = 0
    # ★ 并行读文件：读取 + 解压/解码 + JSON 解析都在 Python 层，GIL 会在
    #   IO/解码时释放；线程池比进程池省去拷贝大字符串的开销。实测读目录
    #   190 MB（8,866 个文件）单线程 12.4s → 8 线程 3.1s（约 4×）。
    #   顺序由 pool.map 保证，最后清洗/拼接仍是单线程，结果与串行一致。
    workers = _prepare_workers(len(files))

    def _read_one(path):
        return path, read_corpus_file(path, json_mode=json_mode)

    if workers > 1 and len(files) > 1:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=workers) as _pool:
            pairs = list(_pool.map(_read_one, files))
    else:
        pairs = [_read_one(f) for f in files]
    for f, (text, enc, n_json) in pairs:
        if not text.strip():
            skipped += 1
            continue
        encodings[os.path.basename(f)] = enc
        if do_strip_ads:
            text = strip_ads(text)
        text = clean_text(text, do_clean=do_clean)
        shown += 1
        # ★ 日志降噪：8,000 多个文件逐个打印会把控制台/日志刷爆（实测扫全盘
        #   md 时一次一屏）。明细只打前 10 个和每 500 个一行，其余看汇总。
        if verbose and (shown <= 10 or shown % 500 == 0):
            if n_json is None:
                utils.info("  %-38s %-24s %s 字符"
                           % (os.path.basename(f)[:36], enc,
                              "{:,}".format(len(text))))
            else:
                utils.info("  %-38s %-24s %s 字符（%d 条记录）"
                           % (os.path.basename(f)[:36], enc,
                              "{:,}".format(len(text)), n_json))
        if not text.endswith("\n"):
            text += "\n"
        texts.append(text)
    if skipped:
        utils.warn("  跳过 %d 个文件（没读到正文：空文件，或 JSON 里没有文字）"
                   % skipped)
    if len(files) > 10:
        utils.info("  共 %d 个文件，明细只显示前 10 个和每 500 个一行"
                   % len(files))
    corpus = "".join(texts)
    del texts
    utils.ok("合并后语料: %s 字符" % "{:,}".format(len(corpus)))

    utils.title("第 3 步 / 语料体检")
    info = inspect_corpus(corpus)
    utils.info("字符数        : %s" % "{:,}".format(info["chars"]))
    utils.info("字节数        : %s" % "{:,}".format(info["bytes"]))
    utils.info("不同字符种类  : %s 种" % "{:,}".format(info["distinct_chars"]))
    utils.info("抽样二字组    : %s 种" % "{:,}".format(info["distinct_pairs_sampled"]))
    utils.info("出现最多的字  : %s" % "".join(ch for ch, _ in info["top_chars"][:20]))
    if info["non_printable_kinds"]:
        utils.warn("有 %d 种不可打印字符，清洗时已尽量处理" % info["non_printable_kinds"])

    # ---- 语料是否像个正经中文小说的粗略判断 ----
    # 实测教训：目录里混进一个几十字的说明文件，会把统计彻底带偏。
    if info["distinct_chars"] < 800:
        utils.warn("不同字符只有 %s 种 —— 正常中文小说至少有 2000~7000 种。"
                   % "{:,}".format(info["distinct_chars"]))
        utils.warn("  语料可能是：① 太短 ② 几乎全是英文 ③ 混进了说明/日志类小文件")
        utils.warn("  请检查上面的文件清单和大小。用 --exclude 文件名 可以排除某个文件。")
    total_mb = total_bytes / 1024 / 1024
    if total_mb < 1.0:
        utils.warn("语料只有 %.2f MB，偏少。中文小说建议至少 2~5 MB 起步。"
                   % total_mb)
    if info["chars"] > 0 and info["distinct_chars"] < 200:
        utils.warn("不同字符不到 200 种，这份语料八成不是中文小说。建议先确认目录里放对了文件。")

    if vocab_size is None:
        auto_vocab = True
        if tokenizer_type == "char":
            vocab_size = min(8192, max(512, info["distinct_chars"] + 8))
        else:
            vocab_size = 8192 if info["distinct_chars"] > 3000 else 4096
    else:
        auto_vocab = False
    # 词表太小会导致大量生僻字退化成「2-3 个字节 token」，压缩比不升反降。
    # 判断依据是「基础词表能装下多少种字符」：预算 = 词表 - 260（4 特殊 + 256 字节兜底）。
    # 覆盖率高（>=70%）就没问题；覆盖率低才警告。
    n_special = 4 + 256
    char_budget = max(0, vocab_size - n_special)
    coverage = min(1.0, char_budget / max(1, info["distinct_chars"]))
    if tokenizer_type != "char":
        utils.info("字符覆盖率    : %.0f%%（基础词表能装下 %s / %s 种字符）"
                   % (coverage * 100,
                      "{:,}".format(min(char_budget, info["distinct_chars"])),
                      "{:,}".format(info["distinct_chars"])))
    if tokenizer_type != "char" and coverage < 0.7:
        utils.warn("词表 %s 偏小：只能容纳 %.0f%% 的字符种类，"
                   % ("{:,}".format(vocab_size), coverage * 100))
        utils.warn("  剩下 %s 种低频字会退化成 2-3 个字节 token，压缩比不升反降。"
                   % "{:,}".format(int(info["distinct_chars"] * (1 - coverage))))
        utils.warn("  （实测：词表 2048 + 3945 种字符 → 覆盖率 39%，压缩比只有 1.365）")
        utils.warn("  建议至少 %s。现在用 %s 继续，效果可能不理想。"
                   % ("{:,}".format(int(info["distinct_chars"] / 0.7) + n_special),
                      "{:,}".format(vocab_size)))
    elif tokenizer_type != "char" and coverage < 0.95:
        utils.info("  提示: 覆盖率未到 100%，有少量低频字会退化成字节 token。"
                   "这对压缩比影响很小，可以不管。")
    utils.info("分词器类型    : %s" % ("字符级 char（一字一 token）" if tokenizer_type == "char"
                                     else "BPE 子词（高频字对合并成词）"))
    utils.info("词表大小      : %s%s"
               % ("{:,}".format(vocab_size),
                  "（自动选的）" if auto_vocab else ""))
    utils.info("embedding 代价: 约 %s 参数（词表 × 512 维，供参考）"
               % "{:,}".format(vocab_size * 512))

    utils.title("第 4 步 / 训练分词器")
    if tokenizer_type == "char":
        sample = corpus
    else:
        limit = int(tokenizer_sample_mb * 1024 * 1024)
        n_bytes = len(corpus.encode("utf-8"))
        if n_bytes > limit:
            # 字节配额换算成字符数（中文 UTF-8 约 3 字节/字）
            char_limit = max(1, int(len(corpus) * (limit / float(n_bytes))))
            n_blocks = 16
            sample = _contiguous_sample(corpus, char_limit, n_blocks=n_blocks)
            utils.info("BPE 训练用 %s / %s 字符的连续块子集"
                       "（%d 块等距铺满全文，保持原文相邻关系，避免只学到开头几本）"
                       % ("{:,}".format(len(sample)), "{:,}".format(len(corpus)),
                          n_blocks))
        else:
            sample = corpus
    tok = train_tokenizer(sample, vocab_size, kind=tokenizer_type, verbose=verbose,
                          min_pair_count=min_pair_count, max_token_chars=max_token_chars)
    tok.meta["source_files"] = list(encodings.keys())
    tok.meta["encodings"] = encodings
    tok_path = os.path.join(data_dir, "tokenizer.txt")
    tok.save(tok_path)
    utils.ok("分词器已存到 %s" % tok_path)
    if utils.json_mode():
        # JSON 模式下 stdout 只能有 JSON 事件；分词器描述改走 stderr。
        sys.stderr.write(tok.describe() + "\n")
    else:
        print(tok.describe())

    utils.title("第 5 步 / 编码全部语料并切分")
    # 先算总量，才能按比例切。语料大时这一步要几十秒，给个提示免得看着像卡死。
    est_sec = info["chars"] / 2_000_000.0      # 实测编码速度约 200 万字符/秒
    utils.info("正在编码 %s 个字符……（预计 %.0f 秒）"
               % ("{:,}".format(info["chars"]), est_sec))
    import time as _time
    _t0 = _time.time()
    n_tokens = len(tok.encode(corpus))
    utils.info("编码完成，用时 %.1f 秒（%.0f 万字符/秒）"
               % (_time.time() - _t0,
                  info["chars"] / max(1e-6, _time.time() - _t0) / 10000))
    comp = info["chars"] / max(1, n_tokens)
    utils.info("总 token 数   : %s" % "{:,}".format(n_tokens))
    utils.info("压缩比        : %.3f 字符/token%s"
               % (comp, "（字符级固定为 1.000）" if tokenizer_type == "char"
                  else "  ← 越大越省上下文"))
    utils.info("训练/验证切分 : %.1f%% / %.1f%%" % ((1 - val_ratio) * 100, val_ratio * 100))

    extra = {"tokenizer_type": tokenizer_type, "corpus_chars": info["chars"],
             "corpus_files": list(encodings.keys()), "compression": round(comp, 4),
             # ★ 词表指纹：让「语料 bin 和分词器是不是同一次 prepare 出来的」
             #   可以被机器核对。只比 vocab_size + 类型是不够的 ——
             #   特殊 token 数量一变，两边大小可能都是 8192，但每个字的 ID 都错位。
             "tokenizer_fingerprint": tok.fingerprint()}
    w = SplitWriter(data_dir, tok.vocab_size, n_tokens, val_ratio, meta_extra=extra)
    chunk = 1_000_000
    for i in range(0, len(corpus), chunk):
        w.add(corpus[i:i + chunk], tok)
    n_train, n_val, bpt = w.close()
    utils.ok("train.bin : %s token（%.2f MB）"
             % ("{:,}".format(n_train),
                os.path.getsize(os.path.join(data_dir, "train.bin")) / 1024 / 1024))
    utils.ok("val.bin   : %s token（%.2f MB）"
             % ("{:,}".format(n_val),
                os.path.getsize(os.path.join(data_dir, "val.bin")) / 1024 / 1024))

    preview = os.path.join(data_dir, "corpus_preview.txt")
    with open(preview, "w", encoding="utf-8") as fp:
        fp.write(corpus[:200000])

    utils.title("数据准备完成")
    utils.info("分词器        : %s" % tok_path)
    utils.info("训练数据      : %s" % os.path.join(data_dir, "train.bin"))
    utils.info("验证数据      : %s" % os.path.join(data_dir, "val.bin"))
    utils.info("抽样预览      : %s（建议打开看看清洗有没有误伤正文）" % preview)
    utils.info("平均字节/token: %.3f（算「每字节多少 bit」要用它）" % bpt)
    utils.info("词表指纹      : %s（训练时会核对，防止语料和分词器配错）"
               % tok.fingerprint())
    print()
    utils.ok("下一步:  python -m lingyu plan --config configs/small.ini")
    return {
        "tokenizer_path": tok_path, "data_dir": data_dir,
        "vocab_size": tok.vocab_size, "n_tokens": n_tokens,
        "n_train": n_train, "n_val": n_val,
        "bytes_per_token": bpt, "compression": comp,
    }
