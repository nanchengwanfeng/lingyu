# -*- coding: utf-8 -*-
"""分词器：把中文文本变成 token 数字，再变回来。纯 Python 实现，零依赖。

为什么要自己写而不是用现成的？
    1. 用别人的分词器，你永远不知道模型内部在干什么；
    2. 自己写才能实现「词表想多大就多大」，而小模型的词表大小直接决定
       embedding 层吃掉多少参数（词表 × 维度 × 2）。
    3. 纯 Python 意味着没有编译依赖，装在哪儿都能跑。

两种方案（下面用同一句中文举例）：

    原文: 大型语言模型正在改变世界          （12 个字）

    字符级 char —— 一个汉字 / 一个标点 = 一个 token
        大 | 型 | 语 | 言 | 模 | 型 | 正 | 在 | 改 | 变 | 世 | 界      12 个 token
        词表约 6000，压缩比 1.0

    BPE（字节级）—— 高频相邻片段自动合并成一个 token
        大 | 型 | 语言 | 模型 | 正在 | 改变 | 世界                      7 个 token
        词表约 8192，压缩比约 1.6

    压缩比的意义：同样 1024 的上下文长度，BPE 能塞进近两倍的文字。

底层一律走 UTF-8 字节（byte-level）：任何文本都不会出现「不认识的字符」，
解码一定能还原成原文，绝不会乱码。
"""

import json
import os
import time
import unicodedata
from collections import Counter

# 特殊 token：占用最前面的几个 ID。
#
# ⚠️ 这个列表**只能往后追加，不能插队、不能重排**。
# 词表 ID 就是 token 的身份，往中间插一个会让所有既有语料和断点全部错位。
#
# <|im_start|> / <|im_end|> 是 ChatML 的角色分隔符（Qwen 等也用这套）。
# 放进基础词表而不是等 SFT 时再加，是因为「预训练时就见过这两个 token」
# 比「SFT 时新加两行随机初始化的 embedding」效果好得多。
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>",
                  "<|im_start|>", "<|im_end|>"]
IM_START, IM_END = 4, 5

BYTE_ESCAPES = {
    0x5C: "\\\\",   # 反斜杠自己要先转义，否则解码歧义
    0x0A: "\\n",
    0x0D: "\\r",
    0x09: "\\t",
    0x20: "\\s",
}


def byte_to_token(b):
    """一个字节 -> 一个可打印的权名（存进词表文件后还能读得懂）。"""
    if 33 <= b <= 126:
        return chr(b)
    if b in BYTE_ESCAPES:
        return BYTE_ESCAPES[b]
    return "<0x%02X>" % b


def token_to_byte(tok):
    """权名 -> 字节。"""
    if tok.startswith("<0x") and tok.endswith(">"):
        return int(tok[3:-1], 16)
    mapping = {"\\\\": 0x5C, "\\n": 0x0A, "\\r": 0x0D, "\\t": 0x09, "\\s": 0x20}
    if tok in mapping:
        return mapping[tok]
    if len(tok) == 1:
        return ord(tok)
    raise ValueError("无法识别的词表条目: " + repr(tok))


def name_to_bytes(name):
    """词表名 -> 字节串（可能是多个字节拼成的 token）。"""
    try:
        return bytes([token_to_byte(name)])
    except ValueError:
        pass
    if name.startswith("<0x"):
        # 形如 <0xE4><0xB8><0xAD> 的拼接
        out = bytearray()
        i = 0
        while i < len(name):
            if name[i] == "<":
                j = name.find(">", i)
                if j == -1:
                    raise ValueError("词表条目坏掉了: " + repr(name))
                out.append(int(name[i + 3:j], 16))
                i = j + 1
            else:
                out.append(ord(name[i]))
                i += 1
        return bytes(out)
    return name.encode("utf-8")


# --- 词表文件里的转义 ---
# 词表按「token<TAB>id<TAB>次数」写盘，但 token 本身可能就是一个制表符、
# 换行符或反斜杠（中文语料里换行和制表很常见）。不转义就会把列切碎，
# 读回来直接报 int('') 失败 —— 这是实测踩过的坑。
#
# ★ `#` 也必须转义，而且这个坑比上面那个**严重得多**。
#
#   读盘时用「行首是不是 #」来识别注释。于是当词表里有一个 token 就是
#   `#` 本身时，它那一行 `#\t57\t123` **长得完全像一条注释**，
#   被静默跳过 —— 词表少一个 token。
#
#   后果不是「少一个字符认不出来」这么轻：
#     · 读回来的词表比写下去时少一项 → `fingerprint()` 变了
#     · `check_data_tokenizer` 看到指纹对不上 → **拒绝开训**
#     · 于是「prepare 完立刻 train」这条最基本的路径**永远走不通**
#
#   实测就是它把 5 步冒烟测试挡下来的：语料记的是 5ba29c265c7797f1，
#   读回来的分词器算出 070211f0f01f563b。
#   而中文书里出现 `#`（章节号、Markdown、网址残留）太正常了。
#
#   转义成 `\#` 之后，行首永远是反斜杠，注释判定不会再误伤。
_ESCAPE_MAP = {"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r",
               "#": "\\#"}
_UNESCAPE_MAP = {"\\": "\\", "t": "\t", "n": "\n", "r": "\r", "#": "#"}


def _escape_token(tok):
    return "".join(_ESCAPE_MAP.get(ch, ch) for ch in tok)


def _unescape_token(tok):
    if "\\" not in tok:
        return tok
    out = []
    i = 0
    while i < len(tok):
        ch = tok[i]
        if ch == "\\" and i + 1 < len(tok):
            nxt = tok[i + 1]
            out.append(_UNESCAPE_MAP.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# ==========================================================================


class BaseTokenizer:
    """两种分词器的共同接口。"""

    kind = "base"

    def __init__(self, vocab=None, counts=None, merges=None, meta=None):
        self.vocab = vocab or {}          # token(字符串) -> id
        self.id_to_token = {}             # id -> token
        self.counts = counts or {}        # token -> 出现次数（用于展示与排序）
        self.merges = merges or []        # BPE 合并规则，按学习顺序，元素是 bytes
        self.rank = {}                    # (bytes, bytes) -> 合并优先级，越小越先合并
        self.meta = meta or {}
        self._rebuild_reverse()

    # ---------------------------------------------------------- 基本属性
    def _rebuild_reverse(self):
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.rank = {m: i for i, m in enumerate(self.merges)}

    @property
    def vocab_size(self):
        return max(self.vocab.values()) + 1 if self.vocab else len(SPECIAL_TOKENS)

    def __len__(self):
        return self.vocab_size

    # ---------------------------------------------------------- 子类实现
    def train(self, text, vocab_size, verbose=True):
        raise NotImplementedError

    def _tokenize_word(self, word):
        raise NotImplementedError

    def _encode_text(self, text):
        raise NotImplementedError

    # ---------------------------------------------------------- 编码 / 解码
    def encode(self, text, add_bos=False, add_eos=False):
        """文本 -> token id 列表。

        切分规则（两种分词器共用）：按句切开，句内连续 ASCII 串当成一个整体
        交给 BPE 合并，其它字符逐个处理。
        """
        ids = []
        if add_bos:
            ids.append(BOS)
        vocab = self.vocab
        append = ids.append
        for tok in self._encode_text(text):
            append(vocab.get(tok, UNK))
        if add_eos:
            ids.append(EOS)
        return ids

    def _encode_text(self, text):
        """子类实现：文本 -> 一串 token 名字。"""
        raise NotImplementedError

    def encode_stream(self, chunks, total_chars=None, report_every=2_000_000,
                      label="编码", quiet=False):
        """流式编码：边读边吐 id，适合几百 MB 的语料，不会把内存撑爆。

        chunks 是一个可迭代对象，每次产出一段 str。
        """
        out = []
        n_chars = 0
        n_tokens = 0
        t0 = time.time()
        for chunk in chunks:
            ids = self.encode(chunk)
            n_chars += len(chunk)
            n_tokens += len(ids)
            out.extend(ids)
            if not quiet and n_chars >= report_every:
                sp = n_chars / max(1e-9, time.time() - t0)
                pct = ""
                if total_chars:
                    pct = "  %.1f%%" % (100.0 * n_chars / total_chars)
                print("    %s: %s 字符 -> %s token  (%.0f 字符/秒)%s" % (
                    label, "{:,}".format(n_chars), "{:,}".format(n_tokens), sp, pct))
        return out, n_chars, n_tokens

    def decode(self, ids):
        """token id 列表 -> 文本。

        两种分词器的 token 都是字符串本身，所以直接拼起来即可；
        单字节兜底 token（<0xNN>）要还原成字节再解码。
        """
        out = []
        buf = bytearray()

        def flush():
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                del buf[:]

        for i in ids:
            if i in (PAD, BOS, EOS):
                continue
            tok = self.id_to_token.get(i)
            if tok is None or tok == "<unk>":
                continue
            if len(tok) == 1:
                flush()
                out.append(tok)
            else:
                try:
                    buf.append(token_to_byte(tok))
                except ValueError:
                    flush()
                    out.append(tok)
        flush()
        return "".join(out)

    # ---------------------------------------------------------- 存取
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            head = {
                "kind": self.kind,
                "vocab_size": self.vocab_size,
                "special_tokens": SPECIAL_TOKENS,
                "meta": self.meta,
            }
            fp.write("# 灵语分词器  JSON 头\n")
            fp.write(json.dumps(head, ensure_ascii=False) + "\n")
            fp.write("# 词表（每行: token<TAB>id<TAB>出现次数；token 里的制表符写成 \\t、换行写成 \\n、反斜杠写成 \\\\）\n")
            for tok in sorted(self.vocab, key=lambda t: self.vocab[t]):
                cnt = self.counts.get(tok, 0)
                fp.write("%s\t%d\t%d\n" % (_escape_token(tok), self.vocab[tok], cnt))
            if self.merges:
                fp.write("# 合并规则（按学习顺序，每行: 左<TAB>右）\n")
                for a, b in self.merges:
                    fp.write("%s\t%s\n" % (_escape_token(a), _escape_token(b)))

    # ---------------------------------------------------------- 指纹
    def fingerprint(self):
        """词表指纹：同一份分词器永远得到同一个值，改一个 ID 就变。

        ## 为什么需要它（这是会静默毁掉训练的一类 bug）

        `check_data_tokenizer` 原本只比对**词表大小**和**分词器类型**。
        但这两样都太弱了：

            data1/tokenizer.txt  （旧：4 个特殊 token，正文从 ID 4 开始）
            新 prepare 出来的     （新：6 个特殊 token，正文从 ID 6 开始）

        两边都是 `bpe`、都是 `8192`，**怎么比都一样**，可每一个汉字的 ID
        都差了 2。拿旧语料喂新词表：不报错、loss 照常下降、生成全是乱码。

        所以要把「ID 分配」本身也纳入核对 —— 词表和合并规则一起做哈希。
        BPE 的合并规则也必须算进去：词表一样但合并顺序不同，
        同一句话会被切成不同的 token 串。

        返回 16 位十六进制字符串（够短，能写进 meta 给人看）。
        """
        import hashlib
        h = hashlib.sha256()
        h.update(self.kind.encode("utf-8"))
        # 按 ID 排序，保证与文件里的书写顺序无关
        for tok, vid in sorted(self.vocab.items(), key=lambda kv: kv[1]):
            h.update(("%d\x1f%s\x1e" % (vid, tok)).encode("utf-8"))
        for a, b in self.merges:
            h.update(("\x1f%s\x1f%s\x1e" % (a, b)).encode("utf-8"))
        return h.hexdigest()[:16]

    @classmethod
    def load(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError("找不到分词器文件: " + path)
        vocab, counts, merges, meta, kind = {}, {}, [], {}, None
        with open(path, "r", encoding="utf-8") as fp:
            first = fp.readline()
            if not first.startswith("#"):
                raise ValueError("分词器文件格式不对，第一行应该是 # 开头的注释")
            head = json.loads(fp.readline())
            kind = head.get("kind")
            meta = head.get("meta", {})
            mode = "vocab"
            for line in fp:
                if line.endswith("\n"):
                    line = line[:-1]
                if line.endswith("\r"):
                    line = line[:-1]
                if not line:
                    continue
                if line.startswith("#"):
                    # ★ 兼容**旧文件**：词表里真的有一个 token 就是 `#` 时，
                    #   它那一行 `#\t57\t123` 长得和注释一模一样，会被跳过。
                    #   新写法会把它转义成 `\#`（见 _ESCAPE_MAP），但老文件
                    #   没有转义 —— 用「第二列是不是纯数字」把数据行认出来。
                    #   不认的话旧分词器读回来少一个 token，指纹跟着变，
                    #   训练被 `check_data_tokenizer` 直接拦下，而用户
                    #   完全看不出为什么「昨天还好好的」。
                    legacy = line.split("\t")
                    if not (len(legacy) >= 2 and legacy[1].strip().isdigit()):
                        if "合并规则" in line:
                            mode = "merges"
                        continue
                parts = line.split("\t")
                if mode == "vocab":
                    if len(parts) >= 2:
                        tok = _unescape_token(parts[0])
                        try:
                            vid = int(parts[1])
                        except ValueError:
                            raise ValueError(
                                "分词器文件坏了：第 %r 行解析不出 token 编号。"
                                "（通常是因为文件被别的程序改过或截断）" % line[:80])
                        vocab[tok] = vid
                        counts[tok] = int(parts[2]) if len(parts) > 2 and parts[2] else 0
                else:
                    if len(parts) >= 2:
                        merges.append((_unescape_token(parts[0]),
                                       _unescape_token(parts[1])))
        if kind == "char":
            return CharTokenizer(vocab=vocab, counts=counts, meta=meta)
        return BPETokenizer(vocab=vocab, counts=counts, merges=merges, meta=meta)

    # ---------------------------------------------------------- 展示
    def describe(self):
        lines = []
        lines.append("  类型            : %s" % ("字符级 (char)" if self.kind == "char" else "字节级 BPE"))
        lines.append("  词表大小        : %s" % "{:,}".format(self.vocab_size))
        lines.append("  合并规则数      : %s" % "{:,}".format(len(self.merges)))
        if self.meta:
            if "train_chars" in self.meta:
                lines.append("  训练语料字符数  : %s" % "{:,}".format(self.meta["train_chars"]))
            if "compression" in self.meta:
                lines.append("  压缩比          : %.3f 字符/token" % self.meta["compression"])
            if "distinct_chars" in self.meta:
                lines.append("  语料中不同字符  : %s 种" % "{:,}".format(self.meta["distinct_chars"]))
        return "\n".join(lines)


# ==========================================================================


class CharTokenizer(BaseTokenizer):
    """字符级：一个汉字 = 一个 token。

    优点：直观，一眼能看懂模型在干什么；实现最短，最不容易出错。
    缺点：压缩比 1.0，同样上下文能装的文字最少；生僻字会落到 <unk>。
    """

    kind = "char"

    def train(self, text, vocab_size, verbose=True, min_count=1):
        counter = Counter(text)
        distinct = len(counter)
        if verbose:
            print("    语料不同字符种类: %s 种" % "{:,}".format(distinct))
        keep = [(ch, n) for ch, n in counter.most_common() if n >= min_count and ch not in ("\x00",)]
        budget = max(1, vocab_size - len(SPECIAL_TOKENS))
        if len(keep) > budget:
            if verbose:
                dropped = keep[budget:]
                print("    词表预算 %s，按词频截断，丢弃 %s 种低频字符（它们会变成 <unk>）"
                      % ("{:,}".format(budget), "{:,}".format(len(dropped))))
            keep = keep[:budget]
        self.vocab = {}
        self.counts = {}
        for i, t in enumerate(SPECIAL_TOKENS):
            self.vocab[t] = i
            self.counts[t] = 0
        for j, (ch, n) in enumerate(keep):
            self.vocab[ch] = len(SPECIAL_TOKENS) + j
            self.counts[ch] = n
        self.merges = []
        self._rebuild_reverse()
        self.meta = {
            "distinct_chars": distinct,
            "train_chars": len(text),
            "compression": 1.0,
            "min_count": min_count,
        }
        if verbose:
            if len(keep) < budget:
                print("    注意: 语料只出现了 %s 种字符，词表实际只用到 %s，"
                      "剩下的额度浪费了（可以把 vocab_size 调小）"
                      % ("{:,}".format(len(keep)), "{:,}".format(self.vocab_size)))
            unknown = sum(n for ch, n in counter.items() if ch not in self.vocab)
            if unknown:
                print("    注意: 有 %s 个字符（共出现 %s 次）不在词表里，会被当成 <unk>"
                      % ("{:,}".format(distinct - len(keep)), "{:,}".format(unknown)))
        return self

    def _encode_text(self, text):
        for ch in text:
            yield ch if ch in self.vocab else "<unk>"

    def _tokenize_word(self, word):
        # 字符级：一个单元就是一个字符
        return (word,) if word in self.vocab else ("<unk>",)


# ==========================================================================


class BPETokenizer(BaseTokenizer):
    """字符级 BPE：以「字符」为起点，反复把最高频的相邻字符对合并成新 token。

    为什么不从字节开始（GPT-2 那种字节级 BPE）？
        中文一个汉字占 3 个 UTF-8 字节，字节对的边界和汉字边界对不齐，
        合并出来的多半是「半个字」，压缩比反而比字符级更差（实测 0.61 token/字符）。
        从字符开始，合并结果天然是「字、词」，这才是中文该有的样子，
        也是 SentencePiece 处理中文的做法。

    算法：反向索引 + 懒惰堆
        教科书版每轮重扫全串（O(n^2)），几 MB 语料要跑几小时。
        这里用双向链表 + 堆，每轮只处理真正出现过的候选，几 MB 只需几秒。
        堆里可能有过期条目，弹出来时用「位置集合」当权威计数校验，
        计数对不上就按正确计数重新入堆 —— 这样结果和朴素实现完全一致。
    """

    kind = "bpe"

    def __init__(self, vocab=None, counts=None, merges=None, meta=None):
        super().__init__(vocab=vocab, counts=counts, merges=merges, meta=meta)
        self._cache = {}

    # ------------------------------------------------------------ 训练
    def train(self, text, vocab_size, verbose=True, report_every=200, min_pair_count=2,
              max_token_chars=4):
        """训练 BPE（v2：按脚本类型预切分，只在片段内部合并）。返回 self。

        min_pair_count  ：出现次数低于这个值的对不合并（滤掉只出现一次的巧合）。
        max_token_chars ：一个 token 最多由几个汉字/字符合成（cjk 段）。
                          字母数字段（latin）单独放宽到整词长度，上限 24 字符；
                          标点、空白永不合并。
        """
        import heapq

        # 1) 切句，再切成同类片段
        pieces = []
        for s in split_units(text):
            pieces.extend(split_pieces(s))
        if not pieces:
            raise ValueError("训练分词器的语料是空的")

        # 2) 统计字符频次，建基础词表
        #    v2 连字母数字也统计进来 —— v1 把 ASCII 串当整体跳过，导致英文
        #    单词只能退化成字节 token（这是「分词器不智能」的另一半原因）。
        char_freq = Counter()
        for _cls, p in pieces:
            for ch in p:
                char_freq[ch] += 1
        if not char_freq:
            raise ValueError("训练分词器的语料切完是空的")

        self.vocab = {}
        self.counts = {}
        for i, t in enumerate(SPECIAL_TOKENS):
            self.vocab[t] = i
            self.counts[t] = 0

        distinct = sorted(char_freq.keys(), key=lambda c: (-char_freq[c], c))
        if verbose:
            print("    片段数          : %s" % "{:,}".format(len(pieces)))
            print("    不同字符种类    : %s 种" % "{:,}".format(len(distinct)))

        # 基础词表预算：至少给 BPE 合并留一些空间
        room = max(0, vocab_size - len(self.vocab))
        base_budget = max(1, min(len(distinct), max(1, room - max(64, room // 4))))
        for ch in distinct[:base_budget]:
            self.vocab[ch] = len(self.vocab)
            self.counts[ch] = char_freq[ch]

        # 生僻字兜底：单字节 token，保证任何文本都能编码，绝不出现 <unk>
        for b in range(256):
            tok = byte_to_token(b)
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
                self.counts[tok] = 0

        if verbose:
            print("    基础词表        : %s 个（%s 个字符 + 256 字节兜底 + %s 特殊 token）"
                  % ("{:,}".format(len(self.vocab)), "{:,}".format(base_budget),
                     len(SPECIAL_TOKENS)))
            if len(distinct) > base_budget:
                print("    注意: 有 %s 种低频字没进基础词表，它们会退化成字节 token"
                      % "{:,}".format(len(distinct) - base_budget))

        # 3) 摊平成链表：片段内部相邻，片段之间断开
        base = []
        limits = []            # 每个节点所属片段的 token 长度上限
        cuts = set()           # 片段最后一个节点的位置（在那里断开）
        piece_kinds = Counter()
        for cls, p in pieces:
            if not p:
                continue
            piece_kinds[cls] += 1
            if cls == "cjk":
                lim = max(2, int(max_token_chars))
            elif cls == "latin":
                lim = min(24, max(8, len(p)))
            else:
                lim = 1        # 标点/空白：单字片段，永不合并
            base.extend(p)
            limits.extend([lim] * len(p))
            cuts.add(len(base) - 1)

        m = len(base)
        if m < 2:
            raise ValueError("训练分词器的语料太短了（至少需要 2 个字符）")
        left = list(range(-1, m - 1))
        right = list(range(1, m + 1))
        right[m - 1] = -1
        alive = bytearray(bytes([1]) * m)
        for b in cuts:
            right[b] = -1
        seq = list(base)
        tok_len = [1] * m
        nxt = m

        def bump(table, a, b, p):
            s = table.get((a, b))
            if s is None:
                s = set()
                table[(a, b)] = s
            s.add(p)
            return len(s)

        pair_pos = {}
        for i in range(m - 1):
            if right[i] != -1 and limits[i] > 1:
                bump(pair_pos, seq[i], seq[i + 1], i)
        heap = [(-len(s), a, b) for (a, b), s in pair_pos.items()]
        heapq.heapify(heap)

        budget = max(0, vocab_size - len(self.vocab))
        if verbose:
            print("    链表节点数      : %s" % "{:,}".format(m))
            print("    计划学合并规则  : %s 条" % "{:,}".format(budget))

        t0 = time.time()
        learned = 0
        while heap and learned < budget:
            neg, a, b = heapq.heappop(heap)
            s = pair_pos.get((a, b))
            if not s:
                continue
            valid = [i for i in s if alive[i] and right[i] != -1
                     and alive[right[i]] and seq[i] == a and seq[right[i]] == b]
            if not valid:
                del pair_pos[(a, b)]
                continue
            if len(valid) < min_pair_count:
                del pair_pos[(a, b)]
                continue
            i = valid[0]
            j = right[i]
            if tok_len[i] + tok_len[j] > limits[i]:
                del pair_pos[(a, b)]
                continue
            if -neg != len(s):
                heapq.heappush(heap, (-len(s), a, b))
                continue

            new_text = a + b
            merged = 0
            for i in valid:
                j = right[i]
                if not (alive[i] and j != -1 and alive[j] and seq[i] == a and seq[j] == b):
                    continue
                li, rj = left[i], right[j]
                k = nxt
                nxt += 1
                seq.append(new_text)
                tok_len.append(tok_len[i] + tok_len[j])
                limits.append(limits[i])
                left.append(li)
                right.append(rj)
                alive.append(1)
                alive[i] = 0
                alive[j] = 0
                if li != -1:
                    right[li] = k
                    old = pair_pos.get((seq[li], a))
                    if old is not None:
                        old.discard(i)
                    cnt = bump(pair_pos, seq[li], new_text, k)
                    heapq.heappush(heap, (-cnt, seq[li], new_text))
                if rj != -1:
                    left[rj] = k
                    old = pair_pos.get((b, seq[rj]))
                    if old is not None:
                        old.discard(j)
                    cnt = bump(pair_pos, new_text, seq[rj], k)
                    heapq.heappush(heap, (-cnt, new_text, seq[rj]))
                merged += 1

            self.merges.append((a, b))
            self.counts[new_text] = merged
            self.vocab[new_text] = len(self.vocab)
            learned += 1
            if verbose and learned % report_every == 0:
                print("    已学合并 %s/%s 条  最新 token %-8r  频率 %-6s  用时 %.0f 秒"
                      % ("{:,}".format(learned), "{:,}".format(budget),
                         new_text, "{:,}".format(merged), time.time() - t0))

        if verbose:
            print("    实际学到合并规则: %s 条，最终词表 %s，用时 %.0f 秒"
                  % ("{:,}".format(learned), "{:,}".format(self.vocab_size), time.time() - t0))
            if learned < budget:
                print("    注意: 没能用满词表预算（可合并的对不够了）。这是正常的——"
                      "语料越小能学到的合并越少。要么把 vocab_size 调小，要么加语料。")

        self._rebuild_reverse()
        self._cache = {}
        self.meta = {
            "train_chars": len(text),
            "distinct_chars": len(distinct),
            "base_units": m,
            "min_pair_count": min_pair_count,
            "max_token_chars": max_token_chars,
            "pretokenize": "script_v2",
            "pieces": sum(piece_kinds.values()),
            "piece_kinds": dict(piece_kinds),
        }
        sample = text[: min(len(text), 200000)]
        ids_s = self.encode(sample)
        if ids_s:
            self.meta["compression"] = len(sample) / float(len(ids_s))
        return self

    # ------------------------------------------------------------ 编码
    def _encode_text(self, text):
        """编码入口：新分词器走 v2（脚本感知预切分），老分词器走 v1 旧逻辑。

        老分词器（meta 里没有 pretokenize）必须继续能加载、能编码 ——
        data1/ 等旧语料和断点依赖它们，不能因为升级就报错。
        """
        if self.meta.get("pretokenize") == "script_v2":
            return self._encode_text_v2(text)
        return self._encode_text_legacy(text)

    def _emit_token_or_bytes(self, piece):
        if piece in self.vocab:
            yield piece
            return
        for byte in piece.encode("utf-8"):
            yield byte_to_token(byte)

    def _bpe_piece(self, piece, limit):
        """在单个片段内部做全局贪心 BPE，返回 token 名字元组（带缓存）。"""
        import heapq
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        rank = self.rank
        vocab = self.vocab
        n = len(piece)
        left = list(range(-1, n - 1))
        right = list(range(1, n + 1))
        right[n - 1] = -1
        alive = bytearray(bytes([1]) * n)
        parts = list(piece)
        nxt = n
        head = 0
        heap = []
        for i in range(n - 1):
            r = rank.get((parts[i], parts[i + 1]))
            if r is not None and len(parts[i]) + len(parts[i + 1]) <= limit:
                heap.append((r, i, parts[i], parts[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, a, b = heapq.heappop(heap)
            j = right[i] if alive[i] else -1
            if j == -1 or not alive[j]:
                continue
            cur_a, cur_b = parts[i], parts[j]
            if cur_a != a or cur_b != b:
                continue
            cur_r = rank.get((cur_a, cur_b))
            if cur_r != r:
                if cur_r is not None:
                    heapq.heappush(heap, (cur_r, i, cur_a, cur_b))
                continue
            if len(cur_a) + len(cur_b) > limit:
                continue
            li, rj = left[i], right[j]
            k = nxt
            nxt += 1
            parts.append(cur_a + cur_b)
            left.append(li)
            right.append(rj)
            alive.append(1)
            alive[i] = 0
            alive[j] = 0
            if li != -1:
                right[li] = k
                rr = rank.get((parts[li], parts[k]))
                if rr is not None:
                    heapq.heappush(heap, (rr, li, parts[li], parts[k]))
            else:
                head = k
            if rj != -1:
                left[rj] = k
                rr = rank.get((parts[k], parts[rj]))
                if rr is not None:
                    heapq.heappush(heap, (rr, k, parts[k], parts[rj]))

        out = []
        i = head
        while i != -1:
            if alive[i]:
                p = parts[i]
                if p in vocab:
                    out.append(p)
                else:
                    for byte in p.encode("utf-8"):
                        out.append(byte_to_token(byte))
            i = right[i]
        names = tuple(out)
        if len(self._cache) < 300000:
            self._cache[piece] = names
        return names

    def _encode_text_v2(self, text):
        """v2 编码：按片段类型分别合并，跨片段的合并不存在。"""
        limit = int(self.meta.get("max_token_chars", 4) or 4)
        for s in split_units(text):
            if not s:
                continue
            for cls, piece in split_pieces(s):
                if not piece:
                    continue
                if cls in ("punct", "space"):
                    for tok in self._emit_token_or_bytes(piece):
                        yield tok
                    continue
                lim = limit if cls == "cjk" else min(24, max(8, len(piece)))
                for tok in self._bpe_piece(piece, lim):
                    yield tok

    def _encode_text_legacy(self, text):
        """v1 旧编码（老分词器兼容）：句内整句合并。"""
        import heapq

        rank = self.rank
        vocab = self.vocab
        limit = self.meta.get("max_token_chars", 4)

        for s in split_units(text):
            if not s:
                continue
            n = len(s)
            if n == 1:
                p = s
                if p in vocab:
                    yield p
                else:
                    for byte in p.encode("utf-8"):
                        yield byte_to_token(byte)
                continue

            left = list(range(-1, n - 1))
            right = list(range(1, n + 1))
            right[n - 1] = -1
            alive = bytearray(bytes([1]) * n)
            parts = list(s)
            nxt = n
            head = 0

            heap = []
            for i in range(n - 1):
                r = rank.get((parts[i], parts[i + 1]))
                if r is not None:
                    heap.append((r, i, parts[i], parts[i + 1]))
            heapq.heapify(heap)

            while heap:
                r, i, a, b = heapq.heappop(heap)
                j = right[i] if alive[i] else -1
                if j == -1 or not alive[j]:
                    continue
                cur_a, cur_b = parts[i], parts[j]
                if cur_a != a or cur_b != b:
                    continue
                cur_r = rank.get((cur_a, cur_b))
                if cur_r != r:
                    if cur_r is not None:
                        heapq.heappush(heap, (cur_r, i, cur_a, cur_b))
                    continue
                if len(cur_a) + len(cur_b) > limit:
                    continue

                li, rj = left[i], right[j]
                k = nxt
                nxt += 1
                parts.append(cur_a + cur_b)
                left.append(li)
                right.append(rj)
                alive.append(1)
                alive[i] = 0
                alive[j] = 0
                if li != -1:
                    right[li] = k
                    rr = rank.get((parts[li], parts[k]))
                    if rr is not None:
                        heapq.heappush(heap, (rr, li, parts[li], parts[k]))
                else:
                    head = k
                if rj != -1:
                    left[rj] = k
                    rr = rank.get((parts[k], parts[rj]))
                    if rr is not None:
                        heapq.heappush(heap, (rr, k, parts[k], parts[rj]))

            i = head
            while i != -1:
                if alive[i]:
                    p = parts[i]
                    if p in vocab:
                        yield p
                    else:
                        for byte in p.encode("utf-8"):
                            yield byte_to_token(byte)
                i = right[i]

    def _tokenize_word(self, word):
        """word 是一个 ASCII 串，或一个非 ASCII 字符。"""
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        parts = list(word)
        rank = self.rank
        limit = self.meta.get("max_token_chars", 4)
        while len(parts) > 1:
            best_i = -1
            best_r = None
            for i in range(len(parts) - 1):
                # 编码阶段同样遵守「一个 token 最多几个字」，两次切分才完全一致
                if len(parts[i]) + len(parts[i + 1]) > limit:
                    continue
                r = rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_r = r
                    best_i = i
            if best_i < 0:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        out = []
        for p in parts:
            if p in self.vocab:
                out.append(p)
            else:
                # 生僻字：退化成单字节 token（一定能编码，绝不 <unk>）
                for byte in p.encode("utf-8"):
                    out.append(byte_to_token(byte))
        names = tuple(out)
        if len(self._cache) < 200000:
            self._cache[word] = names
        return names


def bytes_to_token_name(raw):
    """字节串 -> 词表里的名字。多字节的用可读形式拼接。"""
    if len(raw) == 1:
        return byte_to_token(raw[0])
    return "".join(byte_to_token(b) for b in raw)


_BOUNDARY = None


def split_units(text):
    """把文本切成「句」，每句内部的字符序列是连续的。

    为什么按句子切而不是按字符切：
        BPE 的合并只能发生在相邻单元之间。如果每个汉字都是一个独立单元，
        就没有任何一对是相邻的，一条合并也学不出来（这是实测踩过的坑）。
        按句子切之后，句内「他/抬/起/头」彼此相邻，BPE 才能把
        「抬起」「起来」这类高频组合合并成词。

    切分规则：中文句读标点、换行处断句，边界字符保留在句尾。
    """
    global _BOUNDARY
    if _BOUNDARY is None:
        import re
        # 在句末标点（。！？；…）或换行处断句；末尾没标点的残句也算一句
        _BOUNDARY = re.compile(
            "[^。！？；…\n\r]*[。！？；…\n\r]+"
            "|[^。！？；…\n\r]+"
        )
    if not text:
        return []
    return _BOUNDARY.findall(text)


def iter_chars(text):
    """逐字迭代（字符级分词器用）。"""
    for ch in text:
        yield ch


_ATOM_RE = None


def split_to_atoms(sentence):
    """把句子切成 BPE 的原子单元：

      - 连续 ASCII 可打印字符算一个原子（英文单词、数字不拆）
      - 每个非 ASCII 字符（汉字、中文标点）各自算一个原子

    BPE 只会在同一个原子内部合并。为什么这样定：
        训练和编码必须用完全一样的切分规则，否则学到的词一个也用不上。
        （实测踩过的坑：训练时按整句合并，编码时却按单字切，结果
          「没有」「声音」这些学到的词全部失效，压缩比永远是 1.000。）
    """
    global _ATOM_RE
    if _ATOM_RE is None:
        import re
        _ATOM_RE = re.compile("[\x21-\x7e]+|[^\x21-\x7e]")
    return _ATOM_RE.findall(sentence)


def iter_atoms(text):
    """整段文本 -> 原子序列（同时给出原子边界）。"""
    atoms = []
    for s in split_units(text):
        atoms.extend(split_to_atoms(s))
    return atoms


# ==========================================================================




# ==========================================================================
# v2：按「脚本类型」预切分（pretokenization）
# ==========================================================================
#
# 为什么需要这一层：
#   v1 把一整句连成字符链再合并，标点、空白、中英文彼此相邻，
#   于是会造出「他抬起头，」「AI 技」这种跨边界的 token —— 既占词表，
#   又让模型把标点当成词的一部分学。
#
# v2 先把句子切成同类片段：
#   cjk   汉字 / 假名 / 谚文连续段      —— 片段内合并，最多 max_token_chars 个字
#   latin 字母 / 数字连续段              —— 片段内合并，允许合到整个词（上限 24 字符）
#   punct 标点 / 符号，每个字符一片     —— 永不合并
#   space 空白，每个字符一片            —— 永不合并
#
# 训练和编码必须用同一套切分（这是 v1 文档里反复强调过的那条铁律）。

_CJK_RANGES = (
    (0x3400, 0x4DBF),      # 扩展 A
    (0x4E00, 0x9FFF),      # 基本区
    (0xF900, 0xFAFF),      # 兼容表意
    (0x3040, 0x30FF),      # 平假名 + 片假名
    (0xAC00, 0xD7AF),      # 谚文
    (0x20000, 0x2FA1F),    # 扩展 B~F
)


def char_kind(ch):
    """单个字符属于哪一类片段。"""
    cp = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return "cjk"
    if ch.isspace():
        return "space"
    cat = unicodedata.category(ch)
    if cat[0] == "L" or cat == "Nd":
        return "latin"
    return "punct"


def split_pieces(text):
    """把一段文本切成 (类别, 片段) 列表；标点/空白每字符单独一片。"""
    out = []
    buf = []
    cls = None
    for ch in text:
        k = char_kind(ch)
        if k in ("punct", "space"):
            if buf:
                out.append((cls, "".join(buf)))
                buf = []
                cls = None
            out.append((k, ch))
            continue
        if cls is None or k == cls:
            buf.append(ch)
            cls = k
        else:
            out.append((cls, "".join(buf)))
            buf = [ch]
            cls = k
    if buf:
        out.append((cls, "".join(buf)))
    # 空白并入后一个拉丁词：生成「前导空格」风格（" quick"），
    # 这样英文/代码的空格不会变成独立 token；中文之间不粘（中文不用空格分词）。
    merged = []
    i = 0
    while i < len(out):
        kind, piece = out[i]
        if kind == "space" and i + 1 < len(out) and out[i + 1][0] == "latin":
            nk, npiece = out[i + 1]
            merged.append((nk, piece + npiece))
            i += 2
            continue
        merged.append((kind, piece))
        i += 1
    return merged

def train_tokenizer(text, vocab_size, kind="bpe", verbose=True, min_pair_count=2,
                    max_token_chars=4):
    """按类型训练一个分词器。"""
    kind = (kind or "bpe").lower()
    if kind in ("char", "character", "字符"):
        tok = CharTokenizer()
        tok.train(text, vocab_size, verbose=verbose)
        return tok
    if kind in ("bpe", "byte", "字节"):
        tok = BPETokenizer()
        tok.train(text, vocab_size, verbose=verbose, min_pair_count=min_pair_count,
                  max_token_chars=max_token_chars)
        return tok
    raise ValueError("不认识的分词器类型: %s（可选 char 或 bpe）" % kind)


def load_tokenizer(path):
    return BaseTokenizer.load(path)


def inspect_corpus(text, sample_chars=200000):
    """先体检语料，再决定词表多大 —— 这一步能避免「词表设太大训不满」的坑。"""
    chars = Counter(text)
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    sample = text[:sample_chars]
    pairs = set()
    for i in range(len(sample) - 1):
        pairs.add(sample[i:i + 2])
    non_print = sum(1 for ch in chars if not ch.isprintable() and ch not in "\n\r\t")
    return {
        "chars": n_chars,
        "bytes": n_bytes,
        "distinct_chars": len(chars),
        "distinct_pairs_sampled": len(pairs),
        "sample_chars": len(sample),
        "non_printable_kinds": non_print,
        "top_chars": chars.most_common(20),
    }
