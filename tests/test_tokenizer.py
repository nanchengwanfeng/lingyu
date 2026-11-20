# -*- coding: utf-8 -*-
"""分词器自测：中文压缩比、往返一致性、存取一致性、编码速度。纯 Python，不需要 torch。"""

import os
import sys
import time

# 测试脚本在 tests\ 子目录，引擎包在上一级
# 引擎包在 src/ 下。pytest 已通过 pyproject.toml 的 pythonpath=["src"] 配好，
# 这里再插一次是为了支持直接 python tests/xxx.py 单跑。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from lingyu.tokenizer import (CharTokenizer, BPETokenizer, load_tokenizer,
                              BOS, EOS, inspect_corpus)

FAIL = []


def check(name, cond, detail=""):
    print("  %-36s %s %s" % (name, "[通过]" if cond else "[失败]", detail))
    if not cond:
        FAIL.append(name)


def make_corpus(n_repeat=30):
    """造一段「像小说」的语料：用字多样、有重复结构，能真正喂饱 BPE。"""
    parts = [
        "他抬起头，看着窗外的雨，忽然想起了很多年前的那个下午。",
        "天色已经暗了下来，远处传来隐约的雷声，风把树叶吹得沙沙作响。",
        "她轻声说道：“我们走吧，再晚就来不及了。”他没有回答，只是点了点头。",
        "少年的手紧紧握着那把剑，指节因为用力而变得发白，手臂上青筋暴起。",
        "江湖上的人都说，那把剑的主人从来不会输，可是没有人见过他真正的样子。",
        "雨越下越大，街道上已经没有了行人，只有昏黄的灯光在水面上摇晃。",
        "“你为什么要救我？”她问。他沉默了很久，才慢慢开口，声音沙哑得厉害。",
        "很多事情并不是因为有了答案才去坚持，而是坚持了才会有答案。",
        "山道崎岖，碎石在脚下滚动，每走一步都要付出极大的力气。",
        "他想起师父临终前说过的话，那句话像一根刺，扎在心里整整十年。",
        "长安城的春天来得总是很早，护城河边的柳树抽出嫩绿的新芽。",
        "酒楼二层靠窗的位置坐着一位白衣公子，手里端着一盏茶，目光落在街上。",
        "剑光一闪，空气中响起清脆的金属交击声，两人各自退开三步。",
        "夜里的风从山谷中穿过，带着一股潮湿的泥土气息，还有淡淡的血腥味。",
        "“我们还能回得去吗？”她望着远方的城门，声音轻得几乎听不见。",
        "他忽然笑了，那笑容里带着几分苦涩，几分释然，还有一丝说不清的温柔。",
        "书页翻动的声音在安静的书房里格外清晰，烛火摇晃，映出墙上斑驳的影子。",
        "老人把最后一坛酒推到他面前，说：“喝完这坛，你就该上路了。”",
        "雪落无声，覆盖了整个村庄，屋顶、树梢、田埂，全都变成了一片白。",
        "他知道自己没有退路，也知道前面的路比想象中更加艰难，但他还是要走下去。",
    ]
    return "".join(parts * n_repeat)


TEXT = make_corpus(30)
print("语料：%d 个字符，%d 个字节" % (len(TEXT), len(TEXT.encode("utf-8"))))

# ---------------------------------------------------------------- 语料体检
print("\n--- 语料体检 ---")
info = inspect_corpus(TEXT)
print("  不同字符种类    : %d" % info["distinct_chars"])
print("  抽样中不同二字组: %d" % info["distinct_pairs_sampled"])
print("  出现最多的字    : %s" % "".join(ch for ch, _ in info["top_chars"][:12]))

# ---------------------------------------------------------------- 字符级
print("\n--- 字符级分词器 ---")
ct = CharTokenizer()
ct.train(TEXT, vocab_size=6000, verbose=True)
cids = ct.encode(TEXT[:27])
print("  示例: %r" % TEXT[:27])
print("  -> %d 个 token" % len(cids))
check("字符级 往返一致", ct.decode(cids) == TEXT[:27])
check("字符级 一个汉字一个token", len(cids) == len(TEXT[:27]),
      "%d vs %d" % (len(cids), len(TEXT[:27])))

# ---------------------------------------------------------------- BPE 压缩比曲线
print("\n--- 字符级 BPE：词表大小 vs 压缩比 ---")
print("  %-10s %-10s %-12s %-10s %s" % ("目标词表", "实际词表", "合并规则数", "压缩比", "训练耗时"))
bpe_best = None
print("  说明: 不测 vocab=512 —— 基础词表（字符 + 256 字节兜底）已超过 512，")
print("        预算为负，这是设计限制而非 bug。词表至少要 2 倍于不同字符种类。")
for vs in (1024, 2048, 4096):
    t0 = time.time()
    tk = BPETokenizer()
    tk.train(TEXT, vocab_size=vs, verbose=False, min_pair_count=2)
    dt = time.time() - t0
    ids = tk.encode(TEXT)
    comp = len(TEXT) / float(len(ids))
    print("  %-10d %-10d %-12d %-10.3f %.2f 秒"
          % (vs, tk.vocab_size, len(tk.merges), comp, dt))
    if vs == 2048:
        bpe_best = tk
    if len(tk.merges) == 0:
        FAIL.append("BPE(vs=%d) 一条合并都没学到" % vs)

check("BPE 学到了合并规则", bpe_best is not None and len(bpe_best.merges) > 0,
      "%d 条" % (len(bpe_best.merges) if bpe_best else 0))
print("  说明: 这份测试语料是 20 个句子重复 30 次的人造语料，")
print("        可合并的对有限，学不满 1788 条预算是正常的，不算失败。")

# ---------------------------------------------------------------- BPE 完整校验
print("\n--- BPE 完整校验（2048 词表，详细输出）---")
bt = BPETokenizer()
bt.train(TEXT, vocab_size=2048, verbose=True, min_pair_count=2)

SENT = "他抬起头，看着窗外的雨，忽然想起了很多年前的那个下午。"
ids = bt.encode(SENT)
print("\n  示例: %r" % SENT)
print("  字符级会是 %d 个 token，BPE 是 %d 个 token" % (len(SENT), len(ids)))
pieces = []
for i in ids:
    pieces.append(bt.decode([i]))
print("  切分结果: %s" % " | ".join(pieces))
check("BPE 单句往返一致", bt.decode(ids) == SENT)

all_ids = bt.encode(TEXT)
check("BPE 全语料往返一致", bt.decode(all_ids) == TEXT,
      "token 数 %d / 字符数 %d" % (len(all_ids), len(TEXT)))
comp = len(TEXT) / float(len(all_ids))
print("  全语料压缩比: %.3f 字符/token（字符级是 1.000）" % comp)
check("BPE 压缩比优于字符级", comp > 1.2, "%.3f" % comp)

check("BPE 解码后无替换字符", "\\ufffd" not in bt.decode(all_ids))

# 边界用例
edge = ["", "我", "。", "   ", "\\n\\n", "\\U00020000", "\\U0001F004", "ＡＢＣ", "abc123",
        "「引号」", "\\t制表", "混合 mixed 文本 123", "……——", "。。"]
bad_edge = []
for e in edge:
    got = bt.decode(bt.encode(e))
    if got != e:
        bad_edge.append((e, got))
check("边界用例全部往返一致", not bad_edge, str(bad_edge[:3]))

ids_be = bt.encode("测试", add_bos=True, add_eos=True)
check("bos/eos 正确拼接", ids_be[0] == BOS and ids_be[-1] == EOS and len(ids_be) >= 4,
      str(ids_be))

# ---------------------------------------------------------------- 存取
print("\n--- 存取一致性 ---")
tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tok_test.tmp")
ct.save(tmp)
ct2 = load_tokenizer(tmp)
check("字符级 存/读 词表一致", ct2.vocab_size == ct.vocab_size,
      "%d vs %d" % (ct2.vocab_size, ct.vocab_size))
check("字符级 存/读 编码一致", ct2.encode(TEXT[:500]) == ct.encode(TEXT[:500]))
os.remove(tmp)

bt.save(tmp)
bt2 = load_tokenizer(tmp)
check("BPE 存/读 词表一致", bt2.vocab_size == bt.vocab_size,
      "%d vs %d" % (bt2.vocab_size, bt.vocab_size))
check("BPE 存/读 合并规则一致", len(bt2.merges) == len(bt.merges),
      "%d vs %d" % (len(bt2.merges), len(bt.merges)))
check("BPE 存/读 编码完全一致", bt2.encode(TEXT) == bt.encode(TEXT))
check("BPE 存/读 解码一致", bt2.decode(bt2.encode(TEXT)) == TEXT)
print("  词表文件大小: %d 字节" % os.path.getsize(tmp))
os.remove(tmp)

# ---------------------------------------------------------------- 速度
print("\n--- 编码速度 ---")
big = TEXT * 10
n_chars = len(big)
for name, tk in (("字符级", ct), ("BPE2048", bt)):
    t0 = time.time()
    tk.encode(big)
    dt = time.time() - t0
    print("  %-8s: %s 字符用了 %.2f 秒 -> %s 字符/秒"
          % (name, "{:,}".format(n_chars), dt, "{:,}".format(int(n_chars / dt))))

# ---------------------------------------------------------------- 汇总
print()
if FAIL:
    print("有 %d 项失败: %s" % (len(FAIL), ", ".join(FAIL)))
    sys.exit(1)
print("全部分词器测试通过。")
