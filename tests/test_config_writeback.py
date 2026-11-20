# -*- coding: utf-8 -*-
"""配置写回的回归测试：**改几个值，别把注释抹掉**。

## 为什么值得单独一个文件

出厂配置（`configs/*.ini`）的价值有一大半在注释里 ——
「n_layer 堆几层」「n_kv_head 越小 KV cache 越省」「data_dir 是语料目录，
不是训练输出目录」。新手就是靠这些注释在学的。

而 `save_config()` 原本是「按 spec 重新生成整份文件」，
一次写回就把注释全抹了。用户只是点了下按钮，回头打开配置文件
发现自己看不懂了 —— 而且**没有任何提示**。

所以界面改单个键时走 `patch_config()`（就地改值），
`save_config()` 默认也改成保注释。这里把这两条路都钉住。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lingyu.config import load_config, patch_config, save_config  # noqa: E402

SAMPLE = """\
# ==========================================================================
#  灵语训练配置 —— small
#  n_layer  层数   堆几层 Transformer（越深越会「推理」）
# ==========================================================================

[model]
# ---- 这三个是 nanoGPT 用户最常改的 ----
n_layer     = 16            # 层数
n_embd      = 256           # 宽度

[data]
# ⚠️ data_dir 是**语料目录**（prepare 的产出），不是训练输出目录
data_dir= data1
tokenizer_path= data1/tokenizer.txt

[sample]
prompt = 他抬起头，看着
"""


@pytest.fixture
def cfgfile(tmp_path):
    p = tmp_path / "small.ini"
    p.write_text(SAMPLE, encoding="utf-8")
    return str(p)


# ==================================================== patch_config

def test_patch_changes_only_the_named_keys(cfgfile):
    n = patch_config(cfgfile, {("data", "data_dir"): "data"})
    assert n == 1
    text = open(cfgfile, encoding="utf-8").read()
    assert "data_dir= data\n" in text
    assert "tokenizer_path= data1/tokenizer.txt" in text   # 没点到的不许动


def test_patch_keeps_every_comment_and_line(cfgfile):
    before = open(cfgfile, encoding="utf-8").read().splitlines()
    patch_config(cfgfile, {("data", "data_dir"): "data",
                           ("model", "n_layer"): 32})
    after = open(cfgfile, encoding="utf-8").read().splitlines()
    assert len(before) == len(after), "行数变了，说明结构被动过"
    for b, a in zip(before, after):
        assert a.startswith("#") == b.startswith("#"), \
            "注释行变成了非注释行：%r -> %r" % (b, a)
    assert "#  n_layer  层数   堆几层 Transformer（越深越会「推理」）" in after
    assert "n_layer     = 32            # 层数" in after, "行尾注释丢了"


def test_patch_does_not_duplicate_already_correct_keys(cfgfile):
    """★ 值已经对了就不许再追加一行。

    踩过的坑：只拿「改过的键」判断有没有找到，于是「值已正确」的键
    被误判成「文件里没有」→ 追加一行 → 同一个键出现两次。
    INI 里重复键谁生效取决于解析器，等于给自己埋雷。
    """
    patch_config(cfgfile, {("data", "data_dir"): "data1"})
    lines = open(cfgfile, encoding="utf-8").read().splitlines()
    assert sum(1 for l in lines if l.strip().startswith("data_dir")) == 1


def test_patch_preserves_inline_comment_and_spacing(cfgfile):
    patch_config(cfgfile, {("model", "n_embd"): 384})
    assert "n_embd      = 384           # 宽度" in \
        open(cfgfile, encoding="utf-8").read()


def test_patch_appends_missing_key_into_existing_section(cfgfile):
    n = patch_config(cfgfile, {("data", "data_path"): "chat/x.jsonl"})
    assert n == 1
    lines = open(cfgfile, encoding="utf-8").read().splitlines()
    i = lines.index("[data]")
    j = lines.index("[sample]")
    assert any(l.startswith("data_path = ") for l in lines[i:j]), \
        "新键要落进它自己的段，不能跑到别的段或文件末尾"


def test_patch_creates_missing_section(cfgfile):
    n = patch_config(cfgfile, {("parallel", "backend"): "fsdp"})
    assert n == 1
    text = open(cfgfile, encoding="utf-8").read()
    assert "[parallel]" in text and "backend = fsdp" in text


def test_patch_noop_when_value_already_right(cfgfile):
    assert patch_config(cfgfile, {("data", "data_dir"): "data1"}) == 0


def test_patch_missing_file_is_a_noop(tmp_path):
    assert patch_config(str(tmp_path / "nope.ini"), {("data", "data_dir"): "d"}) == 0


def test_patch_result_is_loadable(cfgfile):
    patch_config(cfgfile, {("data", "data_dir"): "data",
                           ("model", "n_layer"): 24})
    cfg = load_config(cfgfile)
    assert cfg["data"]["data_dir"] == "data"
    assert cfg["model"]["n_layer"] == 24


def test_patch_writes_forward_slash_paths_fine(cfgfile):
    """Windows 路径两种分隔符都要能读回来（config 值不做转义处理）。"""
    patch_config(cfgfile, {("data", "data_dir"): r"C:\path\to\corpus 目录\x"})
    assert load_config(cfgfile)["data"]["data_dir"] == r"C:\path\to\corpus 目录\x"


# ==================================================== save_config

def test_save_config_preserves_comments_by_default(cfgfile):
    cfg = load_config(cfgfile)
    cfg["data"]["data_dir"] = "data"
    cfg["model"]["n_layer"] = 8
    save_config(cfgfile, cfg)
    text = open(cfgfile, encoding="utf-8").read()
    assert "nanoGPT 用户最常改的" in text, "注释被整份重写抹掉了"
    assert load_config(cfgfile)["model"]["n_layer"] == 8


def test_save_config_creates_file_when_absent(tmp_path):
    p = str(tmp_path / "new.ini")
    save_config(p, {"model": {"n_layer": 4}, "data": {"data_dir": "d"}})
    cfg = load_config(p)
    assert cfg["model"]["n_layer"] == 4


def test_save_config_writes_none_as_empty_not_the_word_none(tmp_path):
    """★ `default=None` 的键（data_dir 之类）本意是「没设置」。

    写成字面量 `None` 的话，读回来是**四个字母的字符串** ——
    一个「没设置」的目录会变成名叫 `None` 的目录。
    """
    p = str(tmp_path / "n.ini")
    save_config(p, {"data": {"data_dir": None, "tokenizer_path": ""}})
    text = open(p, encoding="utf-8").read()
    assert "None" not in text
    assert load_config(p)["data"]["data_dir"] in ("", None)


def test_save_config_bool_format(tmp_path):
    p = str(tmp_path / "b.ini")
    save_config(p, {"model": {"use_moe": True, "bias": False}})
    cfg = load_config(p)
    assert cfg["model"]["use_moe"] is True
    assert cfg["model"]["bias"] is False


# ==================================================== 字符串 round-trip

def test_patch_roundtrips_string_with_hash(cfgfile):
    """值里有 `#` 不许被当注释截断。

    真实事故：`prompt = 说 # 你好` 写回再读，只剩「说」——
    GUI 改一次配置保存，值就悄悄变了，界面上还显示正常。
    """
    patch_config(cfgfile, {("sample", "prompt"): "说 # 你好"})
    assert load_config(cfgfile)["sample"]["prompt"] == "说 # 你好"


def test_patch_roundtrips_leading_trailing_spaces(cfgfile):
    patch_config(cfgfile, {("sample", "prompt"): "  前后有空格  "})
    assert load_config(cfgfile)["sample"]["prompt"] == "  前后有空格  "


def test_patch_roundtrips_numeric_looking_string(cfgfile):
    """`prompt = 123` 的类型是 str，读回来不许变成 int。"""
    patch_config(cfgfile, {("sample", "prompt"): "123"})
    v = load_config(cfgfile)["sample"]["prompt"]
    assert v == "123" and isinstance(v, str), "被读成了 %r" % (v,)


def test_patch_roundtrips_quotes_and_hash(cfgfile):
    """字符串本身含引号 + `#`：要转义，不能让引号状态搞错。"""
    val = 'a "b" # c'
    patch_config(cfgfile, {("sample", "prompt"): val})
    assert load_config(cfgfile)["sample"]["prompt"] == val


def test_patch_removes_duplicate_keys(tmp_path):
    """★ 文件里同一个键出现两次时，patch 必须删掉多余的那行。

    INI 解析是「后面的赢」。只改第一行的话，第二次出现的那行会把
    刚写进去的新值静默覆盖回去 —— 用户看到「保存成功」，重启还是旧值。
    """
    p = tmp_path / "dup.ini"
    p.write_text("[data]\ndata_dir = a\ndata_dir = b\n", encoding="utf-8")
    assert patch_config(str(p), {("data", "data_dir"): "c"}) >= 1
    lines = [l for l in open(str(p), encoding="utf-8").read().splitlines()
             if l.strip().startswith("data_dir")]
    assert len(lines) == 1, lines
    assert load_config(str(p))["data"]["data_dir"] == "c"


def test_save_config_json_roundtrip(tmp_path):
    """写 .json 就必须是 JSON，不能把 INI 塞进 .json 文件。"""
    p = str(tmp_path / "cfg.json")
    cfg = {"model": {"n_layer": 4, "dropout": 0.1},
           "data": {"data_dir": "D:/语料"}}
    save_config(p, cfg)
    text = open(p, encoding="utf-8").read()
    assert text.lstrip().startswith("{"), "写成 INI 了：\n" + text
    assert load_config(p) == cfg


def test_save_config_ini_roundtrips_tricky_strings(tmp_path):
    p = str(tmp_path / "t.ini")
    cfg = {"sample": {"prompt": "a # b", "system_prompt": "  x  ", "seed": 1}}
    save_config(p, cfg)
    got = load_config(p)
    assert got["sample"]["prompt"] == "a # b"
    assert got["sample"]["system_prompt"] == "  x  "
    assert got["sample"]["seed"] == 1


# ==================================================== 出厂配置自身

def test_factory_configs_agree_on_the_data_dir():
    """六个出厂配置必须指向**同一个**语料目录名。

    ★ 真实事故：`small.ini` 独自写着 `data1`，其余五个都是 `data`。
      打包版把 `data` 钉成 `<runtime>/data`，也正是语料准备页写的地方；
      而 `small.ini` 那份被钉成 `<runtime>/data1`（一个不存在的目录）——
      用户照着推荐配置准备完语料，训练页却说「还没准备语料」。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cdir = os.path.join(root, "configs")
    seen = {}
    for name in sorted(os.listdir(cdir)):
        if not name.endswith(".ini"):
            continue
        cfg = load_config(os.path.join(cdir, name))
        dd = (cfg.get("data") or {}).get("data_dir")
        seen[name] = dd
    assert seen, "没找到出厂配置"
    assert set(seen.values()) == {"data"}, \
        "出厂配置的 data_dir 不一致：%s" % seen
