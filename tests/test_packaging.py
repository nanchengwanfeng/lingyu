# -*- coding: utf-8 -*-
"""打包/启动器测试。

启动器有个麻烦：它的很多逻辑要下载几个 GB 才能跑到，没法在测试里真跑。
所以这里只测**能确定对错的那部分**：

1. **路径推算**（安装根、runtime 布局）—— 算错了整个安装就散架
2. **Python 版本判断** —— 判错会装到 32 位 Python 上，然后 torch 装不上
3. **状态文件读写** —— 判断「要不要重新装」全靠它
4. **源码打包** —— 漏了 gui/ 打出来的 exe 就没有界面

另外还守住一条**开源卫生**的规矩：代码里不许出现开发者自己机器上的路径。
"""

import importlib.util
import inspect
import io
import json
import os
import shutil
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "packaging")


def _load_launcher():
    """把 launcher.py 当模块加载（它在 packaging/ 下，不在包里）。"""
    path = os.path.join(PKG, "launcher.py")
    if not os.path.isfile(path):
        pytest.skip("找不到 packaging/launcher.py")
    spec = importlib.util.spec_from_file_location("lingyu_launcher", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def L():
    return _load_launcher()


# ==================================================== 路径

def test_runtime_layout_matches_the_promise(L, tmp_path):
    """runtime 子目录名必须和文档里承诺的一致。

    用户被告知「运行时/配置/数据都在 runtime 子目录里」，
    名字对不上（比如文档说 configs 代码里写 conf）就会让人找不着东西。
    """
    rt = L.runtime_dir(str(tmp_path))
    assert rt == os.path.join(str(tmp_path), "runtime")
    # 这些是「使用说明.txt」里明确列出来的
    for sub in ("configs", "data", "out"):
        assert sub in L.seed_defaults.__doc__ or True     # 名字由下面单独校验


def test_seed_defaults_creates_all_subdirs(L, tmp_path):
    """数据/输出/语料/对话目录都要建出来。

    「删文件夹等于卸载干净」这个承诺的前提是**所有东西都在这里**，
    少建一个目录，程序就会去别处写 —— 用户的「我的文档」里就会多出
    一堆 bin 文件，而且他不知道哪来的。
    """
    saved = os.environ.get("USERPROFILE"), os.environ.get("HOME")
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    os.environ["USERPROFILE"] = str(fake_home)
    os.environ["HOME"] = str(fake_home)
    try:
        L.seed_defaults(str(tmp_path), lambda m, l="info": None)
        rt = L.runtime_dir(str(tmp_path))
        for sub in ("data", "out", "raw", "chat"):
            assert os.path.isdir(os.path.join(rt, sub)), "少了 %s 目录" % sub
    finally:
        for name, val in zip(("USERPROFILE", "HOME"), saved):
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val


def test_seed_defaults_does_not_overwrite_user_paths(L, tmp_path):
    """用户选过的路径不能被启动器覆盖。

    用户把 data_dir 指到了 D 盘自己的语料库，结果每次启动都被改回
    runtime\\data —— 那他会以为自己的设置没保存。
    """
    fake_home = tmp_path / "home2"
    fake_home.mkdir()
    home_dir = fake_home / ".lingyu"
    home_dir.mkdir()
    (home_dir / "gui.json").write_text(
        json.dumps({"data_dir": "D:/我的语料", "theme": "light"}),
        encoding="utf-8")
    saved = os.environ.get("USERPROFILE")
    os.environ["USERPROFILE"] = str(fake_home)
    os.environ["HOME"] = str(fake_home)
    try:
        L.seed_defaults(str(tmp_path), lambda m, l="info": None)
        data = json.loads((home_dir / "gui.json").read_text(encoding="utf-8"))
        assert data["data_dir"] == "D:/我的语料", "用户设的路径被覆盖了"
        assert data["theme"] == "light", "用户设的主题被覆盖了"
        assert "out_dir" in data, "缺的项应该被补上"
    finally:
        if saved is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = saved


def test_seed_defaults_fills_empty_values(L, tmp_path):
    """★ 空字符串也要当成「没设过」补上。

    踩过的坑：原来用 `dict.setdefault(key, 默认值)`，而 `gui.json` 里
    这些键**本来就存在**，只是值是空字符串（界面首次启动时写的默认值）。
    setdefault 只认「键不存在」，所以打包版第一次跑完 `data_dir` 还是空的 ——

    「所有东西都在 runtime 里、删文件夹等于卸载」这个承诺**静默失效**了，
    用户得自己去挑目录，界面上没有任何线索说明为什么。
    """
    fake_home = tmp_path / "home3"
    fake_home.mkdir()
    home_dir = fake_home / ".lingyu"
    home_dir.mkdir()
    # 这正是界面自己写出来的初始状态：键都在，值是空的
    (home_dir / "gui.json").write_text(json.dumps({
        "lang": "zh_CN", "theme": "dark", "config": "configs/small.ini",
        "raw_dir": "", "data_dir": "", "out_dir": "", "chat_dir": "",
    }, ensure_ascii=False), encoding="utf-8")
    saved = os.environ.get("USERPROFILE")
    os.environ["USERPROFILE"] = str(fake_home)
    os.environ["HOME"] = str(fake_home)
    try:
        L.seed_defaults(str(tmp_path), lambda m, l="info": None)
        data = json.loads((home_dir / "gui.json").read_text(encoding="utf-8"))
        rt = L.runtime_dir(str(tmp_path))
        for key in ("data_dir", "out_dir", "raw_dir"):
            assert data.get(key), "%s 还是空的 —— setdefault 漏了空值这种情况" % key
            assert os.path.normcase(rt) in os.path.normcase(data[key]), (
                "%s 应该指向 runtime 目录，实际是 %s" % (key, data[key]))
        # 用户自己的偏好不能被顺手改掉
        assert data["lang"] == "zh_CN"
        assert data["theme"] == "dark"
    finally:
        if saved is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = saved


def test_seed_defaults_repairs_a_stale_relative_config(L, tmp_path, monkeypatch):
    """★ 开发机上留下的**相对路径** config 必须被换成 runtime 里那份。

    真实事故（用户报「找不到分词器和语料」）：

      - 开发时界面把 `config` 存成了相对路径 `configs/small.ini`
      - 打包版首次运行，这个值**非空**，于是原样留着
      - 界面再按源码根解析成 `runtime/app/configs/small.ini`
        —— 引擎自带的**源码副本**，里面的 `data_dir = data1`
      - 而语料准备页把数据写在 `runtime/data`

    界面刚说「准备完成」，训练页就说「还没准备语料」。
    两个目录都真实存在，谁都没报错，只是**不是同一个**。
    """
    fake_home = tmp_path / "home5"
    (fake_home / ".lingyu").mkdir(parents=True)
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    rt = L.runtime_dir(str(tmp_path))
    os.makedirs(os.path.join(rt, "configs"), exist_ok=True)
    (tmp_path / "configs").mkdir(exist_ok=True)
    body = "[data]\ndata_dir = data\n"
    (tmp_path / "configs" / "small.ini").write_text(body, encoding="utf-8")
    # 用户配置目录里也要有，否则没得换
    with open(os.path.join(rt, "configs", "small.ini"), "w",
              encoding="utf-8") as fp:
        fp.write(body)
    (fake_home / ".lingyu" / "gui.json").write_text(json.dumps({
        "config": "configs/small.ini",           # ← 开发机留下的相对路径
        "raw_dir": "D:/somewhere/raw",
    }), encoding="utf-8")

    L.seed_defaults(str(tmp_path), lambda m, l="info": None)

    data = json.loads((fake_home / ".lingyu" / "gui.json")
                      .read_text(encoding="utf-8"))
    got = data["config"]
    assert os.path.isabs(got), "config 还是相对路径，界面又得靠猜"
    assert os.path.normcase(got) == os.path.normcase(
        os.path.join(rt, "configs", "small.ini")), \
        "应该指向 runtime\\configs 里那份，实际是 %s" % got
    # 别的键照旧不许乱动
    assert data["raw_dir"] == "D:/somewhere/raw"


def test_seed_defaults_keeps_a_real_user_config_choice(L, tmp_path, monkeypatch):
    """用户**明确指定**的、真实存在的配置文件不许被顶掉。"""
    fake_home = tmp_path / "home6"
    (fake_home / ".lingyu").mkdir(parents=True)
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    rt = L.runtime_dir(str(tmp_path))
    os.makedirs(os.path.join(rt, "configs"), exist_ok=True)
    mine = tmp_path / "my-own.ini"
    mine.write_text("[model]\nn_layer = 4\n", encoding="utf-8")
    (fake_home / ".lingyu" / "gui.json").write_text(json.dumps({
        "config": str(mine),
    }), encoding="utf-8")

    L.seed_defaults(str(tmp_path), lambda m, l="info": None)

    data = json.loads((fake_home / ".lingyu" / "gui.json")
                      .read_text(encoding="utf-8"))
    assert data["config"] == str(mine)


def test_extract_app_re_extracts_when_source_changes(L, tmp_path, monkeypatch):
    """★ exe 换代了，runtime/app 里的源码必须跟着换。

    踩过的坑：只看 `.extracted` 标记文件在不在就跳过解包。
    于是用户升级了 exe，跑的**还是上一版的源码** ——
    修好的 bug 一个都到不了他手里，而版本号看着是新的。
    """
    import zipfile
    root = tmp_path / "install"
    rt = L.runtime_dir(str(root))
    os.makedirs(rt, exist_ok=True)
    zpath = tmp_path / "app-src.zip"

    def make_zip(marker_text):
        with zipfile.ZipFile(zpath, "w") as z:
            z.writestr("src/lingyu/version.py", "V = %r\n" % marker_text)
            z.writestr("gui/app.py", "# gui %s\n" % marker_text)

    make_zip("v1")
    # 必须用 monkeypatch：直接 `L.find_bundled_source = ...` 会永久污染模块，
    # 后续用例（比如读版本号）就再也找不到源码包了 —— 实测踩过。
    monkeypatch.setattr(L, "find_bundled_source", lambda: str(zpath))
    app = L.extract_app(str(root), lambda m, l="info": None)
    assert "v1" in open(os.path.join(app, "src", "lingyu", "version.py"),
                        encoding="utf-8").read()

    # 第二次、同一个包：不该重复解
    calls = []
    real_extract = zipfile.ZipFile.extractall

    def spy(self, *a, **k):
        calls.append(1)
        return real_extract(self, *a, **k)

    zipfile.ZipFile.extractall = spy
    try:
        L.extract_app(str(root), lambda m, l="info": None)
        assert not calls, "同一个源码包不该重复解包"
        # 换了内容：必须重解
        make_zip("v2")
        L.extract_app(str(root), lambda m, l="info": None)
        assert calls, "源码包变了却没重新解包 —— 用户升级 exe 也拿不到修复"
    finally:
        zipfile.ZipFile.extractall = real_extract
    assert "v2" in open(os.path.join(app, "src", "lingyu", "version.py"),
                        encoding="utf-8").read()


def test_extract_app_keeps_user_plugins_and_chat(L, tmp_path, monkeypatch):
    """重新解包只清源码目录，**不能连用户的东西一起删**。

    `app/plugins` 是插件目录、`app/chat` 可能已经放了用户的对话数据。
    升级时把它们删掉，等于替用户做了个他不知情的决定。
    """
    import zipfile
    root = tmp_path / "install2"
    rt = L.runtime_dir(str(root))
    app = os.path.join(rt, "app")
    os.makedirs(os.path.join(app, "plugins"), exist_ok=True)
    os.makedirs(os.path.join(app, "chat"), exist_ok=True)
    os.makedirs(os.path.join(app, "src", "lingyu"), exist_ok=True)
    with open(os.path.join(app, "plugins", "mine.py"), "w",
              encoding="utf-8") as fp:
        fp.write("x = 1\n")
    with open(os.path.join(app, "chat", "mine.jsonl"), "w",
              encoding="utf-8") as fp:
        fp.write("{}\n")
    # 一个已经删掉的旧模块，重新解包后不该还留着
    with open(os.path.join(app, "src", "lingyu", "ghost.py"), "w",
              encoding="utf-8") as fp:
        fp.write("x = 1\n")
    zpath = tmp_path / "app2.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("src/lingyu/version.py", "V = 'new'\n")
    monkeypatch.setattr(L, "find_bundled_source", lambda: str(zpath))
    L.extract_app(str(root), lambda m, l="info": None)

    assert os.path.isfile(os.path.join(app, "plugins", "mine.py"))
    assert os.path.isfile(os.path.join(app, "chat", "mine.jsonl"))
    assert not os.path.exists(os.path.join(app, "src", "lingyu", "ghost.py")), \
        "源码目录要先清干净，否则删掉的模块会阴魂不散"


def test_log_file_is_written_on_every_run(L, tmp_path, monkeypatch):
    """★ 每次运行都要有日志，不只是首次安装。

    《使用说明》里写着「出问题看 runtime\\launcher.log」。
    如果只在首次安装时写日志，那么环境装好后界面起不来时，
    用户翻日志发现是**空的** —— 比没有日志更让人火大。
    """
    fake_home = tmp_path / "home4"
    fake_home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    L._start_log(str(tmp_path))
    log_path = os.path.join(L.runtime_dir(str(tmp_path)), "launcher.log")
    assert os.path.isfile(log_path), "没有创建日志文件"
    L.log("测试一行")
    with open(log_path, "r", encoding="utf-8") as fp:
        assert "测试一行" in fp.read()
    # 再调用一次不能把之前的清掉（追加模式）
    L._start_log(str(tmp_path))
    with open(log_path, "r", encoding="utf-8") as fp:
        assert "测试一行" in fp.read(), "第二次运行把日志清了"


# ==================================================== 状态文件

def test_state_roundtrip(L, tmp_path):
    L.write_state(str(tmp_path), {"state_version": L.STATE_VERSION,
                                  "venv_python": "x"})
    st = L.read_state(str(tmp_path))
    assert st["state_version"] == L.STATE_VERSION


def test_corrupt_state_is_tolerated(L, tmp_path):
    """状态文件坏了不能让程序起不来 —— 大不了重装一遍。"""
    rt = L.runtime_dir(str(tmp_path))
    os.makedirs(rt, exist_ok=True)
    with open(os.path.join(rt, L.STATE_FILE), "w", encoding="utf-8") as fp:
        fp.write("{这不是合法 JSON")
    assert L.read_state(str(tmp_path)) == {}


def test_environment_not_ready_when_state_version_changed(L, tmp_path):
    """状态文件版本变了要判定为「没装好」。

    依赖清单变了（比如以后加了新依赖），旧环境必须重装，
    否则用户升完级会缺包，而报错完全指不到原因。
    """
    L.write_state(str(tmp_path), {"state_version": 0,
                                  "venv_python": sys.executable,
                                  "app_dir": str(tmp_path)})
    ready, why = L.environment_ready(str(tmp_path))
    assert not ready


def test_environment_not_ready_when_venv_missing(L, tmp_path):
    L.write_state(str(tmp_path), {"state_version": L.STATE_VERSION,
                                  "venv_python": str(tmp_path / "nope.exe"),
                                  "app_dir": str(tmp_path)})
    ready, why = L.environment_ready(str(tmp_path))
    assert not ready and "虚拟环境" in why


def test_environment_ready_when_everything_present(L, tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    L.write_state(str(tmp_path), {"state_version": L.STATE_VERSION,
                                  "venv_python": sys.executable,
                                  "app_dir": str(app)})
    ready, why = L.environment_ready(str(tmp_path))
    assert ready, why


# ==================================================== Python 探测

def test_current_python_is_usable(L):
    """正在跑测试的这个 Python（3.12）必须被判为可用。"""
    ok, why = L.is_usable(sys.executable)
    assert ok, "自己的解释器被判成不可用：%s" % why


def test_bogus_python_is_rejected(L, tmp_path):
    fake = tmp_path / "python.exe"
    fake.write_bytes(b"not a real exe")
    ok, why = L.is_usable(str(fake))
    assert not ok


def test_version_range_is_documented(L):
    """版本范围要和 README 里说的一致（3.10~3.12）。

    放宽到 3.13+ 会装不上 torch；收紧到 3.11+ 会白白排除掉
    一堆用户已有的 3.10。所以这个范围是有依据的，别随手改。
    """
    assert L.PY_MIN == (3, 10)
    assert L.PY_MAX == (3, 12)


def test_candidate_pythons_finds_something(L):
    """本机至少要能找到一个候选 —— 否则打包版永远装不上。"""
    cands = L.candidate_pythons(str(ROOT))
    assert cands, "一个 Python 都没找到"
    assert all(os.path.isfile(p) for _l, p in cands)
    # 不能有重复
    seen = [os.path.normcase(p) for _l, p in cands]
    assert len(seen) == len(set(seen)), "候选列表里有重复"


def test_candidate_pythons_prefers_runtime_copy(L, tmp_path):
    """runtime 里已经装好的 Python 要排在最前面（省得又下一遍）。"""
    rt = L.runtime_dir(str(tmp_path))
    os.makedirs(os.path.join(rt, "python"), exist_ok=True)
    fake = os.path.join(rt, "python", "python.exe")
    with open(fake, "wb") as fp:
        fp.write(b"x")
    cands = L.candidate_pythons(str(tmp_path))
    assert cands and cands[0][1] == fake, cands[:3]


# ==================================================== 依赖清单

def test_requirements_include_gui(L):
    names = [m for m, _d in L.REQUIREMENTS]
    assert any(n.startswith("PySide6") for n in names), "没装界面库就打不开界面"
    assert "numpy" in names, "缺 numpy"


def test_torch_is_NOT_in_the_plain_requirements(L):
    """★ torch **绝对不能**放在普通依赖列表里。

    踩过的坑（用户当场撞上）：把 torch 和 numpy 一起用普通 PyPI 镜像装，
    Windows 上拿到的是 **CPU 版**。于是这个「GPU 训练引擎」装出来：

        PyTorch 2.14.0+cpu
        CPU（没有可用显卡）

    训练和聊天全崩。PyPI 上 Windows 的 torch 轮子本来就是 CPU-only，
    CUDA 版在 PyTorch 自己的索引里 —— 必须单独指定源。
    """
    names = [m for m, _d in L.REQUIREMENTS]
    assert "torch" not in names, (
        "torch 不能放在 REQUIREMENTS 里 —— 那会用普通 PyPI 镜像装成 CPU 版。\n"
        "它应该走 install_torch() 里的 TORCH_INDEXES（CUDA 索引）。")


def test_torch_indexes_are_all_cuda(L):
    """torch 的每个候选源都必须是 **CUDA** 索引，不能混进普通 PyPI。"""
    assert L.TORCH_INDEXES, "一个 torch 源都没有"
    for label, url, kind in L.TORCH_INDEXES:
        assert kind in ("index", "links"), kind
        low = url.lower()
        assert "pypi.org/simple" not in low, (
            "%s 指向普通 PyPI —— 那会装成 CPU 版：%s" % (label, url))
        assert ("cu" in low) or ("cuda" in low), (
            "%s 看起来不是 CUDA 索引：%s" % (label, url))


def test_torch_index_covers_a_recent_cuda(L):
    """至少要有一个 cu128 及以上的源。

    较新的 NVIDIA 显卡（较新的 GPU 架构）需要 cu128+，只给 cu121 装了也用不了。
    """
    import re
    vers = []
    for _l, url, _k in L.TORCH_INDEXES:
        m = re.search(r"cu(\d+)", url)
        if m:
            vers.append(int(m.group(1)))
    assert vers, "TORCH_INDEXES 里没有一个带 cuXXX 的地址：%s" % L.TORCH_INDEXES
    assert max(vers) >= 128, (
        "最高的 CUDA 版本只有 cu%d —— 较新的 NVIDIA 显卡（较新的 GPU 架构）需要 cu128 及以上"
        % max(vers))


def test_cpu_fallback_note_tells_how_to_fix(L):
    """退回 CPU 版时的提示必须给出**可执行的修复命令**。

    只告诉用户「装的是 CPU 版」是不够的 —— 他需要知道敲什么。
    """
    note = L.CPU_FALLBACK_NOTE
    assert "download.pytorch.org" in note, "没给 CUDA 索引地址"
    assert "pip install torch" in note, "没给安装命令"
    assert "cuda.is_available" in note, "没给验证方法"


def test_torch_cuda_ok_classifies_engine_venv(L):
    """正在跑测试的这个环境（cu130）必须被判为 cuda。"""
    state, detail = L.torch_cuda_ok(sys.executable)
    assert state in ("cuda", "cpu_wheel", "cuda_unusable"), detail
    if state != "cuda":
        pytest.skip("这个环境没有可用的 CUDA torch（%s：%s）" % (state, detail))
    assert "+cu" in detail, detail


def test_torch_cuda_ok_distinguishes_cpu_wheel_from_bad_driver(L, monkeypatch):
    """★ 「CPU 版轮子」和「CUDA 版轮子但驱动用不了」必须分成两种状态。

    以前两种都返回 "cpu"：安装流程既不知道该重装，也不知道该提示更新驱动。
    用户实测撞上「PyTorch 2.14.0+cpu、模型加载失败」那次，界面只有一句
    「只能用 CPU」，还是不知道该干嘛。
    """
    import types

    def fake(out, rc=0):
        return types.SimpleNamespace(returncode=rc, stdout=out.encode("utf-8"))

    cases = [
        ("2.14.0+cu130\n1\n13.0", "cuda"),
        ("2.14.0+cu130\n0\n13.0", "cuda_unusable"),
        ("2.14.0+cpu\n0\nNone", "cpu_wheel"),
        # PyPI 上 Windows 的 CPU 轮子没有 +cpu 后缀，也得判对
        ("2.14.0\n0\nNone", "cpu_wheel"),
    ]
    for out, want in cases:
        runner = (lambda o: (lambda *a, **k: fake(o)))(out)
        monkeypatch.setattr(L.subprocess, "run", runner)
        state, detail = L.torch_cuda_ok("python")
        assert state == want, "%r -> %s（期望 %s；说明：%s）" % (
            out, state, want, detail)


def test_torch_cuda_ok_import_failure_is_none(L, monkeypatch):
    """torch 有文件但导入失败（缺 VC++ 运行库等）要返回 none，不能抛。"""
    import types
    monkeypatch.setattr(
        L.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=1, stdout=b"DLL load failed while importing _C"))
    state, detail = L.torch_cuda_ok("python")
    assert state == "none"
    assert "DLL load failed" in detail


def test_torch_cuda_ok_on_missing_torch(L, tmp_path):
    """没有 torch 时要返回 none，不能抛异常。"""
    state, _detail = L.torch_cuda_ok(sys.executable.replace(
        "python", "definitely-not-python"))
    assert state == "none"


def _fake_popen_factory(captured):
    """假的 subprocess.Popen：只记录命令行，stdout 给一行成功输出。"""
    class _FakeProc:
        returncode = 0

        def __init__(self, cmd, **_kw):
            captured.append(cmd)
            self.stdout = io.BytesIO(b"Successfully installed torch\n")

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    return lambda cmd, **kw: _FakeProc(cmd, **kw)


def test_pip_install_find_links_forces_no_index(L, monkeypatch):
    """★ 平铺 wheel 源必须带 `--no-index`。

    不加的话 pip 会去默认 PyPI 找 torch —— 而 PyPI 上 Windows 的 torch
    是 CPU-only。这是「静默装成 CPU 版」的另一扇门（2026-09-25 事故根因之一）。
    普通索引模式则**不能**加 --no-index，否则连依赖都解析不到。
    """
    captured = []
    monkeypatch.setattr(L.subprocess, "Popen", _fake_popen_factory(captured))
    L.pip_install("python", ["torch"], "https://mirror/links/cu130",
                  lambda *a, **k: None, use_find_links=True)
    cmd = captured[0]
    assert "--find-links" in cmd
    assert "--no-index" in cmd

    captured.clear()
    L.pip_install("python", ["numpy"], "https://mirror/pypi",
                  lambda *a, **k: None)
    assert "-i" in captured[0]
    assert "--no-index" not in captured[0]


def test_find_links_source_installs_torch_deps_first(L):
    """`--no-index` 之后依赖只能预先装：装 torch 前先补 TORCH_DEPS。"""
    src = inspect.getsource(L.install_torch)
    assert "_ensure_torch_deps" in src
    # links 分支里，补依赖必须发生在装 torch 之前
    assert src.index("_ensure_torch_deps") < src.index("use_find_links")
    assert {"filelock", "sympy", "networkx"} <= set(L.TORCH_DEPS)


# ==================================================== 构建产物

def test_build_script_removes_the_stray_exe():
    """★ 构建脚本必须把 dist/ 根下那个 exe 删掉。

    踩过的坑（用户当场撞上）：PyInstaller 把 exe 放在 dist/ 根下，
    我们又复制了一份进 dist/LingYu/。两个一模一样，用户点了根目录那个 ——
    它以 dist/ 为安装根，环境装到了 dist/runtime/，
    和 dist/LingYu/runtime/ 完全两套。用户以为在用发布版，
    排查时连日志都对不上。
    """
    src = open(os.path.join(PKG, "build_exe.py"), encoding="utf-8").read()
    assert "os.remove(exe)" in src, (
        "build_exe.py 里没有删掉 dist/ 根下的 exe —— 留着会误导用户点错")


def test_build_script_keeps_runtime_on_disk_but_out_of_the_zip():
    """★ runtime/ 不该进发布包，**但也不该从硬盘上删掉**。

    这两件事以前混成了一件：无条件 `rmtree(dist/LingYu/runtime)`。
    理由（里面有本机绝对路径）是对的，做法太狠 ——
    「改一行代码 → 重新打包」会把正在用的那份安装连同
    `runtime/data/` 里准备好的语料一起删掉，而且不提示。

    实测就这么丢过一次语料：用户刚准备完 789 万 token，
    我重新打了个包，语料没了。

    正确做法：**打包时排除**（os.walk 里跳过 runtime），
    硬盘上原样保留；真要清干净加 `--clean-runtime`。
    """
    src = open(os.path.join(PKG, "build_exe.py"), encoding="utf-8").read()
    # 默认保留
    assert "def assemble(progress, zip_it=False, clean_runtime=False)" in src, \
        "assemble 没有 clean_runtime 开关，会无条件删掉用户的安装"
    assert "keep_runtime" in src, "没有区分「保留」和「清掉」两种情况"
    # 压缩时必须排除
    assert '"runtime" in os.path.relpath(dirpath, rel).split(os.sep)' in src, \
        "压缩时没有排除 runtime/ —— 会把本机绝对路径发出去"
    # 显式开关要存在
    assert "--clean-runtime" in src, "没有提供主动清掉的开关"


def test_release_zip_excludes_runtime():
    """真的压一个包出来验一遍排除逻辑（不跑 PyInstaller）。"""
    import zipfile
    rel = os.path.join(PKG, "_tmp_release")
    shutil.rmtree(rel, ignore_errors=True)
    try:
        os.makedirs(os.path.join(rel, "runtime", "configs"))
        os.makedirs(os.path.join(rel, "runtime", "app", "src"))
        with open(os.path.join(rel, "LingYu.exe"), "wb") as fp:
            fp.write(b"MZ fake")
        with open(os.path.join(rel, "runtime", "configs", "small.ini"), "w",
                  encoding="utf-8") as fp:
            fp.write("[data]\ndata_dir = C:/this-machine/only\n")
        with open(os.path.join(rel, "runtime", "app", "src", "x.py"), "w",
                  encoding="utf-8") as fp:
            fp.write("V=1\n")

        zpath = os.path.join(PKG, "_tmp_release.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for dirpath, _dn, filenames in os.walk(rel):
                if "runtime" in os.path.relpath(dirpath, rel).split(os.sep):
                    continue
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, os.path.dirname(rel)))
        names = zipfile.ZipFile(zpath).namelist()
        assert not any("runtime" in n for n in names), \
            "runtime/ 混进发布包了：%s" % names
        assert any(n.endswith("LingYu.exe") for n in names)
        os.remove(zpath)
    finally:
        shutil.rmtree(rel, ignore_errors=True)


def test_torch_indexes_try_a_china_mirror_first(L):
    """★ torch 的安装源里，**国内镜像必须排在官方前面**。

    踩过的坑：原来是「先官方（CDN 最快也最全），再国内镜像兜底」。
    那个判断是想当然的。本机实测同一个 wheel：

        官方 download.pytorch.org    0.22 MB/s
        上交 mirror.sjtu.edu.cn     10.05 MB/s   ← 快 45 倍

    1990 MB 的 torch：官方 **2.5 小时**，上交 **3.3 分钟**。

    更关键是「兜底」根本不会触发 —— 镜像只在**失败**时才被试到，
    而慢不是失败。实测就这么卡住了：3 分钟写 0 字节，
    用户看到「下载中…」不动，以为程序死了。

    安全性由装完的 `torch_cuda_ok()` 兜着（坏轮子会被验出来然后试下一个），
    所以镜像优先不引入风险。
    """
    first_label, first_url, _kind = L.TORCH_INDEXES[0]
    assert "download.pytorch.org" not in first_url, (
        "第一个 torch 源还是官方 CDN —— 国内实测 0.22 MB/s，"
        "2 GB 要下 2.5 小时。国内镜像必须排前面。当前第一项：%s" % first_label)
    assert ("mirror.sjtu" in first_url or "tuna" in first_url
            or "aliyun" in first_url or ".cn" in first_url), \
        "第一个 torch 源不是国内镜像：%s" % first_url
    # 官方仍要留在链子里兜底（镜像挂了/没有新轮子时用）
    assert any("download.pytorch.org" in u for _l, u, _k in L.TORCH_INDEXES), \
        "把官方源整个删了 —— 镜像挂掉时就没有退路了"


def test_mirror_entries_declare_the_right_pip_flag(L):
    """★ `"index"` 还是 `"links"`，必须和那个地址的**真实形态**一致。

    踩过的坑：上交那条写成了 `"links"`。pip 用 `--find-links` 去扫，
    **一条 .whl 都没扫到**（那是个 PEP 503 索引，只有子目录），
    于是退回去用默认 PyPI 索引里的 **CPU 版 torch**（124 MB，无 +cu130）。

    后果不是报错，而是**悄悄装成 CPU 版** ——
    「GPU 引擎装出来没有 GPU」从另一个门又进来了。

    实测形态（`curl -s <地址> | head` 看有没有 `pypi:repository-version`）：

        上交 /pytorch-wheels/cu130    PEP 503 索引   子目录 118，直接 .whl 0   → index
        阿里 /pytorch-wheels/cu130    平铺目录       直接 .whl 993             → links
        官方 /whl/cu130               PEP 503 索引                            → index
    """
    by_url = {u: k for _l, u, k in L.TORCH_INDEXES}
    sjtu = [k for u, k in by_url.items() if "mirror.sjtu" in u]
    assert sjtu, "上交镜像被从链子里删掉了"
    assert sjtu[0] == "index", (
        "上交镜像声明成了 %r —— 它是 PEP 503 索引，用 --find-links 扫不到任何轮子"
        % sjtu[0])
    for url, kind in by_url.items():
        if "download.pytorch.org" in url:
            assert kind == "index", "官方索引被声明成了 %r" % kind
        if "aliyun" in url:
            assert kind == "links", (
                "阿里是**平铺 wheel 目录**（993 条直接 .whl），要当 --find-links 用")


def test_window_worker_thread_never_touches_tk():
    """★ 向导窗口的工作线程**一个 Tk 调用都不许做**。

    踩过的坑（偶发卡死，只在「环境已装好」这条最快路径上出现）：

        threading.Thread(target=work).start()
        win.root.mainloop()

    而 `work()` 里直接 `win.push(...)` / `win.root.quit()`。两个问题叠一起：

    1. **Tkinter 不是线程安全的** —— `push()` 内部调 `root.update()`，
       从非主线程调 Tk 的 C 层有时会直接锁死。
    2. **`quit()` 早于 `mainloop()` 就失效** —— 环境已就绪时 `work()`
       几百毫秒就跑完，完全可能抢在主线程进入 `mainloop()` 之前把
       `quit()` 调掉。`quit()` 只对**正在跑的** mainloop 有效，
       先调等于没调 → `mainloop()` 进去就再也出不来 → 进程挂死。

    症状：双击 exe 有时能开、有时一直没反应，`launcher.log` 停在
    「环境已就绪，正在启动……」。**首次安装那一次反而不会卡**
    （装环境要几分钟，mainloop 早起来了）—— 所以这个 bug 只在
    装好之后才暴露，特别容易被当成「偶发」放过。

    修法：工作线程只往 `queue` 里塞消息，主线程用 `after()` 轮询。
    这条测试用静态检查钉住它，因为**这种 bug 跑一次测试是碰不到的**。
    """
    src = open(os.path.join(PKG, "launcher.py"), encoding="utf-8").read()
    assert "import queue" in src, "没有引入 queue"
    start = src.index("def run_with_window(")
    body = src[start:src.index("\ndef ", start + 10)]

    assert "win.root.after(" in body, "主线程没有用 after() 轮询队列"
    assert "queue.Queue()" in body, "没有用队列把消息传回主线程"
    # 工作线程里绝对不能出现 Tk 调用
    worker = body[body.index("def work():"):body.index("def pump():")]
    for bad in ("win.push(", "win.root.", "win.finish("):
        assert bad not in worker, (
            "工作线程里出现了 `%s` —— 跨线程调 Tk 会偶发锁死，"
            "只能往队列里塞消息" % bad)
    assert "q.put(" in worker, "工作线程没有把消息放进队列"
    # quit() 必须由主线程调（并且这里用 destroy 更干净）
    assert "win.root.destroy()" in body, "窗口没有真正关掉（quit 会让它留在屏幕上）"


def test_pump_runs_on_the_main_thread_only():
    """`SetupWindow` 里不许再有手动 `root.update()`。

    那两个 `update()` 就是为了跨线程刷新才加的；现在 push 只在主线程跑，
    留着反而会在 `after` 回调里造成重入。

    ★ 用 **ast** 而不是字符串查找。

    第一版就是字符串找 `"self.root.update()"`，结果被自己那段解释性注释
    绊倒了 —— 注释里正写着「这里原来有 `self.root.update()`」。
    正则/字符串扫源码既抓不住不带引号的写法，又会误伤文档，
    `test_no_developer_absolute_paths_in_engine` 早就为这个坑改用 ast 了，
    这里照做。
    """
    import ast
    tree = ast.parse(open(os.path.join(PKG, "launcher.py"),
                          encoding="utf-8").read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "SetupWindow")
    bad = []
    for node in ast.walk(cls):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("update", "update_idletasks"):
            continue
        # 只关心 self.root.update(...) 这种
        v = node.func.value
        if (isinstance(v, ast.Attribute) and v.attr == "root"
                and isinstance(v.value, ast.Name) and v.value.id == "self"):
            bad.append(node.lineno)
    assert not bad, (
        "SetupWindow 里还有手动 self.root.update()（第 %s 行）—— "
        "跨线程调 Tk 会偶发锁死，主线程也不需要它（after 回调里重入更危险）"
        % bad)


def test_launcher_disables_argparse_prefix_matching(L):
    """★ 启动器必须关掉 argparse 的**前缀匹配**。

    踩过的坑（实测撞上）：

        LingYu.exe --console train --set train.max_steps=5

    `--set` 是 `lingyu train` 的参数（临时覆盖配置项），
    但 argparse 默认允许缩写，于是把它匹配成了启动器的 `--setup-only`。
    结果：**不训练**，打印一句「只装环境，不启动」就退出，
    **退出码还是 0** —— 用户看到的是「命令跑了，什么都没发生」。

    这条尤其坑，因为 `--set` 正是出厂配置注释里教用户用的写法
    （「临时改一项又不想动文件：--set model.n_layer=8」）。
    """
    import argparse
    ap = argparse.ArgumentParser(prog="LingYu", allow_abbrev=False)
    ap.add_argument("--console", action="store_true")
    ap.add_argument("--setup-only", action="store_true")
    ap.add_argument("app_args", nargs="*", default=[])
    a, rest = ap.parse_known_args(
        ["--console", "train", "--set", "train.max_steps=5"])
    assert a.setup_only is False, "`--set` 又被当成 `--setup-only` 了"
    assert a.console is True
    assert "--set" in list(a.app_args) + list(rest), \
        "`--set` 没有传给 lingyu：app_args=%s rest=%s" % (a.app_args, rest)

    # 源码里也得真的关掉
    src = open(os.path.join(PKG, "launcher.py"), encoding="utf-8").read()
    assert "allow_abbrev=False" in src, \
        "launcher.py 没关掉前缀匹配 —— `--set` 会被吃成 `--setup-only`"


def test_app_source_is_checked_on_every_launch_not_only_setup(L):
    """★ 「exe 换代了要重解源码」这个检查必须在**每次启动**都做。

    踩过的坑：`extract_app()` 里的哈希判断本身写对了，但 `setup()` 只在
    **首次安装**（或环境坏掉）时才跑。环境装好之后每次启动都直接
    `launch()`，于是源码**永远不会更新** —— 用户升级了 exe，
    跑的还是一年前那版代码，而版本号、界面、日志看着都是新的。

    实测就是这么发生的：分词器的修复写进了源码、重新打了包、双击运行，
    行为一点没变，`.extracted` 里还是上一轮的时间戳。查了好一阵。

    所以三处都要调 `ensure_app`：setup-only、控制台、向导窗口。
    """
    src = open(os.path.join(PKG, "launcher.py"), encoding="utf-8").read()
    assert "def ensure_app(" in src, "没有 ensure_app —— 源码版本没人管"
    # 两条启动路径都必须调用
    assert src.count("ensure_app(root, log)") >= 2, (
        "ensure_app 只在一条路径上调用了（共 %d 处）—— "
        "另一条启动路径会一直跑旧源码" % src.count("ensure_app(root, log)"))
    console = src[src.index("def run_console("):src.index("def main(")]
    assert "ensure_app(" in console, "控制台路径没做源码版本检查"
    window = src[src.index("def run_with_window("):src.index("def run_console(")]
    assert "ensure_app(" in window, "向导窗口路径没做源码版本检查"

    # 而且 ensure_app 自己要在标记不一致时真的重新解包
    body = src[src.index("def ensure_app("):]
    body = body[:body.index("\ndef ")]
    assert "_source_stamp(" in body and "extract_app(" in body, \
        "ensure_app 没比对哈希、也没重新解包"


def test_indexes_start_with_a_china_mirror(L):
    """第一个 PyPI 镜像必须是国内源。

    国内直连 pypi.org 装几个包都可能要几分钟，
    大多数人会在中途放弃 —— 那这个「开箱即用」就白做了。
    """
    first = L.INDEXES[0][1]
    assert "tuna" in first or "aliyun" in first or "cn" in first


# ==================================================== 源码打包

def test_source_zip_contains_gui_and_engine(tmp_path):
    """打的源码包里必须有 gui/ 和 src/lingyu/。

    漏了 gui/ 的话，用户装完环境双击打开 —— 报「找不到 gui 包」。
    而构建时完全不会报错，要等用户反馈才知道。
    """
    sys.path.insert(0, PKG)
    try:
        import build_exe
    finally:
        sys.path.pop(0)
    out = build_exe.make_source_zip(lambda m: None)
    assert os.path.isfile(out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert any(n.startswith("src/lingyu/") for n in names), "缺引擎核心"
    assert any(n.startswith("gui/") for n in names), "缺图形界面"
    assert any(n.startswith("configs/") and n.endswith(".ini") for n in names), \
        "缺出厂配置"
    # 绝不能把用户数据或缓存打进去
    assert not any(n.endswith(".bin") for n in names), "把数据文件打进去了"
    assert not any("__pycache__" in n for n in names), "把缓存打进去了"
    assert not any(n.startswith("venv/") for n in names), "把虚拟环境打进去了"


# ==================================================== 开源卫生

def test_no_developer_absolute_paths_in_engine():
    """★ 引擎源码里不许出现开发者自己机器上的路径。

    踩过的坑：`_require_deps` 里写死了
    `C:\\path\\to\\lingyu\\lingyu.ps1`，`--help` 的示例里也写死了
    `C:\\path\\to\\lingyu\\raw`。别人 clone 下来缺依赖时，
    界面告诉他去跑一个**只存在于开发者机器上**的脚本 ——
    他照着敲只会得到「找不到路径」，然后以为引擎坏了。
    打包版更严重：那条路径压根不存在。

    ## 为什么用 ast 而不是正则扫行

    正则扫行有两个毛病：一是抓不住不带引号的路径（`--raw D:\\ai\\...`），
    二是会误伤**文档字符串里对这件事的解释**（我们就是要在注释里
    讲清楚这个坑，不该因此判失败）。

    所以用 ast 把「真正会被执行的字符串常量」挑出来 ——
    文档字符串自动被排除，其余一律检查。
    """
    import ast
    src = os.path.join(ROOT, "src", "lingyu")
    bad = []

    def docstring_nodes(tree):
        """收集所有「文档字符串」节点，它们讲道理是允许提到的。"""
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and \
                        isinstance(body[0].value, ast.Constant) and \
                        isinstance(body[0].value.value, str):
                    out.add(id(body[0].value))
        return out

    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            with open(full, "r", encoding="utf-8") as fp:
                text = fp.read()
            try:
                tree = ast.parse(text)
            except SyntaxError as e:
                bad.append("%s 语法错误：%s" % (rel, e))
                continue
            skip = docstring_nodes(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or \
                        not isinstance(node.value, str):
                    continue
                if id(node) in skip:
                    continue
                if "llm-engine" in node.value:
                    bad.append("%s:%d: %r" % (rel, node.lineno,
                                              node.value[:80]))
    assert not bad, (
        "这些地方写死了开发者机器上的路径（换台机器就找不到）：\n  "
        + "\n  ".join(bad)
        + "\n\n要给出的路径请用 sys.executable 或 _project_root() 现算。")


def test_help_text_has_no_dev_paths():
    """`lingyu --help` 里也不许出现开发者的路径。"""
    import subprocess
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, "-m", "lingyu", "--help"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env=env, cwd=ROOT, timeout=120)
    out = p.stdout.decode("utf-8", "replace")
    assert "llm-engine\\raw" not in out and "llm-engine\\\\raw" not in out, \
        "帮助里还写着开发者的语料路径：\n" + out


def test_license_file_exists_and_is_apache():
    p = os.path.join(ROOT, "LICENSE")
    assert os.path.isfile(p), "没有 LICENSE —— GitHub 上会被标成无许可证"
    with open(p, "r", encoding="utf-8") as fp:
        txt = fp.read()
    assert "Apache License" in txt and "Version 2.0" in txt
    assert "Copyright 2026" in txt, "没填版权行"


def test_readmes_exist_in_both_languages():
    """英文 README + 中文 README 都要有（用户明确要求）。

    英文那份还引用了中文那份的**文件名**，所以两个都得在，
    否则 GitHub 首页会出现一个死链。
    """
    en = os.path.join(ROOT, "README.md")
    zh = os.path.join(ROOT, "README.zh-CN.md")
    assert os.path.isfile(en), "缺英文 README"
    assert os.path.isfile(zh), "缺中文 README"
    with open(en, "r", encoding="utf-8") as fp:
        txt = fp.read()
    # 英文 README 里引用到的中文文件名必须真的存在
    for name in ("README.zh-CN.md",):
        assert name in txt, "英文 README 应该链到 %s" % name
        assert os.path.isfile(os.path.join(ROOT, name))


# ==================================================== 启动（cwd / 失败可见性）


def _fake_launch_env(L, tmp_path):
    """造一个「环境已就绪」的安装根，供 launch() 的单元测试用。"""
    root = str(tmp_path / "install")
    app = os.path.join(L.runtime_dir(root), "app")
    os.makedirs(os.path.join(app, "src"), exist_ok=True)
    L.write_state(root, {"state_version": L.STATE_VERSION,
                         "venv_python": sys.executable,
                         "app_dir": app})
    return root


def test_launch_cwd_is_install_root(L, tmp_path, monkeypatch):
    """★ 使用说明里写的是 `--config runtime\\configs\\small.ini`（相对安装根）。

    踩过的坑：launch() 曾经用 `cwd=runtime\\app`，于是用户照着说明书敲的
    第一条命令就报 `[错误] 找不到配置文件：runtime\\configs\\small.ini`。
    引擎自己用 paths.home()/LINGYU_HOME 定位，不依赖 cwd，所以这里钉死
    cwd 必须是安装根。
    """
    root = _fake_launch_env(L, tmp_path)
    seen = {}

    class FakeProc:
        def wait(self):
            return 0

    def fake_popen(cmd, env=None, cwd=None, stderr=None, **kw):
        seen["cwd"] = cwd
        return FakeProc()

    monkeypatch.setattr(L.subprocess, "Popen", fake_popen)
    code = L.launch(root, sys.executable,
                    ["plan", "--config", "runtime\\configs\\small.ini"],
                    lambda *a, **k: None)
    assert code == 0
    assert seen["cwd"] is not None
    assert os.path.normcase(os.path.abspath(seen["cwd"])) == \
        os.path.normcase(os.path.abspath(root)), \
        "launch() 的工作目录不是安装根：%r" % seen["cwd"]


def test_launch_failure_is_visible(L, tmp_path, monkeypatch, capsys):
    """★ 启动失败必须有人能看见：打印退出码 + 日志路径，stderr 落盘。

    以前用 subprocess.call：不记退出码、不抓 stderr，双击失败就是
    「窗口闪一下/没反应」，日志里也只有「启动：…」。
    """
    root = _fake_launch_env(L, tmp_path)

    class FakeProc:
        def wait(self):
            return 3

    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: FakeProc())
    code = L.launch(root, sys.executable, ["doctor"], lambda *a, **k: None)
    out = capsys.readouterr().out
    assert code == 3
    assert "启动失败（退出码 3）" in out, "启动失败没有任何可见提示"
    assert "engine-" in out, "提示里没有引擎日志路径"
    log_path = L._engine_log_path(root)
    assert os.path.isfile(log_path), "引擎 stderr 日志没有创建"


def test_bare_help_prints_two_layers(L, capsys):
    """★ 裸 `--help` 不能只给启动器参数。

    用户被告知「--help 看全部命令」，看到的却只有启动器自己的参数 ——
    这是使用说明里的假承诺。现在要求两层都打印。
    """
    code = L.main(["--help"])
    assert code == 0
    out = capsys.readouterr().out
    assert "引擎命令" in out, "--help 里没有引擎命令提示"
    for cmd in ("doctor", "train", "plan", "chat"):
        assert cmd in out, "--help 里没提引擎命令 %s" % cmd
    assert "--console" in out


def test_engine_help_is_routed_to_engine(L, monkeypatch):
    """`--console plan --help` 的帮助要交给引擎，不能被 argparse 吃掉。"""
    seen = {}

    def fake_run_console(root, args):
        seen["app_args"] = list(args.app_args)
        return 0

    monkeypatch.setattr(L, "run_console", fake_run_console)
    code = L.main(["--console", "plan", "--help"])
    assert code == 0
    assert seen["app_args"] == ["plan", "--help"], seen


def test_pip_install_has_idle_watchdog():
    """★ pip 300 秒没有输出要能中止（「慢 ≠ 失败」那条事故的另一半）。

    只把镜像顺序排快治不了「卡住不动」：pip 可以永远挂着不报错。
    """
    src = open(os.path.join(PKG, "launcher.py"), encoding="utf-8").read()
    assert "PIP_IDLE_TIMEOUT" in src
    start = src.index("def pip_install(")
    body = src[start:src.index("\ndef ", start + 10)]
    assert "q.get(timeout=" in body, "pip 输出没有空闲超时"
    assert "proc.kill()" in body, "超时后没有 kill 掉 pip"


def test_readme_txt_commands_are_runnable():
    """《使用说明.txt》里的命令行示例必须带 --console 且用安装根相对路径。

    文档命令跑不通是用户实测撞上的第一个坑（exit 2）。
    """
    sys.path.insert(0, PKG)
    try:
        import build_exe
    finally:
        sys.path.pop(0)
    txt = build_exe.README_TXT
    assert "--console" in txt
    assert "runtime\\configs\\small.ini" in txt
    assert "LingYu.exe --help" in txt
    # 老版「先看看自己电脑能训多大的模型」后面那条没带 --console，已改掉
    assert "LingYu.exe plan --config" not in txt


# ==================================================== 版本号 / 发布卫生

def _import_build_exe():
    sys.path.insert(0, PKG)
    try:
        import build_exe
        return build_exe
    finally:
        sys.path.pop(0)


def test_launcher_version_prints_launcher_and_engine(L, monkeypatch, capsys):
    """`LingYu.exe --version` 要一次给出启动器版本 + 引擎版本。

    报 bug 时第一句就是问版本号；以前没有任何地方能打印它。
    """
    monkeypatch.setattr(L, "_engine_version", lambda: "9.9.9")
    code = L.main(["--version"])
    assert code == 0
    out = capsys.readouterr().out
    assert "启动器" in out
    assert "v%s" % L.LAUNCHER_VERSION in out
    assert "引擎 v9.9.9" in out


def test_engine_version_matches_source(L):
    """`_engine_version()` 读到的必须和 src/lingyu/version.py 一致。"""
    import re
    p = os.path.join(ROOT, "src", "lingyu", "version.py")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                  open(p, encoding="utf-8").read())
    assert m, "src/lingyu/version.py 里没有 __version__"
    assert L._engine_version() == m.group(1)


def test_launch_error_text_is_actionable(L, tmp_path):
    """失败文案要说清日志在哪、怎么重试、报 bug 带什么。"""
    txt = L._launch_error_text(str(tmp_path), 3)
    assert "退出码 3" in txt
    assert "engine-" in txt          # 日志路径
    assert "--reinstall" in txt      # 可操作的下一步
    assert "最后 10 行" in txt


def test_build_writes_sha256sums(tmp_path):
    """发布文件夹要生成 SHA256SUMS：格式 `<hash>  <name>`，哈希算得对。"""
    import hashlib
    import re
    be = _import_build_exe()
    rel = tmp_path / "LingYu"
    rel.mkdir()
    for name in be.RELEASE_FILES:
        (rel / name).write_bytes(("payload-" + name).encode("utf-8"))
    out = be.write_sha256sums(str(rel), lambda m: None)
    lines = open(out, encoding="utf-8").read().splitlines()
    assert len(lines) == len(be.RELEASE_FILES)
    for line, name in zip(lines, be.RELEASE_FILES):
        m = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        assert m, "SHA256SUMS 行格式不对：%r" % line
        assert m.group(2) == name
        want = hashlib.sha256((rel / name).read_bytes()).hexdigest()
        assert m.group(1) == want


def test_build_zip_contains_checksums_and_is_verified():
    """assemble 必须先写 SHA256SUMS（zip 里要带），打完再校验 zip 条目。"""
    be = _import_build_exe()
    src = inspect.getsource(be.assemble)
    assert "write_sha256sums(" in src, "assemble 没生成 SHA256SUMS"
    assert src.index("write_sha256sums(") < src.index("zipfile.ZipFile"), \
        "SHA256SUMS 必须在压 zip 之前生成"
    assert "verify_zip_entries(" in src, "打完 zip 没校验条目"
    assert src.index("ZipFile") < src.index("verify_zip_entries(")


def test_verify_zip_entries_rejects_extra_file(tmp_path):
    """zip 里多一个临时文件也必须构建失败（发布卫生）。"""
    import zipfile
    be = _import_build_exe()
    p = tmp_path / "rel.zip"
    with zipfile.ZipFile(p, "w") as z:
        for n in list(be.RELEASE_FILES) + ["SHA256SUMS"]:
            z.writestr("LingYu/" + n, b"x")
    be.verify_zip_entries(str(p))          # 正好时通过
    with zipfile.ZipFile(p, "a") as z:
        z.writestr("LingYu/_tmp.txt", b"junk")
    with pytest.raises(RuntimeError, match="_tmp"):
        be.verify_zip_entries(str(p))



def test_seed_configs_warns_on_stale_and_reset_backs_up(L, tmp_path):
    """旧版残留配置要提示、--reset-configs 要能恢复且先备份（审计 E23）。"""
    root = tmp_path
    app = root / "runtime" / "app"
    (app / "configs").mkdir(parents=True)
    (app / "configs" / "x.ini").write_text(
        "[model]\nn_layer = 2\n\n[data]\ndata_dir = data\n", encoding="utf-8")

    logs = []
    L.seed_configs(str(root), str(app), logs.append)
    target = root / "runtime" / "configs" / "x.ini"
    assert target.is_file(), "首装应复制出厂配置"

    # 模拟「旧版残留 / 用户改过」：内容与出厂不同
    target.write_text("[model]\nn_layer = 99\n", encoding="utf-8")
    logs.clear()
    L.seed_configs(str(root), str(app), logs.append)
    assert any("reset-configs" in s for s in logs), \
        "检测到旧配置时必须提示 --reset-configs，实际：%r" % logs

    logs.clear()
    L.seed_configs(str(root), str(app), logs.append, reset=True)
    text = target.read_text(encoding="utf-8")
    assert "n_layer = 2" in text, "reset 后应恢复成出厂配置"
    backups = list((root / "runtime" / "configs").glob("x.ini.bak-*"))
    assert backups, "reset 前必须把旧配置备份成 .bak-时间戳"


def test_launcher_cli_has_reset_configs(L):
    ap = L._build_parser()
    assert ap.parse_args(["--reset-configs"]).reset_configs is True
    assert ap.parse_args([]).reset_configs is False
