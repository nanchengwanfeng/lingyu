# -*- coding: utf-8 -*-
r"""全局状态：配置、设置、信号总线。

## 为什么需要它

八个页面之间有依赖：

    语料准备完  -> 概览页的语料卡片要更新、训练页要允许开跑
    训练开始    -> 监控页要开始画曲线、侧边栏状态点要变
    训练存断点  -> 断点页要刷新、SFT/聊天/续写页的下拉框要多一项

如果让页面互相 import 直接调用，很快就会出现循环依赖和「谁该刷新谁」
的混乱。所以这里做一个**信号总线**：页面只管发信号和听信号，
互相不认识。

## 设置存哪

存到用户目录下的 `~/.lingyu/gui.json`，**不放进项目目录** ——
项目目录是 git 仓库，用户的个人偏好不该出现在 `git status` 里。
"""

import json
import os
import time

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from .i18n import set_lang

SETTINGS_DIR = os.path.join(os.path.expanduser("~"), ".lingyu")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "gui.json")

DEFAULTS = {
    "lang": "zh_CN",
    "theme": "dark",
    "config": "configs/small.ini",
    "raw_dir": "",
    "data_dir": "",
    "out_dir": "",
    "chat_dir": "",
    "window_w": 1280,
    "window_h": 820,
}


def load_settings():
    s = dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fp:
            s.update(json.load(fp) or {})
    except (OSError, ValueError):
        pass
    return s


def save_settings(s):
    try:
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fp:
            json.dump(s, fp, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False


class _ProbeSignals(QObject):
    done = Signal(dict)


def _smi_info(timeout=2.0):
    """用 nvidia-smi 快速探显卡（不得进程、不占显存、不需要先 import torch）。

    1.4 后 GUI 也走动作窗口了；用户点「重新检测」时如果又要等 5 秒
    import torch，体感就像「点了没反应」。nvidia-smi 一般 0.1–0.3 秒。
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        first = out.stdout.strip().splitlines()[0]
        name, total_mb, free_mb = [x.strip() for x in first.split(",")[:3]]
        total = int(float(total_mb)) * 1024 ** 2
        free = int(float(free_mb)) * 1024 ** 2
        return {
            "ok": True,
            "device": "%s（%.1f GB）" % (name, total / 1024 ** 3),
            "device_short": name.replace("NVIDIA ", ""),
            "vram_free": free, "vram_total": total,
            "vram_text": "%.1f / %.1f GB" % (free / 1024 ** 3,
                                             total / 1024 ** 3),
        }
    except Exception:
        return None


class _EnvProbe(QRunnable):
    """在**后台线程**里 import torch，拿显卡和显存信息。

    ## 为什么必须开线程

    `import torch` 在 8GB 的机器上要 3~8 秒。如果放在主线程：

        app.exec() 之前 import  -> 窗口压根没出现，用户以为没启动
        app.exec() 之后 import  -> 窗口出来了，但整界面冻结 3~8 秒，
                                   点哪都没反应，看起来像卡死

    两种都不能接受。所以丢到线程里，界面先显示「检测中…」，
    拿到结果再填数字。
    """

    def __init__(self):
        super().__init__()
        self.sig = _ProbeSignals()

    def run(self):
        info = {"ok": False, "device": "—", "device_short": "—",
                "vram_free": 0, "vram_total": 0, "vram_text": "—",
                "torch": "未安装", "python": "", "error": ""}
        import platform
        info["python"] = platform.python_version()
        smi = _smi_info()
        if smi:
            info.update(smi)
        try:
            import torch
            info["torch"] = torch.__version__
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                props = torch.cuda.get_device_properties(0)
                # nvidia-smi 的数字通常是 0.1 秒内拿到的、且与驱动口径一致；
                # torch 再探一次是为了拿到 sm 版本和设备名补充。两者都拿到时，
                # 以 nvidia-smi 的显存为准（它不占显存、也不会被 CUDA 上下文
                # 自身的缓存块干扰）。
                if not smi:
                    info.update({
                        "vram_free": int(free), "vram_total": int(total),
                        "vram_text": "%.1f / %.1f GB" % (
                            free / 1024 ** 3, total / 1024 ** 3),
                    })
                info.update({
                    "ok": True,
                    "device": "%s（%.1f GB，sm_%d%d）" % (
                        props.name, props.total_memory / 1024 ** 3,
                        props.major, props.minor),
                    "device_short": props.name.replace("NVIDIA ", ""),
                })
            elif not smi:
                info["device"] = "CPU（没有可用显卡）"
                info["device_short"] = "CPU"
                info["ok"] = True
        except Exception as e:
            info["error"] = str(e)
        try:
            self.sig.done.emit(info)
        except RuntimeError:
            pass


class Context(QObject):
    """页面之间共享的一切。全局单例 `ctx`。"""

    # ---- 信号总线 ----
    corpus_changed = Signal()      # 语料目录内容变了
    config_changed = Signal()      # 当前配置换了
    theme_changed = Signal(str)    # 主题换了
    lang_changed = Signal(str)     # 语言换了
    checkpoints_changed = Signal()  # 断点目录有新东西
    task_started = Signal(str)     # 任务开始（任务名）
    task_finished = Signal(str, int)  # 任务结束（任务名, 退出码）
    # 任务事件总线：不管哪个窗口/哪次重建，页面都从 ctx 收任务输出。
    # 见 attach_runner 的说明（审计 1.7 的根因）。
    task_line = Signal(str, str, str)          # 任务名, 文本, 级别
    task_progress = Signal(str, dict)          # 任务名, 进度信息
    task_validation = Signal(str, int, float)  # 任务名, step, loss
    env_ready = Signal(dict)       # 后台环境探测完成

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        set_lang(self.settings.get("lang", "zh_CN"))
        self._config = None
        self._config_path = None
        self.running_task = None       # 当前跑着的任务名，空 = 空闲
        #: 正在跑的任务的 Runner。**必须挂在 ctx 上**：任务开跑后页面可能
        #: 因为换语言/换主题被重建，新页面实例拿不到旧 Runner —— 那样「停止」
        #: 按钮会消失、任务却还在后台跑（审计 1.7）。
        self.task_runner = None
        #: 监控页当前看的是哪条数据线（"pretrain" / "sft"）。挂在 ctx 上，
        #: 窗口换语言/主题重建后仍停在同一个数据源（审计 1.2）。
        self.monitor_kind = self.settings.get("monitor_kind", "pretrain") or "pretrain"
        self.project_root = self._find_root()
        # 环境信息由后台线程填。**初始值必须是「不知道」而不是假数字** ——
        # 显示 0 GB 会让用户以为显卡没被识别。
        self.gpu = {"ok": False, "device": "检测中…", "device_short": "…",
                    "vram_free": 0, "vram_total": 0, "vram_text": "检测中…",
                    "torch": "检测中…", "python": "", "probing": True}
        self._probing = False
        self._probe = None
        #: 上次环境探测完成的时间戳（time.time()）。概览页据此显示「检测于 …」
        #: 并提供「重新检测」按钮 —— 显存是变化最快的数字，不能让用户重开界面。
        self.gpu_ts = 0.0

    # ------------------------------------------------------------ 环境探测
    def refresh_env(self):
        """重新探测一次环境（显存/设备）。正在探测时会忽略。"""
        self.start_env_probe()

    def start_env_probe(self):
        """启动一次后台环境探测。

        同一时间只允许一个在跑；**跑完之后允许再次调用**（例如用户
        装好 torch/驱动后点「重新检测」，不用重启界面）—— 审计 G-15。
        """
        if self._probing:
            return
        self._probing = True
        probe = _EnvProbe()
        # ★ 必须留一个强引用。QThreadPool 在 runnable 跑完后会**自动删除**
        #   它的 C++ 对象（autoDelete 默认开），信号载体 `probe.sig`
        #   跟着被回收，于是 done 信号发出来也没人收 ——
        #   表现为界面上永远停在「检测中…」。
        self._probe = probe
        probe.sig.done.connect(self._on_env)
        QThreadPool.globalInstance().start(probe)

    def _on_env(self, info):
        # ★ 必须复位：否则探测失败一次以后就再也探不了（审计 G-15）。
        self._probing = False
        self._probe = None
        info["probing"] = False
        self.gpu = info
        self.gpu_ts = time.time()
        self.env_ready.emit(info)

    def shutdown(self):
        """应用退出前调用：等后台探针线程收尾。

        ★ 为什么需要：窗口开了 50ms 后探针启动，用户若立刻关窗口，进程
        退出时线程里的 `import torch` 还要几秒；Qt 的 QRunnable 在 Python
        解释器收尾阶段回调会直接 0xC0000005（实测 `pytest tests/test_gui.py`
        全绿但退出码 -1073741819）。等它跑完再退，最多多等 5 秒。
        """
        self._probing = False
        try:
            QThreadPool.globalInstance().waitForDone(5000)
        except Exception:
            pass

    def vram_free(self):
        """可用显存字节数。**没探测到就返回 0，绝不猜一个数。**"""
        return int(self.gpu.get("vram_free") or 0)

    # ------------------------------------------------------------ 路径
    def _find_root(self):
        """用户数据根目录（打包版 = `runtime/`，开发版 = 项目根）。

        ★ 走 `lingyu.paths`，**不要自己按 `__file__` 往上数**。
          自己数出来的在打包版是 `runtime/app` —— 那是源码的家，
          不是用户语料和配置的家。这两个目录长得都挺合理，
          所以出错时完全没有症状，只是「东西不在你以为的地方」。
        """
        from lingyu import paths
        return paths.home()

    def abspath(self, p):
        """把相对路径按**用户数据根**解析（配置文件里写相对路径时要用）。"""
        from lingyu import paths
        return paths.abspath(p)

    def save(self):
        save_settings(self.settings)

    # ------------------------------------------------------------ 配置
    @property
    def config_path(self):
        return self._config_path

    def load_config(self, path=None):
        """读配置（带 spec 默认值补全）。失败时返回 None 并把错误抛给调用方。

        ## 找配置文件的规则

        显式传 `path` 就按它解析；否则拿设置里存的名字去**逐个候选根找**
        （见 `lingyu.paths.find_config`）。

        ★ 找到之后把**绝对路径**写回设置。
          以前存的是相对路径 `configs/small.ini`，再按源码根解析成
          `runtime/app/configs/small.ini` —— 出厂那份带相对路径的副本。
          于是界面读的配置和启动器给用户准备的配置**根本不是同一个文件**，
          用户在界面上改的参数也没落到该落的地方。
        """
        from lingyu import paths
        from lingyu.config import load_config, resolve
        if path:
            full = paths.abspath(path)
        else:
            full = paths.find_config(
                self.settings.get("config") or paths.DEFAULT_CONFIG)
        cfg = resolve(load_config(full))
        self._config = cfg
        self._config_path = full
        self.settings["config"] = full
        self.config_changed.emit()
        return cfg

    @property
    def config(self):
        if self._config is None:
            try:
                self.load_config()
            except Exception:
                self._config = {}
        return self._config

    def save_config(self, cfg=None, path=None):
        from lingyu.config import save_config
        cfg = cfg or self.config
        path = path or self._config_path
        # 把内部用的临时键去掉，别写进用户的配置文件
        clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
        save_config(path, clean)
        return path

    def load_custom_config(self):
        """切到「自定义」：用 spec 默认值，不载入任何配置文件。"""
        from lingyu.config import resolve
        self._config = resolve({})
        self._config_path = None
        self.config_changed.emit()
        return self._config

    def ensure_config_file(self):
        """自定义配置还没有落盘文件时，写一份到用户数据根的 configs/。"""
        if self._config_path:
            return self._config_path
        from lingyu import paths
        path = os.path.join(paths.configs_dir(), "custom.ini")
        self.save_config(self.config, path)
        self._config_path = path
        self.settings["config"] = path
        self.save()
        return path

    # ------------------------------------------------------------ 便捷取值
    def get(self, sec, key, default=None):
        from lingyu.config import get
        return get(self.config, sec, key, default)

    def data_dir(self):
        return self.abspath(self.get("data", "data_dir", "") or "")

    def tokenizer_path(self):
        tp = self.get("data", "tokenizer_path", "") or ""
        if tp:
            return self.abspath(tp)
        dd = self.data_dir()
        return os.path.join(dd, "tokenizer.txt") if dd else ""

    def out_dir(self):
        return self.abspath(self.get("train", "out_dir", "out") or "out")

    def sft_out_dir(self):
        return self.abspath(self.get("post", "out_dir", "out_sft") or "out_sft")

    def has_corpus(self):
        return self.corpus_status()[0]

    def corpus_status(self):
        """语料到底在不在。返回 `(在不在, 查的哪个目录, 为什么不在)`。

        ## 为什么非要带上「哪个目录」

        踩过的坑：打包版的配置指向 `runtime/app/data1`，而语料准备页
        把数据写在 `runtime/data`。两个目录都合法、都真实存在、
        谁都没报错 —— 只是一个空一个满。

        这时界面上只有一句「还没准备语料」，用户只会说
        **「我明明刚准备完啊」**，然后两边都查不出问题。

        把「查的是哪个目录」和「为什么不算数」一起显示出来，
        这类问题一眼就能看出来。
        """
        dd = self.data_dir()
        if not dd:
            return False, "", "配置里没写 [data] data_dir"
        if not os.path.isdir(dd):
            return False, dd, "这个目录不存在"
        if not os.path.isfile(os.path.join(dd, "train.bin")):
            return False, dd, "目录里没有 train.bin"
        return True, dd, ""

    def list_checkpoints(self, d=None, pattern=(".pt",)):
        """列出断点文件，按修改时间倒序。"""
        d = d or self.out_dir()
        out = []
        if not d or not os.path.isdir(d):
            return out
        try:
            names = os.listdir(d)
        except OSError:
            return out          # 没权限/目录瞬间消失：当成空目录，别让界面崩
        for name in names:
            if not name.endswith(pattern):
                continue
            p = os.path.join(d, name)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append({"path": p, "name": name, "size": st.st_size,
                        "mtime": st.st_mtime})
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    def runs(self):
        """列出跑过的训练：out 根下每个含 history.csv 的目录。

        返回 [{"dir", "name", "kind", "history", "meta"}, …]，按修改时间倒序。
        `kind` 取自 run.json（pretrain / sft）；没有 run.json 的老目录按名字猜。
        分析页只读这份列表，不写任何东西。
        """
        import json as _json
        seen, out = set(), []
        roots = set()
        for d in (self.out_dir(), self.sft_out_dir()):
            if d:
                roots.add(os.path.dirname(os.path.abspath(d)))
                roots.add(os.path.abspath(d))
        for root in roots:
            try:
                names = sorted(os.listdir(root))
            except OSError:
                continue
            for name in names:
                d = os.path.join(root, name)
                hist = os.path.join(d, "history.csv")
                if not os.path.isfile(hist) or d in seen:
                    continue
                seen.add(d)
                meta = {}
                try:
                    p = os.path.join(d, "run.json")
                    if os.path.isfile(p):
                        with open(p, encoding="utf-8") as fp:
                            meta = _json.load(fp) or {}
                except Exception:
                    meta = {}
                kind = str(meta.get("kind") or "")
                if kind not in ("pretrain", "sft"):
                    low = name.lower()
                    kind = "sft" if ("sft" in low or "post" in low) else "pretrain"
                try:
                    mtime = os.path.getmtime(hist)
                except OSError:
                    mtime = 0
                out.append({"dir": os.path.abspath(d), "name": name,
                            "kind": kind, "history": hist, "meta": meta,
                            "mtime": mtime})
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    def checkpoint_dirs(self):
        """历史断点可能在哪：预训练目录、SFT 目录，以及 out 根下同级 run。

        返回 [{"kind": "pretrain"|"sft", "dir": 绝对路径}, ...]（去重、稳定顺序）。
        """
        seen, out = set(), []
        for kind, d in (("pretrain", self.out_dir()), ("sft", self.sft_out_dir())):
            if not d or d in seen:
                continue
            seen.add(d)
            out.append({"kind": kind, "dir": d})
        return out

    def all_checkpoints(self):
        """所有可续训/可用的断点（预训练 + SFT + out 根下的历史 run）。

        断点记录页面以前只看 `train.out_dir` 一个目录，于是 SFT 的
        `ckpt_sft.pt` 在列表里根本不存在（用户实测「没法从 SFT 断点续训」）。
        """
        seen, out = set(), []
        base = [x["dir"] for x in self.checkpoint_dirs()]
        dirs = list(self.checkpoint_dirs())
        # out 根目录下的历史 run 目录（chat_pretrain / chat_medium / out_sft …）
        roots = set()
        for d in base:
            roots.add(os.path.dirname(os.path.abspath(d)))
        for root in roots:
            try:
                names = sorted(os.listdir(root))
            except OSError:
                continue
            for name in names:
                d = os.path.join(root, name)
                if d in base or not os.path.isdir(d):
                    continue
                kind = "sft" if ("sft" in name.lower()) else "pretrain"
                dirs.append({"kind": kind, "dir": d})
        for item in dirs:
            for f in self.list_checkpoints(item["dir"]):
                if f["path"] in seen:
                    continue
                seen.add(f["path"])
                f["kind"] = item["kind"]
                f["dir"] = item["dir"]
                out.append(f)
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    def read_ckpt_meta(self, path):
        """读断点旁车 JSON（不 import torch）。老断点返回 {}。"""
        try:
            from lingyu.train import read_checkpoint_meta
            return read_checkpoint_meta(path)
        except Exception:
            return {}

    # ------------------------------------------------------------ 任务
    def begin_task(self, name):
        self.running_task = name
        self.task_started.emit(name)

    def end_task(self, name, code):
        self.running_task = None
        self.task_runner = None
        self.task_finished.emit(name, code)

    def attach_runner(self, name, runner):
        """把正在跑的任务的 Runner 接进全局总线（任务开始后立刻调用）。

        ## 为什么需要

        「运行状态」以前只活在某一个页面实例里：`_set_running(True)` 让
        停止按钮出现。但页面可能被重建（换语言会重建整个窗口，换主题/重开
        窗口也一样）—— 新页面还是默认态，「开始训练/续训」按钮回来了，
        停止按钮却没了，而后台任务其实还在跑（审计 1.7）。

        现在任务只认 `ctx.runner` / `ctx.running_task`：
        - 任务输出通过 ctx 的 `task_line/task_progress/task_validation` 广播，
          任何页面（当前窗口的、重建后的）都能收到并显示；
        - 停止按钮统一调 `ctx.stop_current_task()`，不依赖页面自己的 Runner。

        这里用 try/except 兜底：旧 Runner 若没有某个信号，也不能把主流程弄挂。
        """
        self.task_runner = runner
        try:
            runner.line.connect(
                lambda text, level, _n=name: self.task_line.emit(_n, text, level))
            runner.progress.connect(
                lambda info, _n=name: self.task_progress.emit(_n, info))
            runner.validation.connect(
                lambda step, loss, _n=name: self.task_validation.emit(_n, step, loss))
        except Exception:
            pass

    def stop_current_task(self):
        """停掉当前任务。没有任务/已结束返回 False（调用方回退到自己的 Runner）。"""
        runner = self.task_runner
        if runner is None:
            return False
        try:
            if not runner.is_running():
                return False
            runner.stop()
            return True
        except Exception:
            return False

    def is_busy(self):
        return bool(self.running_task)


# 全局单例
ctx = Context()
