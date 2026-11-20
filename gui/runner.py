# -*- coding: utf-8 -*-
r"""子进程管理：**GUI 绝不 import torch**。

## 这条规矩为什么不能破

PySide6 界面本身启动只要零点几秒，但 `import torch` 在 8GB 的机器上
要 3~8 秒（要加载几个 GB 的 DLL 和 CUDA 运行库）。

如果把 torch import 进 GUI 进程：

- 打开界面要等好几秒，用户以为程序卡死了
- **常驻显存**：CUDA 上下文一建就是几百 MB，本来就只有 6.9GB 可用，
      白白占掉一块
- 训练崩了会把整个界面一起带走

所以：**界面进程里只有 PySide6**，所有要 torch 的活（训练、SFT、采样、聊天）
都通过 `lingyu` 命令行在子进程里跑。界面只负责显示子进程的输出。

## 三件事要做好

1. **流式读取**：子进程的输出要一行行实时送出来，不能等它跑完。
   用 QProcess 而不是 subprocess，因为它原生接 Qt 事件循环，不会阻塞界面。

2. **进度解析**：训练的输出行长得像
   `  第  100/10000 步  loss 4.1234  lr 6.00e-04  |g| 1.23  0:00:12`
   这里把它解析成结构化数据（step/loss/lr/gnorm），曲线和数字面板才有得画。

3. **能停得住**：点「停止」要真的停掉子进程**以及它的子进程**。
   Windows 上 `terminate()` 只杀直接子进程，训练里的 DataLoader worker
   会变成孤儿进程继续吃内存。所以这里先发 Ctrl+C 式的优雅中断
   （`lingyu` 自己会捕获 KeyboardInterrupt 并安全存档），
   实在不行再强杀整棵进程树。
"""

import json
import os
import re
import sys
import tempfile

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

# 训练进度行的格式（train.py 里打出来的）
_RE_STEP = re.compile(
    r"第\s*([\d,]+)\s*/\s*([\d,]+)\s*步.*?loss\s+([\d.]+).*?lr\s+([\d.eE+-]+)"
    r"(?:.*?\|g\|\s*([\d.]+))?")
# 验证行
_RE_VAL = re.compile(r"\[验证\]\s*第\s*([\d,]+)\s*步.*?loss\s+([\d.]+)")
# SFT 的进度行： 第   10/2000 步  loss 2.8479  lr 5.87e-04  |g| 4.06  0:00:12
_RE_SFT = _RE_STEP
# 「[存档] 第 N 步 → 路径」
_RE_CKPT = re.compile(r"\[存档\]\s*第\s*([\d,]+)\s*步\s*→\s*(.+)$")
# 通用：以 [错误] / [注意] / [通过] 开头的行
_RE_LEVEL = re.compile(r"^\s*\[(错误|注意|通过|提示|完成)\]\s*(.*)$")


def child_environment():
    """给 `python -m lingyu` 子进程用的完整环境。

    ★ 别拿 `QProcess().processEnvironment()` 当起点：新建 QProcess 的这个
    值是**空的**，往里 insert 几个变量（UTF-8 等）就等于把父进程环境
    整个替换掉 —— PATH / PYTHONPATH / LINGYU_HOME 全丢。

    实测踩过（2026-10-18）：打包版聊天页点「加载模型」报
    `No module named lingyu`，而同一命令在终端里跑得好好的 ——
    差的正是启动器设的 PYTHONPATH。

    以 systemEnvironment()（= 当前进程环境，含启动器设的变量）为底，
    再显式覆盖 os.environ，保证子进程和 GUI 看到的环境一致。
    """
    env = QProcessEnvironment.systemEnvironment()
    for key, value in os.environ.items():
        env.insert(key, value)
    return env


def _to_int(s):
    try:
        return int(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


class Runner(QObject):
    """跑一个 `lingyu` 子进程，把输出翻译成信号。

    信号：
        line(str, str)          一行原始输出 + 级别（error/warn/ok/info/title/dim）
        progress(dict)          解析出的进度 {step, total, loss, lr, gnorm}
        validation(int, float)  验证结果
        checkpoint(int, str)    存了断点
        finished(int, bool)     退出码 + 是否被用户停掉
    """

    line = Signal(str, str)
    progress = Signal(dict)
    validation = Signal(int, float)
    checkpoint = Signal(int, str)
    finished = Signal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None
        self._stopping = False
        self._buf = ""
        self._last = {}
        # 兜底强杀定时器：正常退出/重新启动时必须撤销，
        # 否则上一轮的 4 秒定时器会误杀下一轮任务（审计 G-01）。
        self._stopping_proc = None
        # 优雅停止协议用的标记文件路径（由 start() 创建，stop() 写入）
        self._stop_file = None
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)

    # ------------------------------------------------------------ 状态
    def is_running(self):
        return self.proc is not None and \
            self.proc.state() != QProcess.NotRunning

    @property
    def last_progress(self):
        return dict(self._last)

    # ------------------------------------------------------------ 启动
    def start(self, args, cwd=None, env_extra=None):
        """跑 `python -m lingyu <args>`。

        用 `-m lingyu` 而不是 `lingyu.exe`：这样即使没装命令行入口、
        或者装在了别的环境里，只要项目在就能跑起来。
        """
        if self.is_running():
            self.line.emit("已经有一个任务在跑了", "warn")
            return False

        self._stopping = False
        self._buf = ""
        self._last = {}
        self._kill_timer.stop()      # 撤销上一轮残留的兜底强杀
        self._stopping_proc = None

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(cwd or self._project_root())

        # ★ 统一走 child_environment()：别用空的 processEnvironment() 当底，
        #   否则一 insert 就等于替换掉整个父环境（打包版会丢 PYTHONPATH）。
        env = child_environment()
        # 强制 UTF-8：不加这两个，Windows 上子进程的中文输出会按 GBK 编出来，
        # 界面上就是一屏乱码（而且报错信息也看不懂）
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONUNBUFFERED", "1")     # 否则输出会攒一大块才吐出来
        for k, v in (env_extra or {}).items():
            env.insert(k, v)
        # ★ 优雅停止协议（与引擎侧约定）：给子进程一个"停止标记文件"路径。
        #   文件在启动时**不存在**——引擎轮询它是否出现；用户点停止时
        #   runner 才写入，引擎看到后保存断点并退出（见 stop()）。
        self._cleanup_stop_file()
        fd, stop_path = tempfile.mkstemp(prefix="lingyu_stop_", suffix=".flag")
        os.close(fd)
        try:
            os.unlink(stop_path)   # 只保留唯一路径，启动时不能让它存在
        except OSError:
            pass
        self._stop_file = stop_path
        env.insert("LINGYU_STOP_FILE", stop_path)
        proc.setProcessEnvironment(env)

        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        self.proc = proc
        cmd = [sys.executable, "-u", "-m", "lingyu"] + list(args)
        self.line.emit("$ " + " ".join(cmd[1:]), "dim")
        proc.start(cmd[0], cmd[1:])
        return True

    def _project_root(self):
        """子进程的工作目录 = **用户数据根**（打包版是 `runtime/`）。

        ★ 和 `lingyu.paths.home()` 保持一致。这里以前按 `__file__` 往上数，
          打包版算出来是 `runtime/app`，于是配置里的相对路径
          （`data_dir = data`）在子进程里解析到 `runtime/app/data` ——
          和界面显示的、以及语料准备写下的目录都不是同一个。
        """
        from lingyu import paths
        return paths.home()

    # ------------------------------------------------------------ 输出
    def _on_output(self):
        if self.proc is None:
            return
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buf += raw
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, text):
        if not text.strip():
            return
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                if self._handle_event(stripped):
                    return
            except (ValueError, KeyError, TypeError):
                pass
        level = "info"
        m = _RE_LEVEL.match(text)
        if m:
            kind, rest = m.group(1), m.group(2)
            level = {"错误": "error", "注意": "warn", "通过": "ok",
                     "提示": "info", "完成": "ok"}.get(kind, "info")
            text = rest or text
        elif text.lstrip().startswith("=="):
            level = "dim"
        elif text.strip().startswith("["):
            level = "title"

        # ---- 解析进度 ----
        pm = _RE_STEP.search(text)
        if pm:
            step = _to_int(pm.group(1))
            total = _to_int(pm.group(2))
            try:
                loss = float(pm.group(3))
            except ValueError:
                loss = None
            info = {"step": step, "total": total, "loss": loss,
                    "lr": pm.group(4), "gnorm": pm.group(5)}
            self._last.update(info)
            self.progress.emit(info)
            # 进度行本身也显示，但压成一行简短形式，免得日志被刷屏
            if step and total:
                pct = 100.0 * step / max(1, total)
                text = "第 %s/%s 步（%.1f%%）  loss %s  lr %s%s" % (
                    "{:,}".format(step), "{:,}".format(total), pct,
                    ("%.4f" % loss) if loss is not None else "—",
                    pm.group(4),
                    ("  |g| %s" % pm.group(5)) if pm.group(5) else "")
            level = "info"

        vm = _RE_VAL.search(text)
        if vm:
            s = _to_int(vm.group(1))
            try:
                v = float(vm.group(2))
            except ValueError:
                v = None
            if s is not None and v is not None:
                self.validation.emit(s, v)
                self._last["val_loss"] = v

        cm = _RE_CKPT.search(text)
        if cm:
            s = _to_int(cm.group(1))
            if s is not None:
                self.checkpoint.emit(s, cm.group(2).strip())

        self.line.emit(text, level)

    # ------------------------------------------------------ JSON 事件（1.4）
    def _handle_event(self, text):
        """把引擎 `--json` 打的一行事件翻译成信号 + 一行人类可读日志。

        返回 True 表示这行确实是结构化事件；False 表示只是恰好以 { 开头的
        普通日志，交给原来的正则解析。
        """
        obj = json.loads(text)
        if not isinstance(obj, dict) or "event" not in obj:
            return False
        kind = str(obj.get("event"))
        task = {"train": "pretrain", "sft": "SFT"}.get(obj.get("task"), "")
        label = ("[%s] " % task) if task else ""
        if kind == "start":
            bits = [label + "开始"]
            if obj.get("max_steps"):
                bits.append("总步数 %s" % "{:,}".format(int(obj["max_steps"])))
            if obj.get("batch_size"):
                bits.append("batch %s" % obj["batch_size"])
            if obj.get("block_size"):
                bits.append("block %s" % obj["block_size"])
            if obj.get("start_step"):
                bits.append("从第 %s 步续训" % "{:,}".format(int(obj["start_step"])))
            self.line.emit("，".join(bits), "title")
        elif kind == "step":
            info = {"step": obj.get("step"), "total": obj.get("total"),
                    "loss": obj.get("loss"), "lr": obj.get("lr"),
                    "gnorm": obj.get("grad_norm"), "tok_s": obj.get("tok_s"),
                    "elapsed": obj.get("elapsed")}
            self._last.update({k: v for k, v in info.items() if v is not None})
            self.progress.emit(info)
            step, total = obj.get("step"), obj.get("total")
            if step and total:
                pct = 100.0 * step / max(1, total)
                speed = obj.get("tok_s")
                self.line.emit(
                    "%s第 %s/%s 步（%.1f%%）  loss %s  lr %s%s" % (
                        label, "{:,}".format(int(step)), "{:,}".format(int(total)),
                        pct,
                        ("%.4f" % obj["loss"]) if obj.get("loss") is not None else "—",
                        obj.get("lr", "—"),
                        ("  %.0f tok/s" % speed) if speed else ""), "info")
        elif kind == "eval":
            s = int(obj.get("step") or 0)
            v = float(obj.get("val_loss") or 0.0)
            self.validation.emit(s, v)
            self._last["val_loss"] = v
            self.line.emit("%s第 %s 步  验证 loss %.4f" %
                           (label, "{:,}".format(s), v), "info")
        elif kind == "ckpt":
            s = int(obj.get("step") or 0)
            path = str(obj.get("path") or "")
            self.checkpoint.emit(s, path)
            self.line.emit("%s已存档：第 %s 步 → %s%s" %
                           (label, "{:,}".format(s), path,
                            "（目前最好）" if obj.get("best") else ""), "ok")
        elif kind == "sample":
            self.line.emit("%s试写：%s" % (label, obj.get("text", "")), "dim")
        elif kind == "probe":
            self.line.emit("%s问：%s" % (label, obj.get("question", "")), "info")
            self.line.emit("%s答：%s" % (label, obj.get("answer", "")), "dim")
        elif kind == "done":
            steps = obj.get("steps")
            bits = [label + "完成"]
            if steps is not None:
                bits.append("共 %s 步" % "{:,}".format(int(steps)))
            if obj.get("best_val") is not None:
                bits.append("最好验证 loss %.4f" % float(obj["best_val"]))
            if obj.get("stopped"):
                bits.append("已按停止请求收尾")
            if obj.get("interrupted"):
                bits.append("被中断")
            self.line.emit("，".join(bits), "ok")
        elif kind == "error":
            self.line.emit("%s[错误] %s" % (label, obj.get("message", "")), "error")
        else:
            return False
        return True

    def _on_error(self, err):
        if err == QProcess.FailedToStart:
            self.line.emit("启动失败 —— 检查 Python 环境是否完整", "error")

    def _on_finished(self, code, _status):
        if self._buf.strip():
            self._handle_line(self._buf.strip())
            self._buf = ""
        stopped = self._stopping
        self._stopping = False
        # 正常退出：撤销兜底强杀，并清掉"目标进程"标记——
        # 这台定时器绝不能再影响下一个任务。
        self._kill_timer.stop()
        self._stopping_proc = None
        proc, self.proc = self.proc, None
        if proc is not None:
            proc.deleteLater()
        # 把"用户主动停止"告诉上层（gui/app.py 据此不把停止报成失败）
        self._cleanup_stop_file()
        try:
            from .context import ctx
            ctx.last_finished_stopped = bool(stopped)
        except Exception:
            pass
        self.finished.emit(int(code), stopped)

    # ------------------------------------------------------------ 停止
    def _cleanup_stop_file(self):
        """删掉停止标记文件（任务开始/结束都要清，避免残留误停下一轮）。"""
        path, self._stop_file = self._stop_file, None
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass

    def stop(self, grace_ms=20000):
        """优雅停止：写停止标记文件，等引擎保存断点后自己退出。

        协议（与引擎侧 core-train 约定）：
            环境变量 LINGYU_STOP_FILE = 本 runner 创建的唯一临时路径；
            启动时该文件不存在；用户点停止 → runner 写入该文件 →
            引擎轮询到文件后保存断点并退出。

        宽限期（默认 20 秒）内没退出才强杀，并如实说明可能丢进度。
        `clicked` 信号传进来的 False / 0 按"没传"处理。
        """
        if not self.is_running():
            return
        try:
            grace_ms = int(grace_ms or 20000)
        except (TypeError, ValueError):
            grace_ms = 20000
        self._stopping = True
        proc = self.proc
        self._stopping_proc = proc
        wrote = False
        if self._stop_file:
            try:
                with open(self._stop_file, "w", encoding="utf-8") as fp:
                    fp.write("stop\n")
                wrote = True
            except OSError:
                wrote = False
        if wrote:
            self.line.emit("已请求停止：正在保存断点并退出…", "warn")
        else:
            # 没有标记文件（例如子进程不是通过 start() 起的）——退回直接终止
            self.line.emit("已请求停止：没有停止标记文件，直接结束进程…", "warn")
            try:
                proc.terminate()
            except Exception:
                pass
        self._kill_timer.start(grace_ms)

    def _force_kill(self):
        proc = self._stopping_proc
        if (proc is not None and self._stopping
                and self.proc is proc and self.is_running()):
            self.line.emit(
                "子进程没有响应停止请求，已强制结束"
                "（最后一段进度可能丢失）", "warn")
            try:
                proc.kill()
            except Exception:
                pass
        self._stopping_proc = None
        self._cleanup_stop_file()


class Poller(QObject):
    """定时轮询一个文件，内容变了就发信号。

    训练曲线用这个：`train.py` 会持续往 `history.csv` 追加
    `step,loss,val_loss,lr,gnorm,elapsed`，界面每隔一秒读一次就够了。
    比解析 stdout 可靠 —— 日志格式改了曲线也不会断。
    """

    changed = Signal(list)      # 每行解析成 list[str]

    def __init__(self, interval_ms=1000, parent=None):
        super().__init__(parent)
        self.path = None
        self._mtime = None
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    def watch(self, path):
        self.path = path
        self._mtime = None
        self._timer.start()
        self._tick()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            mt = os.path.getmtime(self.path)
        except OSError:
            return
        if mt == self._mtime:
            return
        self._mtime = mt
        rows = []
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        rows.append(line.split(","))
        except OSError:
            return
        self.changed.emit(rows)
