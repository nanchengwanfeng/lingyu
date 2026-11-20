# -*- coding: utf-8 -*-
r"""路径解析 —— **全项目唯一**回答「根目录在哪、配置去哪找」的地方。

## 为什么要有这个模块

同一个引擎有三种装法，路径规则必须一致。不一致的症状是
「命令行能跑、界面找不到语料」这类最难查的问题：

    开发 / pip install -e .     <项目>/src/lingyu/...        根 = <项目>
    打包版                      <runtime>/app/src/lingyu/    根 = <runtime>

打包版靠 `LINGYU_HOME` 区分 —— 启动器拉起子进程前把它设成 `runtime/`。

## 真实事故（这个模块就是为了修它）

打包版界面把默认配置存成了**相对路径** `configs/small.ini`，
再按「源码根」解析成 `<runtime>/app/configs/small.ini` —— 那是**引擎自带的那份**，
里面的 `data_dir = data1` 于是落到 `<runtime>/app/data1`。
而用户在「语料准备」页准备出来的语料在 `<runtime>/data`。

于是：界面刚说完「数据准备完成」，训练页就说「还没准备语料」。
两个目录都合法、都没有报错、都真实存在 —— 只是**不是同一个**。
"""

import os

#: 默认配置文件（相对名，会在各个候选根下依次找）
DEFAULT_CONFIG = "small.ini"


def package_root():
    """引擎源码所在的项目根（`src` 的上一级）。

    打包版会算到 `<runtime>/app` —— 那是**源码**的家，不是用户数据的家。
    要用户数据的位置请用 `home()`。
    """
    here = os.path.dirname(os.path.abspath(__file__))       # .../src/lingyu
    return os.path.dirname(os.path.dirname(here))           # .../


def home():
    """用户数据与用户配置的根目录。

    - 打包版：启动器设了 `LINGYU_HOME`，指向 `runtime/`
    - 开发版：没设，就等于项目根

    配置文件里的**相对路径**一律按这里解析，所以同一份配置
    在两种装法下都指向「引擎自己的目录」，不会跑到工作目录去。
    """
    env = os.environ.get("LINGYU_HOME")
    if env and os.path.isdir(env):
        return os.path.normpath(os.path.abspath(env))
    return package_root()


def _has_ini(d):
    try:
        return any(n.endswith(".ini") for n in os.listdir(d))
    except OSError:
        return False


def configs_dir():
    """用户配置目录。

    打包版是 `<runtime>/configs`（启动器把出厂配置复制过去并把路径钉成绝对的）；
    开发版没有这个目录，退回源码里的 `<项目>/configs`。
    """
    h = os.path.join(home(), "configs")
    if _has_ini(h):
        return h
    return os.path.join(package_root(), "configs")


def abspath(p):
    """把配置里的相对路径解析成绝对路径。

    ★ 基准是 `home()`，**不是当前工作目录**。
    用 cwd 的话，同一份配置在「双击 exe」和「命令行」下会指向不同的地方 ——
    这正是上面那起事故的成因之一。
    """
    if not p:
        return p
    p = os.path.expanduser(str(p))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(home(), p))


def find_config(name=None, extra=()):
    """找一个真实存在的配置文件，返回绝对路径。

    候选顺序（先找到先用）：

        1. `home()/名字`        —— 打包版 `runtime/configs/small.ini`
                                  开发版 `<项目>/configs/small.ini`
        2. `configs_dir()/文件名`
        3. `package_root()/名字`

    ★★ **刻意不把「当前工作目录」放进候选。**

    这一条踩过两次，所以写下来：打包版的工作目录是 `runtime/app`，
    于是 `abspath("configs/small.ini")` 会命中
    `runtime/app/configs/small.ini` —— 引擎自带的**源码副本**，
    里面是相对路径 `data_dir = data1`。
    它真实存在、能正常读出来、格式完全合法，只是**不是用户那份**。
    界面上没有任何异常，只有语料「凭空消失」。

    命令行传 `--config` 时不走这个函数（用户已经明确给了路径），
    所以去掉 cwd 候选不损失任何能力，只去掉了一个陷阱。

    一个都不存在时返回「用户配置目录/文件名」—— 那是新建配置该去的地方。
    """
    name = name or DEFAULT_CONFIG
    base = os.path.basename(name)
    if os.path.isabs(name):
        return os.path.normpath(name)
    cands = [os.path.join(home(), name),
             os.path.join(configs_dir(), base),
             os.path.join(package_root(), name)]
    for c in extra:
        cands.append(c)
    for c in cands:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return os.path.normpath(cands[1])


def describe():
    """诊断用：把三套路径一次列清楚。`lingyu doctor` 和界面都会用。"""
    return {"home": home(), "package_root": package_root(),
            "configs_dir": configs_dir()}
