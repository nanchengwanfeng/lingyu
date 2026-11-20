# -*- coding: utf-8 -*-
"""引擎目录瘦身：把「打包版 runtime\venv」里与开发 venv 完全相同的文件
换成硬链接，从而不重复占用磁盘；另可删掉离线 torch 轮子。

为什么安全：
  - 只处理 **大小和 SHA-256 都一致** 的文件，内容不变，功能/性能零损失。
  - Windows 同盘符才允许硬链接；D: 盘满足（runtime 和 venv 都在 D:）。
  - 一个环境里改/升级包时，pip 会删旧文件再写新文件：删掉的是那条目录项，
    另一个环境仍指向原 inode，**不会被破坏**；两边从此各自独立，直到再次运行本脚本。

用法：
    python packaging/slim.py --status        # 只看能省多少
    python packaging/slim.py --apply         # 执行（可重复运行，幂等）
    python packaging/slim.py --apply --delete-wheels
"""
import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_VENV = os.path.join(ROOT, "venv", "Lib", "site-packages")
RT_VENV = os.path.join(ROOT, "dist", "LingYu", "runtime", "venv",
                       "Lib", "site-packages")
WHEELS = os.path.join(ROOT, "wheels")


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iter_files(base):
    for dirpath, _dirs, files in os.walk(base):
        for name in files:
            p = os.path.join(dirpath, name)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if not os.path.isfile(p):
                continue
            yield p, st.st_size


def plan(verbose=False):
    """返回 (可链文件列表, 已链大小, 待链大小)。"""
    dev_by_rel = {}
    for p, size in _iter_files(DEV_VENV):
        rel = os.path.relpath(p, DEV_VENV)
        dev_by_rel[rel] = (p, size)
    candidates = []
    linked = 0
    for p, size in _iter_files(RT_VENV):
        rel = os.path.relpath(p, RT_VENV)
        dev = dev_by_rel.get(rel)
        if not dev:
            continue
        devp, devsize = dev
        if devsize != size:
            continue
        try:
            if os.path.samefile(devp, p):  # 已经是硬链接
                linked += size
                continue
        except OSError:
            pass
        candidates.append((p, devp, size, rel))
    return candidates, linked


def apply(candidates):
    saved = 0
    failed = []
    for rt, dev, size, rel in candidates:
        try:
            # 只对内容完全一致的才链：先对比哈希（大文件按 8MB 分块读）
            if _sha256(rt) != _sha256(dev):
                failed.append((rel, "内容不同"))
                continue
            tmp = rt + ".linking"
            if os.path.exists(tmp):
                os.remove(tmp)
            os.link(dev, tmp)
            os.replace(tmp, rt)
            saved += size
        except OSError as exc:
            failed.append((rel, str(exc)))
    return saved, failed


def _dir_size(base):
    total = 0
    for _p, size in _iter_files(base):
        total += size
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delete-wheels", action="store_true")
    args = ap.parse_args()
    if not os.path.isdir(RT_VENV):
        print("没有 dist\\LingYu\\runtime\\venv，先跑 packaging/build_exe.py")
        return 1
    before = _dir_size(RT_VENV)
    candidates, linked = plan()
    print("runtime venv : %.2f GB" % (before / 1e9))
    print("已是硬链接   : %.2f GB" % (linked / 1e9))
    print("可去重文件   : %d 个，%.2f GB" % (len(candidates),
                                            sum(c[2] for c in candidates) / 1e9))
    if not args.apply:
        print("\n（只统计。执行：python packaging/slim.py --apply）")
    else:
        saved, failed = apply(candidates)
        print("\n已硬链接 %d 个，实际可省 %.2f GB"
              % (len(candidates) - len(failed), saved / 1e9))
        if failed:
            print("跳过 %d 个：" % len(failed))
            for rel, why in failed[:8]:
                print("  %-60s %s" % (rel[:58], why))
        after = _dir_size(RT_VENV)
        print("runtime venv 现在（逻辑大小）: %.2f GB（物理占用见磁盘属性）"
              % (after / 1e9))
    if args.delete_wheels and os.path.isdir(WHEELS):
        total = 0
        for dirpath, _d, files in os.walk(WHEELS):
            for name in files:
                p = os.path.join(dirpath, name)
                total += os.path.getsize(p)
                os.remove(p)
        for dirpath, dirs, _f in os.walk(WHEELS, topdown=False):
            for d in dirs:
                try:
                    os.rmdir(os.path.join(dirpath, d))
                except OSError:
                    pass
        print("已删除离线轮子目录：%.2f GB（重装/修复改为在线下载）" % (total / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
