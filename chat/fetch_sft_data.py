# -*- coding: utf-8 -*-
"""下载 Alpaca-GPT4 中文指令数据集，并转成灵语 SFT 能直接吃的 JSONL。

## 为什么需要这个脚本

`configs/chat.ini` 的 SFT 数据默认指向 `external_data/alpaca_gpt4_zh.jsonl`，
而仓库里**不带**这个文件（原始 CSV 30MB，放进源码包不合适）。所以第一次
想跑聊天流程时，先执行一次本脚本：

    python chat/fetch_sft_data.py

它会下载 → 过滤 → 打乱 → 写成 JSONL。产物约 32MB，放在 `external_data/`
（不打包进发布物，删了重跑即可）。

## 数据来源与许可证

- 来源：ModelScope `AI-ModelScope/alpaca-gpt4-data-zh`
- 许可证：**CC BY-NC 4.0（非商用）**。自己玩、做研究没问题；要商用请换
  数据集，或去数据页确认最新许可条款。
- 原始数据是 GPT-4 生成的指令数据（约 4.9 万条），这里按总长度 ≤ 700
  字符过滤，以适配 `configs/chat.ini` 的 max_seq_len=512。

## 参数

    --out-dir    产物目录（默认 external_data，相对引擎根目录）
    --max-chars  单条样本最大字符数（默认 700；把 max_seq_len 调大后可以放开）
    --seed       打乱种子（默认 1234，固定它保证每次生成同一份数据）
    --keep-csv   保留原始 train.csv（默认删掉，省 30MB 磁盘）
    --limit      最多保留多少条（默认 0 = 全保留）
"""

import argparse
import csv
import json
import os
import random
import sys
import urllib.request

URL = ("https://modelscope.cn/api/v1/datasets/AI-ModelScope/"
       "alpaca-gpt4-data-zh/repo?Revision=master&FilePath=train.csv")
LICENSE_NOTE = "CC BY-NC 4.0（非商用；来源 ModelScope AI-ModelScope/alpaca-gpt4-data-zh）"


def download(url, dst):
    """带进度地下载。不用 requests —— 引擎的运行时依赖只有 numpy。"""
    print("下载 %s" % url)
    req = urllib.request.Request(url, headers={"User-Agent": "lingyu-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, \
            open(dst, "wb") as fp:
        total = int(resp.headers.get("Content-Length") or 0)
        got, last = 0, -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fp.write(chunk)
            got += len(chunk)
            if total:
                pct = got * 100 // total
                if pct >= last + 10:
                    last = pct
                    print("  %3d%%  %5.1f / %5.1f MB" % (pct, got / 1e6, total / 1e6))
    print("  -> %s（%.2f MB）" % (dst, os.path.getsize(dst) / 1e6))


def convert(csv_path, jsonl_path, max_chars, seed, limit):
    rows, seen = [], 0
    with open(csv_path, encoding="utf-8", newline="") as fp:
        rd = csv.DictReader(fp)
        for r in rd:
            seen += 1
            ins = (r.get("instruction") or "").strip()
            inp = (r.get("input") or "").strip()
            out = (r.get("output") or "").strip()
            if not ins or not out:
                continue
            if len(ins) + len(inp) + len(out) > max_chars:
                continue
            # 把 input 拼到 instruction 后面：plain 模板只有「用户 / 助手」两段，
            # 没有单独的输入槽位。
            prompt = ins + ("\n" + inp if inp else "")
            rows.append({"instruction": prompt, "output": out})
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    with open(jsonl_path, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("原始 %d 条 → 保留 %d 条（%d 条空样本或超过 %d 字符被过滤）"
          % (seen, len(rows), seen - len(rows), max_chars))
    print("-> %s（%.2f MB）" % (jsonl_path, os.path.getsize(jsonl_path) / 1e6))
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="下载 Alpaca-GPT4 中文指令数据并转成灵语 SFT JSONL")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=os.path.join(root, "external_data"))
    ap.add_argument("--max-chars", type=int, default=700)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-csv", action="store_true")
    ap.add_argument("--skip-download", action="store_true",
                    help="跳过下载，直接用 out-dir 里已有的 train.csv 转换")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "alpaca_gpt4_zh.csv")
    jsonl_path = os.path.join(args.out_dir, "alpaca_gpt4_zh.jsonl")

    if os.path.isfile(jsonl_path):
        print("已存在：%s" % jsonl_path)
        print("如需重新生成，先删掉它（或换 --out-dir）。")
        return 0

    try:
        if args.skip_download:
            if not os.path.isfile(csv_path):
                print("[错误] --skip-download 但找不到 %s" % csv_path,
                      file=sys.stderr)
                return 1
            print("跳过下载，使用已有：%s" % csv_path)
        else:
            download(URL, csv_path)
    except Exception as e:                      # noqa: BLE001 - 给用户人话
        print("\n[错误] 下载失败：%s" % e, file=sys.stderr)
        print("  1) 检查网络 / 代理（ModelScope 国内一般可直连）", file=sys.stderr)
        print("  2) 也可以手动下载后重跑：", file=sys.stderr)
        print("     %s" % URL, file=sys.stderr)
        print("     存成 %s，再加 --skip-download（或用 --keep-csv 跑）" % csv_path,
              file=sys.stderr)
        return 1

    n = convert(csv_path, jsonl_path, args.max_chars, args.seed, args.limit)
    if not args.keep_csv:
        os.remove(csv_path)
    print()
    print("完成：%d 条样本 → %s" % (n, jsonl_path))
    print("许可证：%s" % LICENSE_NOTE)
    print()
    print("下一步：")
    print("  1) lingyu train --config configs/chat.ini   # 预训练")
    print("  2) lingyu sft   --config configs/chat.ini   # SFT（会读上面这个 jsonl）")
    print("  3) lingyu chat  --config configs/chat.ini   # 对话")
    return 0


if __name__ == "__main__":
    sys.exit(main())
