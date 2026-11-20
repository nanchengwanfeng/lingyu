# -*- coding: utf-8 -*-
"""断点权重的**无损**压缩（逐字节可逆，不做任何量化）。

思路（参考 ZipNN 论文 2411.05239 的 bit-plane / 字节平面思路）：
  1. 把 float32 张量按**字节平面**重排：同一权重的高字节放一起、低字节放一起；
  2. 浮点数的符号位 + 指数位集中在同一平面里，重复度高、极易压缩；
     尾数平面接近随机，压缩率有限 —— 所以整体收益是有上限的（实测 1.15~1.25x）。
  3. 用 zlib 分别压缩每个平面，存为 dict（dtype/shape/data），读时逆变换。

它**不改变任何数值**：读回来与存之前逐 bit 相等（有测试钉住）。
代价是存/读各多花几秒 CPU（在 CPU 上做，不占显存）。

可选环境变量：
  LINGYU_PACK_CKPT=0     关闭压缩（存旧格式）
  LINGYU_PACK_LEVEL=6    zlib/lzma 级别（1 快 / 9 小，默认 6）
  LINGYU_PACK_METHOD=zlib 压缩算法：zlib（默认）或 lzma（更小、更慢）
  LINGYU_PACK_THREADS=8  并行压缩线程数（默认 min(8, CPU 数)）
"""
import lzma
import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import torch


def _level():
    try:
        return max(1, min(9, int(os.environ.get("LINGYU_PACK_LEVEL", "6"))))
    except ValueError:
        return 6


def pack_enabled():
    return os.environ.get("LINGYU_PACK_CKPT", "1") not in ("0", "false", "False")


def _method():
    m = os.environ.get("LINGYU_PACK_METHOD", "zlib").strip().lower()
    return "lzma" if m == "lzma" else "zlib"


def _threads():
    try:
        n = int(os.environ.get("LINGYU_PACK_THREADS", "0"))
    except ValueError:
        n = 0
    if n > 0:
        return max(1, min(32, n))
    return max(1, min(8, os.cpu_count() or 1))


def _compress_bytes(data):
    """返回 (codec, blob)。lzma 压缩率高但慢，只由用户显式开启。"""
    if _method() == "lzma":
        return "lzma", lzma.compress(data, preset=_level())
    return "zlib", zlib.compress(data, _level())


def _decompress_bytes(blob, codec):
    if codec == "lzma":
        return lzma.decompress(blob)
    return zlib.decompress(blob)


def _plane_bytes(t):
    """张量 -> 按字节平面重排后的 bytes（不复制数值，只换字节顺序）。"""
    import numpy as np
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)  # bitcast：字节完全不变，只为绕开 numpy 不支持 bf16
    arr = t.detach().contiguous().cpu().numpy()
    raw = arr.tobytes()
    itemsize = arr.dtype.itemsize
    if itemsize == 1 or not raw:
        return raw
    u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, itemsize)
    # 每个字节位置的所有权重连续放一起
    return u8.T.copy().tobytes()


def _unplane_bytes(data, shape, dtype):
    import numpy as np
    torch_dtype = _TORCH_DTYPE[dtype]
    np_dtype = _NP_DTYPE[dtype]
    itemsize = np.dtype(np_dtype).itemsize
    n = 1
    for s in shape:
        n *= int(s)
    total = n * itemsize
    if total == 0:
        return torch.empty(shape, dtype=torch_dtype)
    if itemsize == 1:
        raw = data[:total]
    else:
        u8 = np.frombuffer(data[:total], dtype=np.uint8)
        raw = u8.reshape(itemsize, -1).T.copy().tobytes()
    return torch.frombuffer(bytearray(raw), dtype=torch_dtype).reshape(shape).clone()


# 只处理训练/推理里真实出现的 dtype；其余原样走 torch.save（安全兜底）
_TORCH_DTYPE = {
    "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.int16,
    "float64": torch.float64, "int64": torch.int64, "int32": torch.int32,
    "int16": torch.int16, "int8": torch.int8, "uint8": torch.uint8,
    "bool": torch.bool,
}
_NP_DTYPE = {
    "float32": "float32", "float16": "float16", "bfloat16": "int16",
    "float64": "float64", "int64": "int64", "int32": "int32",
    "int16": "int16", "int8": "int8", "uint8": "uint8", "bool": "bool",
}


def pack_tensor(t):
    """张量 -> 可被 torch.save 安全序列化的 dict（无损）。"""
    if not isinstance(t, torch.Tensor):
        return t
    dtype = str(t.dtype).replace("torch.", "")
    if dtype not in _TORCH_DTYPE:
        return t
    shape = list(t.shape)
    planes = _plane_bytes(t)
    codec, blob = _compress_bytes(planes)
    return {
        "__ly_pack__": 2,
        "dtype": dtype,
        "shape": shape,
        "nbytes": len(planes),
        "codec": codec,
        "z": blob,
        "fmt": _format_marker(t),
    }


def unpack_tensor(d):
    dtype = d["dtype"]
    shape = d["shape"]
    planes = _decompress_bytes(d["z"], d.get("codec", "zlib"))
    t = _unplane_bytes(planes, shape, dtype)
    if d.get("fmt") == "bfloat16":
        # bfloat16 用 int16 平面搬运（bitcast），再反推回 bfloat16
        t = t.view(torch.bfloat16)
    return t


def _format_marker(t):
    return "bfloat16" if t.dtype == torch.bfloat16 else ""


def _collect_tensors(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_tensors(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_tensors(v, out)


def _rebuild_tensors(obj, packed, idx):
    if isinstance(obj, torch.Tensor):
        val = packed[idx[0]]
        idx[0] += 1
        return val
    if isinstance(obj, dict):
        return {k: _rebuild_tensors(v, packed, idx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rebuild_tensors(v, packed, idx) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_rebuild_tensors(v, packed, idx) for v in obj)
    return obj


def pack_payload(obj):
    """递归把 payload 里所有张量换成压缩 dict；zlib/lzma 会释放 GIL，多线程加速。"""
    tensors = []
    _collect_tensors(obj, tensors)
    threads = _threads()
    if threads > 1 and len(tensors) >= 8:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            packed = list(ex.map(pack_tensor, tensors))
    else:
        packed = [pack_tensor(t) for t in tensors]
    return _rebuild_tensors(obj, packed, [0])


def _collect_packed(obj, out):
    if isinstance(obj, dict):
        if obj.get("__ly_pack__") in (1, 2) and "z" in obj:
            out.append(obj)
            return
        for v in obj.values():
            _collect_packed(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_packed(v, out)


def _rebuild_unpacked(obj, unpacked, idx):
    if isinstance(obj, dict):
        if obj.get("__ly_pack__") in (1, 2) and "z" in obj:
            val = unpacked[idx[0]]
            idx[0] += 1
            return val
        return {k: _rebuild_unpacked(v, unpacked, idx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rebuild_unpacked(v, unpacked, idx) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_rebuild_unpacked(v, unpacked, idx) for v in obj)
    return obj


def unpack_payload(obj):
    packed = []
    _collect_packed(obj, packed)
    threads = _threads()
    if threads > 1 and len(packed) >= 8:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            unpacked = list(ex.map(unpack_tensor, packed))
    else:
        unpacked = [unpack_tensor(d) for d in packed]
    return _rebuild_unpacked(obj, unpacked, [0])


def wrap(payload):
    """包成 torch.save 用的 dict：有压缩标记，读时一眼能认出来。"""
    if not pack_enabled():
        return payload
    try:
        return {"__compression__": "plane-zlib", "payload": pack_payload(payload)}
    except Exception:
        # 压缩失败绝不能拦住训练：退回未压缩，照常保存
        return payload


def unwrap(payload):
    if isinstance(payload, dict) and payload.get("__compression__") == "plane-zlib":
        return unpack_payload(payload["payload"])
    return payload


# ------------------------------------------------------------------ 文件容器
# torch.save 的 pickle 协议 2 会把 bytes 膨胀 1.5x（实测 26MB -> 39MB），
# 而压缩后的权重全是 bytes。所以新格式用「魔数 + pickle 协议 4」直接写盘：
#   b"LYCKPT1" + pickle(payload)
# 旧格式（torch.save 的 zip）仍然由 load_checkpoint 读取。
MAGIC = b"LYCKPT1"


def looks_packed_file(path):
    try:
        with open(path, "rb") as fp:
            return fp.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def _dump_packed(path, payload):
    import pickle
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as fp:
            fp.write(MAGIC)
            pickle.dump(payload, fp, protocol=4)
        os.replace(tmp, path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def _load_packed(path):
    import pickle
    with open(path, "rb") as fp:
        magic = fp.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("不是灵语压缩断点")
        return pickle.load(fp)


def save_checkpoint_file(path, payload):
    """有压缩标记时写自有容器（省 1.5x bytes 膨胀），否则退回 torch.save。"""
    wrapped = wrap(payload)
    if isinstance(wrapped, dict) and wrapped.get("__compression__"):
        _dump_packed(path, wrapped)
        return True
    import torch
    torch.save(wrapped, path)
    return False


def load_checkpoint_file(path, map_location="cpu"):
    """先认自有容器，再退回 torch.load（老断点兼容）。"""
    if looks_packed_file(path):
        return unwrap(_load_packed(path))
    import torch
    try:
        ck = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=map_location)
    return unwrap(ck)
