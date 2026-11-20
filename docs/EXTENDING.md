# 扩展灵语

这份文档回答两个问题：

1. **怎么换模型架构**（保留训练 / 微调 / 采样 / GUI 全套能力）
2. **怎么加自定义组件**（注意力、FFN、归一化、位置编码、优化器）

引擎对外只依赖一份**最小接口**。只要满足它，其余代码一行都不用改。

---

## 1. 换模型架构（约 15 分钟）

### 1.1 接口长什么样

见 `src/lingyu/model.py` 的 `ModelProtocol`：

| 方法 | 作用 | 必须 |
|---|---|---|
| `forward(idx, targets=None, cache=None, cache_pos=0, collect_aux=True)` | 前向；给了 `targets` 时返回 `(logits, loss, aux)` | ✅ |
| `generate(idx, max_new_tokens, ...)` | 自回归续写，返回含 prompt 的完整序列 | ✅ |
| `structural_fingerprint()` | 返回「换了就必须重训」的结构字段（续训一致性检查） | ✅ |

`forward` 的约定（训练循环真实依赖这几条）：

- `logits` 形状 `(B, T, vocab_size)`
- `targets` 里 **-1 表示这个位置不算损失**（SFT 的 loss mask 靠它，`ignore_index=-1`）
- `targets is None` 时**只需要返回最后一个位置的 logits**（生成时省算力），`loss`、`aux` 给 `None`
- `collect_aux=False` 时允许跳过 MoE 统计
- MoE 模型在 `aux` 里返回 `{"aux_loss", "z_loss", "loads"}`；非 MoE 返回 `None`

`generate` 至少接受 `max_new_tokens`；建议同时支持 `temperature / top_k / top_p /
eos_id / use_cache`，GUI 与 `lingyu sample` 会传这些参数。

### 1.2 最小实现

完整可运行版本见 `examples/models/tiny_lm.py`（下面是为了讲解压缩过的片段）：

```python
# 节选：examples/models/tiny_lm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from lingyu.registry import register


class TinyLM(nn.Module):
    """示例：一个没有注意力的"词袋"语言模型。

    只用来演示接口，不追求效果 —— 它连语序都看不见。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d = int(self.cfg["d_model"])
        self.tok = nn.Embedding(self.cfg["vocab_size"], d)
        self.head = nn.Linear(d, self.cfg["vocab_size"], bias=False)

    def forward(self, idx, targets=None, cache=None, cache_pos=0,
                collect_aux=True):
        x = self.tok(idx).mean(dim=1, keepdim=True)          # (B, 1, d)
        logits = self.head(x)                                 # (B, 1, V)
        if targets is None:
            return logits, None, None
        # 广播到每个位置，只为满足契约；真实模型应逐位置预测
        logits = logits.expand(-1, targets.shape[1], -1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-1)
        return logits, loss, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, **kwargs):
        out = idx
        for _ in range(int(max_new_tokens)):
            logits, _, _ = self(out)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out

    def structural_fingerprint(self):
        return {"vocab_size": self.cfg.get("vocab_size"),
                "n_layer": self.cfg.get("n_layer")}


register("model", "tiny")(TinyLM)
```

### 1.3 接进配置

```ini
[model]
type = tiny            ; 注册表里的名字；不写默认 lingyu
n_layer = 1            ; 用不到的结构字段也建议写上，断点检查要用
n_embd = 64
n_head = 4
block_size = 256
vocab_size = 0         ; 0 = 跟随分词器
```

把示例复制进插件目录（CLI 每个进程启动时自动加载：`plugins/` 是约定目录）：

```bash
# Windows
copy examples\models\tiny_lm.py plugins\tiny_lm.py
# macOS / Linux
cp examples/models/tiny_lm.py plugins/tiny_lm.py
```

之后所有命令照常：

```bash
lingyu train --config examples/configs/example_tiny.ini
lingyu plan  --config examples/configs/example_tiny.ini
lingyu gui
```

`build_model()` 会按 `[model] type` 从注册表取实现；找不到时会把**所有可用名字**列出来。

### 1.4 一定要守住的几条

1. **`structural_fingerprint()` 必须诚实**。它返回的字段变了，引擎就会拒绝续训 ——
   这是保护，不是麻烦。把真正影响权重形状/语义的字段都放进去。
2. **断点里会保存 `type`**。所以自定义模型的断点，换回默认模型不能直接续训（引擎会拦）。
3. **不要改 `vocab_size` 的语义**：模型拿到的 `cfg["vocab_size"]` 永远等于分词器词表。
4. **GUI 的参数表单自动生成**：`[model]` 段的字段来自 `src/lingyu/config/spec.py`；
   自定义模型如果要用新参数，需要往 spec 里加（或直接用 `--set` 传）。

---

## 2. 加自定义组件

组件种类（`src/lingyu/registry.py` 的 `KINDS`）：

```
model / attention / ffn / norm / pos / optimizer / scheduler / backend
```

### 2.1 写一个组件

```python
from lingyu.registry import register

@register("ffn", "gelu_mlp")
class GeluMLP:
    def __init__(self, cfg):
        ...
    def forward(self, x):
        ...
```

### 2.2 让配置能选到它

配置里的名字就是注册名：

```ini
[model]
ffn = gelu_mlp
```

组件名也会写进断点的结构指纹，换组件后旧断点会被拒绝续训。

### 2.3 三种安装方式

| 方式 | 做法 | 适合 |
|---|---|---|
| **插件目录** | 把 `.py` 丢进项目根的 `plugins/`（或 `LINGYU_PLUGINS` 指向的目录） | 临时试验、私有组件 |
| **examples/ 示例** | 把 `examples/models/tiny_lm.py` 复制进 `plugins/` 即可运行 | 教学、参考实现 |
| **改源码** | 在 `model.py` 末尾 `register(...)` | 要进主干的组件 |

插件在**启动时自动扫描**，语法错误不会炸掉引擎，只会在 `lingyu doctor` /
`lingyu plugins` 里列为「加载失败」并给出原因。

---

## 3. 自检清单

```bash
# 1. 结构对不对：跑契约测试（会自动覆盖注册表里所有模型）
pytest tests/test_model_contract.py -v

# 2. 配置能不能解析、显存估算是否合理
lingyu plan --config examples/configs/example_tiny.ini

# 3. 真跑两步（CPU 也能跑，验证前向 / 反向 / 断点）
lingyu train --config examples/configs/example_tiny.ini --device cpu --set train.max_steps=2
```

---

## 4. 目前**不**在契约内的东西

诚实说明，避免你按错误预期设计插件：

- **分词器**：仍是固定实现（BPE / 字符级），不可插拔（路线图）
- **数据加载 / 损失函数 / 指标**：注册表已有种类占位，但训练循环还没走注册表（路线图）
- **训练循环本身**：固定 `train.py`；自定义优化器需实现与 `torch.optim.Optimizer`
  兼容的接口（`configure_optimizer` 已支持注册表名）
- **分布式后端**：`fsdp/deepspeed/megatron` 目前是「适配层 + 脚本生成」，
  不是完整的分布式训练实现

以上每一项都会在 `docs/PROJECT_STATE.md` 与 `docs/MATURITY.md` 里标注状态。

---

## 5. 写一个注意力内核（示例：局部窗口注意力）

注意力是模型里最值得替换的部分之一。引擎把 SDPA 调用收敛到 **一个方法**：
`src/lingyu/model.py` 的 `Attention._attend(q, k_all, v_all, T, offset)`。
覆写它就能换内核，而 KV cache、RoPE、int8 缓存、分块预填充都不需要重写。

可运行的完整示例（约 40 行核心代码）：

```text
examples/plugins/attention_local_window.py   # 局部窗口注意力
examples/configs/local_window.ini            # 用它的训练配置
```

步骤：

```bash
# 1. 复制示例到引擎的 plugins/ 目录（丢 .py 即自动加载）
copy examples\plugins\attention_local_window.py plugins\

# 2. 配置里写 attention = local_window
lingyu train --config examples\configs\local_window.ini --set data.data_dir=data1

# 3. 可选：窗口大小（不设默认 512；0 = 全注意力）
set LINGYU_ATTN_WINDOW=1024
```

要点与坑：

- `_attend` 的 `offset` 是 query 的绝对位置起点（分块预填 / KV cache 时 > 0），
  写掩码时必须带上它，否则位置会整体错位。
- 示例用环境变量传窗口，是为了**不改引擎的参数契约**；真实产品把参数加进
  `config/spec.py` 的 `[model]` 段即可。
- 训练和推理必须用**同一个窗口**；换窗口 = 换了掩码规则，旧断点不能直接续训。
- 局部窗口省的是长上下文注意力计算/显存，代价是长距离依赖，别指望两者兼得。
- 测试参考：`tests/test_plugin_attention.py`（注册、前向/反向、cache 解码、
  窗口确实生效、窗口 0 与全注意力一致）。
