# 插件目录

把 `.py` 文件丢进这个目录，引擎启动时会**自动加载**。
不用改引擎的任何代码，就能替换或新增模型组件。

## 最小示例

在这个目录下新建 `my_attention.py`：

```python
from lingyu.registry import register


@register("attention", "my_attention", override=True)
class MyAttention:
    """自己的注意力实现。"""

    def __init__(self, cfg, layer_idx):
        # cfg 是模型配置字典，layer_idx 是第几层
        self.n_head = cfg["n_head"]
        self.head_dim = cfg["head_dim"]
        ...

    def forward(self, x, cache=None, cache_pos=0):
        # 输入 x: (batch, seq_len, d_model)
        # 必须返回同样形状的张量
        return x
```

然后在配置里点名用它：

```ini
[model]
attention = my_attention
```

## 可以替换的组件（`kind`）

| kind | 是什么 | 接口要求 |
|---|---|---|
| `attention` | 注意力 | `__init__(cfg, layer_idx)` + `forward(x, cache, cache_pos) -> x` |
| `ffn` | 前馈层 | `__init__(cfg)` + `forward(x) -> x` |
| `norm` | 归一化 | `__init__(dim, eps)` + `forward(x) -> x` |
| `pos` | 位置编码 | `forward(seq_len, device, dtype) -> (cos, sin)` |

## 三层优先级

同名时**后面的覆盖前面的**：

1. 内置实现（`lingyu/model/` 里注册的）
2. 插件目录（这个目录里的）
3. 配置文件点名（`model.attention = 谁`）

**覆盖内置实现必须写 `override=True`**。不写会报错 —— 这是故意的，
静默覆盖别人的实现是 bug 的温床。

## 出错了怎么办

插件加载失败**不会让引擎起不来**（一个好插件不该被坏插件连累），
但错误会被记下来。查看方式：

```bash
lingyu doctor      # 会列出插件加载错误
```

报错信息长这样：

```
没有注册名为 'my_attention' 的 attention 实现。
可用的有：mha、gqa、mqa、from_plugin
另外，以下插件加载失败（可能就是你要找的实现）：
  /path/plugins/my_attention.py -> SyntaxError: invalid syntax
插件目录：/path/plugins
```

## 换个插件目录

设置环境变量即可：

```bash
set LINGYU_PLUGINS=D:\my-plugins        # Windows
export LINGYU_PLUGINS=/home/me/plugins  # Linux/macOS
```

## 命名建议

插件注册名用**小写加下划线**。以 `_` 或 `.` 开头的文件会被跳过
（可以拿来放公共代码，比如 `_utils.py`）。
