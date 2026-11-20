# -*- coding: utf-8 -*-
"""灵语（LingYu）—— 一个引擎，从 13M 到万亿参数。

同一套配置语法、同一个 GUI、同一条代码路径；只有「执行后端」随规模切换。

    import lingyu

    # 读配置
    from lingyu.config import load_config, get, validate

    # 模型：按 [model] type 从注册表取实现（默认 lingyu）
    from lingyu.model import build_model, config_from_section
    model_cfg = config_from_section(cfg.get("model", {}), tok.vocab_size)
    model = build_model(model_cfg, device="cuda")

    # 命令行（等价于 `lingyu` 命令）
    from lingyu.cli import main
    main(["doctor"])

模块结构（v0.3 起是「平铺模块 + 子包并存」，没有 model/ 这类空目录）：
    src/lingyu/
      config/          参数规格、加载、校验（单一事实源）
      backends/        执行后端（local / fsdp / deepspeed / megatron 适配层）
      model.py         内置模型 LingYu + 模型协议 / 工厂（可插拔）
      registry.py      组件注册表（模型、注意力、FFN、归一化、位置编码…）
      components.py    组件分发（刻意不 import model，破循环导入）
      tokenizer.py     分词器（BPE / 字符级）
      prepare.py       语料扫描与 tokenize
      data.py          数据集与批采样
      train.py         预训练循环、断点、结构一致性检查
      sft.py           后训练 SFT（loss mask / 对话模板扩展）
      chat.py          命令行对话
      compression.py   断点无损压缩
      devices.py       设备探测（CUDA / MPS / CPU）
    gui/               PySide6 桌面界面（可选依赖）
    packaging/         启动器与打包脚本
    plugins/           第三方组件目录（放 .py 即生效）
    examples/          插件与自定义模型示例

导出 HuggingFace 格式（export/）仍在路线图中，尚未实现。
"""

from .version import __version__, CKPT_FORMAT, CONFIG_FORMAT, PROJECT_NAME

__all__ = ["__version__", "CKPT_FORMAT", "CONFIG_FORMAT", "PROJECT_NAME"]
