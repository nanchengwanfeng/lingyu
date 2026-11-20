# 贡献指南

欢迎参与灵语。仓库保持小而透明：一个 Python 包（`src/lingyu/`）、一个 PySide6 界面（`gui/`）
和一套 pytest 测试（`tests/`）。没有构建系统，也不需要生成器。

## 开发环境

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -e ".[dev,gui]"
# GPU 训练需要 CUDA 版 PyTorch（见 README 的说明）
```

GUI 依赖 PySide6（LGPL，可商用）。只做引擎开发可以不装 `gui` 扩展。

## 跑测试

```bash
pytest tests -q            # 全量（CPU 即可，无需 GPU）
pytest tests/test_gui.py -q
```

- GUI 测试用 offscreen 平台：`QT_QPA_PLATFORM=offscreen`。
- 训练相关的 slow 测试默认不跑；需要时 `pytest -m slow`。
- 提交前至少跑一遍 `pytest tests -q`，并在改动 GUI 后运行 `python gui/audit.py` 看截图。

## 代码约定

- Python 3.11+；`src/` 与 `gui/` 内模块导入用绝对导入。
- 配置项只在 `src/lingyu/config/spec.py` 定义一次；不要在其他地方重复校验。
- 界面颜色/间距/圆角只能用 `gui/theme.py` 的语义名，不要在控件里写死色值。
- 改断点格式必须同步升级 `version.CKPT_FORMAT` 并保留旧格式读取路径。

## 提交 PR

1. 从 `main` 开分支，一个 PR 只做一件事。
2. 说明动机、行为变化、验证方式（测试名称或截图）。
3. 不要提交 `out/`、`data/`、`dist/`、`venv/`、模型权重或语料；这些都已在 `.gitignore` 里。
4. 文档变更与代码变更放在同一个 PR 里。

## 新增模型 / 插件

见 [docs/EXTENDING.md](docs/EXTENDING.md)。最小流程：实现 `ModelProtocol`，
用 `@register("model", "名字")` 注册，把文件放进 `plugins/`，在配置里写 `[model] type = 名字`。

## 报告问题

请附：命令或操作路径、完整报错、`lingyu doctor --json` 输出、以及（如果与训练有关）
配置文件和 `out/<run>/train.log`。不要贴语料原文或模型权重。

## 许可证

贡献即表示同意以 [Apache-2.0](LICENSE) 授权。
