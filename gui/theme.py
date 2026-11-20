# -*- coding: utf-8 -*-
"""界面主题：配色、字体、间距、以及那套 Qt 样式表。

## 审美约定（改界面前先读这十条）

1. **一屏只有一个主色**：紫蓝（accent）。别的地方只用中性灰做层次。
2. **层次靠明度，不靠阴影**：Qt 画阴影慢，深色下也糊；用背景 + 1px 边框分层。
3. **圆角只四档**：6 / 8 / 10 / 12（小控件 / 输入 / 卡片 / 大容器）。
4. **间距只用 4 的倍数**：4 / 8 / 12 / 16 / 24 / 32。
5. **文字三级**：正文 text、次要 text_dim、说明 text_muted —— 不要第四级。
6. **标题要轻**：页面标题 23pt/700，卡片标题 12pt/600；不要满屏粗体。
7. **数字用等宽**：所有指标、耗时、loss 都是 mono，宽度自然对齐。
8. **颜色只从这张表取**：页面代码里出现 `#RRGGBB` 一律算 bug。
9. **两套主题共用语义色名**：切主题时页面代码一行都不改。
10. **动效只做两处**：切页淡入、悬停变色。别再加弹跳/旋转。

页面代码统一 `from .theme import th` 然后 `th.SURFACE` 这样用；
`tests/test_theme_has_all_semantic_colors` 保证两套主题的键完全一致。
"""

# ==========================================================================
# 间距与圆角：**只用这几档**，不要在页面里写别的值
# ==========================================================================
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32}
RADIUS = {"sm": 6, "md": 8, "lg": 12, "input": 8, "pill": 999}

# 字号（pt）。中文在 10pt 以下会糊，所以最小 9 只给极小的角标用。
FONT_SIZE = {"tiny": 9, "small": 10, "body": 11, "label": 12,
             "title": 14, "h2": 18, "h1": 23}

# 中文字体优先，其次西文。Qt 会按顺序回退，所以中英混排都不会出现方框。
FONT_UI = '"Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", ' \
          '"Segoe UI", "Noto Sans CJK SC", sans-serif'
# 日志、曲线、数字用等宽字体，对齐才好看
FONT_MONO = '"Cascadia Mono", "Consolas", "JetBrains Mono", ' \
            '"DejaVu Sans Mono", monospace'


# ==========================================================================
# 深色主题（默认）：近黑底 + 紫蓝主色，参考 Linear / Raycast 那类工具界面
# ==========================================================================
DARK = {
    # 背景层次：从最底到最亮，一共四层。
    "bg":          "#08090C",   # 窗口底色（比纯黑暖一点点）
    "sidebar":     "#0B0C11",   # 侧边栏
    "surface":     "#12141B",   # 卡片
    "surface_2":   "#191D27",   # 输入框 / 次级按钮
    "surface_hi":  "#151821",   # 卡片渐变顶部
    "surface_lo":  "#0D0F15",   # 卡片渐变底部 / 内嵌区
    "hover":       "#1E2230",
    "active":      "#272C3D",

    "border":      "#232838",   # 常规描边
    "border_soft": "#181C28",   # 极淡分隔线
    "border_focus": "#8B7CFF",

    "text":        "#ECEFF5",
    "text_dim":    "#A5ACBC",
    "text_muted":  "#6B7284",
    "text_on_accent": "#FFFFFF",

    "accent":      "#8B7CFF",   # 主色：紫
    "accent_hover": "#A99BFF",
    "accent_2":    "#5B8CFF",   # 渐变另一端：蓝
    "accent_soft": "#1D1A33",   # 主色极淡底（选中项）
    "success":     "#3DDC97",
    "warning":     "#F5B74E",
    "danger":      "#FF7A7A",
    "info":        "#5BC0EB",
    "success_soft": "#12261E",
    "warning_soft": "#2A2113",
    "danger_soft": "#2C171B",
    "info_soft":   "#10222B",

    # 曲线配色
    "chart_grid":  "#1C2130",
    "chart_axis":  "#333B4E",
    "chart_line":  "#8B7CFF",
    "chart_val":   "#3DDC97",
    "chart_fill_a": "#2B2350",
    "chart_fill_b": "#12141B",
}

# ==========================================================================
# 浅色主题
# ==========================================================================
LIGHT = {
    "bg":          "#F5F6FA",
    "sidebar":     "#FFFFFF",
    "surface":     "#FFFFFF",
    "surface_2":   "#F2F4F9",
    "surface_hi":  "#FFFFFF",
    "surface_lo":  "#FAFBFE",
    "hover":       "#EEF1F8",
    "active":      "#E3E8F3",

    "border":      "#E0E5EF",
    "border_soft": "#EEF1F7",
    "border_focus": "#6C5CE7",

    "text":        "#161922",
    "text_dim":    "#4C556B",
    "text_muted":  "#8A93A6",
    "text_on_accent": "#FFFFFF",

    "accent":      "#6C5CE7",
    "accent_hover": "#5A4AD6",
    "accent_2":    "#3B6FE0",
    "accent_soft": "#EFECFF",
    "success":     "#0E9F6E",
    "warning":     "#B7791F",
    "danger":      "#DC2626",
    "info":        "#0284C7",
    "success_soft": "#E8F7F1",
    "warning_soft": "#FBF3E3",
    "danger_soft": "#FCECEC",
    "info_soft":   "#E6F4FA",

    "chart_grid":  "#ECF0F6",
    "chart_axis":  "#C9D0DC",
    "chart_line":  "#6C5CE7",
    "chart_val":   "#0E9F6E",
    "chart_fill_a": "#D8D1FA",
    "chart_fill_b": "#FFFFFF",
}

THEMES = {"dark": DARK, "light": LIGHT}


class Theme:
    """当前主题。页面代码统一 `from .theme import th` 然后用 `th.SURFACE`。"""

    def __init__(self, name="dark"):
        self.set(name)

    def set(self, name):
        self.name = name if name in THEMES else "dark"
        self._c = THEMES[self.name]
        for k, v in self._c.items():
            setattr(self, k.upper(), v)

    def __getitem__(self, key):
        return self._c[key]

    @property
    def is_dark(self):
        return self.name == "dark"


# 全局单例。切主题时改它的属性，然后重新 apply 样式表。
th = Theme("dark")


# ==========================================================================
# 样式表
# ==========================================================================
def stylesheet(t=None):
    """生成整份 QSS。切换主题时重新调用并 setStyleSheet 即可。"""
    t = t or th
    return _QSS.format(
        bg=t.BG, sidebar=t.SIDEBAR, surface=t.SURFACE, surface_2=t.SURFACE_2,
        surface_hi=t.SURFACE_HI, surface_lo=t.SURFACE_LO,
        hover=t.HOVER, active=t.ACTIVE,
        border=t.BORDER, border_soft=t.BORDER_SOFT, border_focus=t.BORDER_FOCUS,
        text=t.TEXT, text_dim=t.TEXT_DIM, text_muted=t.TEXT_MUTED,
        on_accent=t.TEXT_ON_ACCENT,
        accent=t.ACCENT, accent_hover=t.ACCENT_HOVER, accent_2=t.ACCENT_2,
        accent_soft=t.ACCENT_SOFT,
        success=t.SUCCESS, warning=t.WARNING, danger=t.DANGER, info=t.INFO,
        success_soft=t.SUCCESS_SOFT, warning_soft=t.WARNING_SOFT,
        danger_soft=t.DANGER_SOFT, info_soft=t.INFO_SOFT,
        font_ui=FONT_UI, font_mono=FONT_MONO,
        r_input=RADIUS["input"], r_md=RADIUS["md"], r_lg=RADIUS["lg"],
        r_sm=RADIUS["sm"], r_pill=RADIUS["pill"],
        fs_tiny=FONT_SIZE["tiny"], fs_small=FONT_SIZE["small"],
        fs_body=FONT_SIZE["body"], fs_label=FONT_SIZE["label"],
        fs_title=FONT_SIZE["title"], fs_h2=FONT_SIZE["h2"],
        fs_h1=FONT_SIZE["h1"],
    )


# 说明：下面这份 QSS 里**不出现任何具体色值**（渐变端点也用命名色），
# 全部由上面的配色表填进来。加一套主题只要再写一张表。
_QSS = """
* {{
    font-family: {font_ui};
    font-size: {fs_body}pt;
    outline: none;
}}

QWidget {{
    background: {bg};
    color: {text};
}}

QMainWindow {{
    background: {bg};
}}

QToolTip {{
    background: {surface_2};
    color: {text};
    border: 1px solid {border};
    border-left: 2px solid {accent};
    border-radius: {r_sm}px;
    padding: 7px 10px;
    font-size: {fs_small}pt;
}}

/* 参数说明的「?」徽标：灰色小圆，悬停才出完整说明（审计 1.5）。 */
#HelpBadge {{
    color: {text_muted};
    background: {surface_2};
    border: 1px solid {border_soft};
    border-radius: 8px;
    font-size: {fs_tiny}pt;
    font-weight: 700;
}}
#HelpBadge:hover {{
    color: {on_accent};
    background: {accent};
    border: 1px solid {accent};
}}

/* ---------------------------------------------------------------- 顶栏 */
#TopBar {{
    background: {sidebar};
    border-bottom: 1px solid {border_soft};
}}
#TopCrumb {{
    color: {text_dim};
    font-size: {fs_body}pt;
    font-weight: 500;
    letter-spacing: 0.3px;
}}
#TopTask {{
    color: {text_muted};
    font-size: {fs_small}pt;
}}
QPushButton#TopIcon {{
    background: transparent;
    color: {text_dim};
    border: 1px solid {border_soft};
    border-radius: {r_input}px;
    padding: 3px 9px;
    font-size: {fs_body}pt;
}}
QPushButton#TopIcon:hover {{
    background: {hover};
    color: {text};
    border-color: {border};
}}

/* ---------------------------------------------------------------- 侧边栏 */
#Sidebar {{
    background: {sidebar};
    border-right: 1px solid {border_soft};
}}
#BrandMark {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 {accent_2}, stop:1 {accent});
    border-radius: 6px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
}}
#Brand {{
    color: {text};
    font-size: {fs_h2}pt;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0px;
}}
#BrandSub {{
    color: {text_muted};
    font-size: {fs_tiny}pt;
    padding: 0px;
}}
#BrandVersion {{
    color: {text_muted};
    font-size: {fs_tiny}pt;
    padding: 0px;
}}
#NavSection {{
    color: {text_muted};
    font-size: {fs_tiny}pt;
    font-weight: 700;
    padding: 16px 18px 6px 12px;
    letter-spacing: 1.4px;
}}
QPushButton#NavItem {{
    background: transparent;
    color: {text_dim};
    border: none;
    border-left: 2px solid transparent;
    border-radius: {r_md}px;
    margin: 1px 8px;
    padding: 8px 12px;
    text-align: left;
    font-size: {fs_label}pt;
}}
QPushButton#NavItem:hover {{
    background: {hover};
    color: {text};
}}
QPushButton#NavItem:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {accent_soft}, stop:1 transparent);
    color: {accent};
    border-left: 2px solid {accent};
    font-weight: 600;
}}
#SidebarStatus {{
    background: {surface_lo};
    border: 1px solid {border_soft};
    border-radius: {r_md}px;
    color: {text_muted};
    font-size: {fs_tiny}pt;
}}

/* ------------------------------------------------------------ 卡片 / 分区 */
#Card {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 {surface_hi}, stop:1 {surface_lo});
    border: 1px solid {border};
    border-radius: {r_lg}px;
}}
#CardFlat {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 {surface_hi}, stop:1 {surface_lo});
    border: 1px solid {border_soft};
    border-radius: {r_md}px;
}}
#CardTitle {{
    color: {text};
    font-size: {fs_label}pt;
    font-weight: 600;
    letter-spacing: 0.3px;
}}
#CardHint {{
    color: {text_muted};
    font-size: {fs_small}pt;
}}
#PageTitle {{
    color: {text};
    font-size: {fs_h1}pt;
    font-weight: 700;
    letter-spacing: 0.3px;
}}
#PageDesc {{
    color: {text_muted};
    font-size: {fs_small}pt;
}}
#SectionLabel {{
    color: {text_muted};
    font-size: {fs_tiny}pt;
    font-weight: 700;
    letter-spacing: 1.2px;
}}
#Divider {{
    background: {border_soft};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}
#Muted {{
    color: {text_muted};
    font-size: {fs_small}pt;
}}

/* ---------------------------------------------------------------- 标签页 */
QTabWidget::pane {{
    border: none;
    background: transparent;
}}
QTabBar {{
    background: transparent;
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: transparent;
    color: {text_muted};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 14px;
    margin-right: 2px;
    font-size: {fs_body}pt;
}}
QTabBar::tab:hover {{
    color: {text};
}}
QTabBar::tab:selected {{
    color: {accent};
    border-bottom: 2px solid {accent};
    font-weight: 600;
}}
QTabBar::tab:disabled {{
    color: {border};
}}

/* ---------------------------------------------------------------- 按钮 */
QPushButton {{
    background: {surface_2};
    color: {text};
    border: 1px solid {border};
    border-radius: {r_input}px;
    padding: 7px 14px;
    font-size: {fs_body}pt;
}}
QPushButton:hover {{
    background: {hover};
    border-color: {border_focus};
}}
QPushButton:pressed {{
    background: {active};
}}
QPushButton:disabled {{
    color: {text_muted};
    background: {surface};
    border-color: {border_soft};
}}
QPushButton#Primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {accent}, stop:1 {accent_2});
    color: {on_accent};
    border: 1px solid transparent;
    border-radius: {r_input}px;
    padding: 8px 20px;
    font-weight: 600;
    letter-spacing: 0.2px;
}}
QPushButton#Primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {accent_hover}, stop:1 {accent_2});
}}
QPushButton#Primary:pressed {{
    background: {accent};
}}
QPushButton#Primary:disabled {{
    background: {surface_2};
    color: {text_muted};
    border-color: {border_soft};
}}
QPushButton#Danger {{
    background: transparent;
    color: {danger};
    border: 1px solid {border};
    border-radius: {r_input}px;
}}
QPushButton#Danger:hover {{
    background: {danger_soft};
    border-color: {danger};
}}
QPushButton#Ghost {{
    background: transparent;
    border: 1px solid transparent;
    color: {text_dim};
    border-radius: {r_input}px;
}}
QPushButton#Ghost:hover {{
    background: {hover};
    color: {text};
}}
QPushButton#Chip {{
    background: transparent;
    color: {text_dim};
    border: 1px solid {border};
    border-radius: {r_pill}px;
    padding: 4px 12px;
    font-size: {fs_small}pt;
}}
QPushButton#Chip:checked {{
    background: {accent_soft};
    color: {accent};
    border-color: {accent};
}}
QPushButton#Chip:hover {{
    color: {text};
    border-color: {border_focus};
}}

/* ---------------------------------------------------------------- 输入 */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {surface_lo};
    color: {text};
    border: 1px solid {border};
    border-radius: {r_input}px;
    padding: 7px 10px;
    selection-background-color: {accent};
    selection-color: {on_accent};
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {active};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent};
    background: {surface};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {text_muted};
    background: {surface};
}}
QLineEdit[echoMode="2"] {{
    lineedit-password-character: 9679;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 16px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {text_muted};
    width: 0px; height: 0px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {text_muted};
    width: 0px; height: 0px;
}}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
    border-bottom-color: {text};
}}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
    border-top-color: {text};
}}

/* ---------------------------------------------------------------- 下拉框 */
QComboBox {{
    background: {surface_lo};
    color: {text};
    border: 1px solid {border};
    border-radius: {r_input}px;
    padding: 7px 10px;
    min-width: 70px;
}}
QComboBox:hover {{ border-color: {active}; }}
QComboBox:focus {{ border-color: {accent}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
    border: none;
    background: transparent;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {text_muted};
    width: 0px; height: 0px;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {surface_2};
    color: {text};
    border: 1px solid {border};
    border-radius: {r_input}px;
    padding: 4px;
    outline: none;
    selection-background-color: {accent_soft};
    selection-color: {accent};
}}
QComboBox QAbstractItemView::item {{
    padding: 6px 8px;
    border-radius: {r_sm}px;
    min-height: 18px;
}}
QComboBox QAbstractItemView::item:hover {{
    background: {hover};
    color: {text};
}}

/* ---------------------------------------------------------------- 勾选 */
QCheckBox, QRadioButton {{
    color: {text};
    spacing: 8px;
    padding: 3px 0px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {border};
    background: {surface_lo};
}}
QCheckBox::indicator {{ border-radius: 5px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {accent};
}}
QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
}}
QRadioButton::indicator:checked {{
    background: {accent};
    border: 4px solid {surface_lo};
    outline: 1px solid {accent};
}}

/* ---------------------------------------------------------------- 进度条 */
QProgressBar {{
    background: {surface_2};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {accent}, stop:1 {accent_2});
    border-radius: 4px;
}}

/* ---------------------------------------------------------------- 滚动条 */
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {active};
    border-radius: 5px;
    min-height: 28px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {text_muted}; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background: {active};
    border-radius: 5px;
    min-width: 28px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: {text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---------------------------------------------------------------- 列表/表格 */
QListWidget, QTreeWidget, QTableWidget {{
    background: transparent;
    color: {text};
    border: 1px solid {border};
    border-radius: {r_input}px;
    outline: none;
    padding: 4px;
}}
QListWidget::item {{
    padding: 8px 10px;
    border-radius: {r_sm}px;
    color: {text_dim};
}}
QListWidget::item:hover {{ background: {hover}; color: {text}; }}
QListWidget::item:selected {{
    background: {accent_soft};
    color: {accent};
}}
QHeaderView::section {{
    background: transparent;
    color: {text_muted};
    border: none;
    border-bottom: 1px solid {border};
    padding: 7px 8px;
    font-size: {fs_tiny}pt;
    font-weight: 700;
    letter-spacing: 0.6px;
}}
QTableWidget {{
    gridline-color: transparent;
    alternate-background-color: {surface_lo};
    selection-background-color: {accent_soft};
    selection-color: {accent};
}}
QTableWidget::item {{ padding: 6px 8px; border: none; }}
QTableWidget::item:selected {{ background: {accent_soft}; color: {accent}; }}
QTableCornerButton::section {{ background: transparent; border: none; }}

/* ---------------------------------------------------------------- 其它 */
#LogView {{
    background: {surface_lo};
    color: {text_dim};
    border: 1px solid {border_soft};
    border-radius: {r_input}px;
    font-family: {font_mono};
    font-size: {fs_small}pt;
    padding: 8px;
}}
#ChatArea {{
    background: {surface_lo};
    border: 1px solid {border_soft};
    border-radius: {r_lg}px;
}}
#ChatInput {{
    background: {surface_lo};
    color: {text};
    border: 1px solid {border};
    border-radius: {r_input}px;
    padding: 10px 14px;
    font-size: {fs_label}pt;
}}
#ChatInput:focus {{
    border-color: {accent};
    background: {surface};
}}
#StatusBar {{
    background: {sidebar};
    border-top: 1px solid {border_soft};
    color: {text_muted};
    font-size: {fs_small}pt;
}}
QLabel#StatValue {{
    color: {text};
    font-family: {font_mono};
    font-size: {fs_h2}pt;
    font-weight: 700;
    letter-spacing: 0.2px;
}}
QLabel#StatLabel {{
    color: {text_muted};
    font-size: {fs_tiny}pt;
    letter-spacing: 0.4px;
}}
QFrame#StatTile {{
    background: {surface_lo};
    border: 1px solid {border_soft};
    border-radius: {r_md}px;
}}
QTextBrowser#HelpView {{
    background: transparent;
    border: 1px solid {border_soft};
    border-radius: {r_md}px;
    padding: 12px 16px;
}}
QListWidget#HelpList {{
    background: transparent;
    border: 1px solid {border_soft};
    border-radius: {r_md}px;
}}
QSplitter::handle {{ background: {border_soft}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

/* 帮助页正文：Markdown 渲染出来的标题/引用/代码块 */
QTextBrowser#HelpView h1 {{ font-size: {fs_h2}pt; }}
QTextBrowser#HelpView h2 {{ font-size: {fs_title}pt; }}
QTextBrowser#HelpView h3 {{ font-size: {fs_label}pt; }}
QTextBrowser#HelpView code {{
    font-family: {font_mono};
    background: {surface_2};
}}
QTextBrowser#HelpView pre {{
    background: {surface_lo};
    border: 1px solid {border_soft};
    border-radius: {r_sm}px;
    padding: 8px;
}}
QTextBrowser#HelpView blockquote {{
    border-left: 2px solid {accent};
    color: {text_dim};
    margin-left: 2px;
    padding-left: 10px;
}}
"""
