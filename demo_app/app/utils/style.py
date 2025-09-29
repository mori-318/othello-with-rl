"""UIスタイル設定を集中管理するモジュール。"""


import tkinter as tk
from tkinter import ttk
from typing import Dict

RETRO_COLORS: Dict[str, str] = {
    "bg": "#000000",
    "panel": "#1a1a1a",
    "accent": "#00ff00",
    "accent_dark": "#00cc00",
    "text": "#ffffff",
    "subtext": "#cccccc",
    "alert": "#ff0000",
    "success": "#00ff00",
}

RETRO_FONT_FAMILY = "Press Start 2P"
FALLBACK_FONT_FAMILY = "Helvetica"
_CURRENT_FONT_FAMILY = RETRO_FONT_FAMILY

def load_fonts(root: tk.Misc) -> str:
    """レトロ風フォントを優先的に設定する。

    Args:
        root (tk.Misc): Tkinterルートウィンドウ。

    Returns:
        str: 利用可能なフォントファミリー名。
    """
    # Tkinterでフォントを明示的に登録
    global _CURRENT_FONT_FAMILY

    font_family = RETRO_FONT_FAMILY
    try:
        root.option_add("*Font", (font_family, 12))
    except tk.TclError:
        font_family = FALLBACK_FONT_FAMILY
        root.option_add("*Font", (font_family, 12))
    root.option_add("*TCombobox*Listbox.font", (font_family, 12))
    _CURRENT_FONT_FAMILY = font_family
    return font_family


def get_font(size: int, weight: str = "normal") -> tuple:
    """現在利用可能なフォント情報を取得する。

    Args:
        size (int): フォントサイズ。
        weight (str): フォントウエイト。

    Returns:
        tuple: Tkinterで使用可能なフォント指定。
    """
    if weight == "normal":
        return (_CURRENT_FONT_FAMILY, size)
    return (_CURRENT_FONT_FAMILY, size, weight)


def style_button(widget: ttk.Widget) -> None:
    """ボタンにレトロ風スタイルを適用する。

    Args:
        widget (ttk.Widget): スタイル対象のウィジェット。
    """
    widget.configure(style="RetroButton.TButton")


def style_label(widget: ttk.Widget, emphasis: bool = False) -> None:
    """ラベルにレトロ風スタイルを適用する。

    Args:
        widget (ttk.Widget): 対象ラベル。
        emphasis (bool): 強調表示するか。
    """
    style_name = "RetroLabelEmphasis.TLabel" if emphasis else "RetroLabel.TLabel"
    widget.configure(style=style_name)


def style_heading(widget: ttk.Widget) -> None:
    """見出し用スタイルを適用する。

    Args:
        widget (ttk.Widget): 見出しラベル。
    """
    widget.configure(style="RetroHeading.TLabel")


def style_panel(frame: ttk.Widget, *, inset: bool = False) -> None:
    """パネル用の背景スタイルを適用する。

    Args:
        frame (ttk.Widget): 対象フレーム。
        inset (bool): 内側パネルスタイルを適用するか。
    """
    style_name = "RetroSubPanel.TFrame" if inset else "RetroPanel.TFrame"
    frame.configure(style=style_name)


def apply_global_style(root: tk.Misc) -> None:
    """アプリ全体のスタイル設定を行う。

    Args:
        root (tk.Misc): Tkinterルートウィンドウ。
    """
    font_family = load_fonts(root)
    root.configure(background=RETRO_COLORS["bg"])

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("RetroRoot.TFrame", background=RETRO_COLORS["bg"], borderwidth=0)
    style.configure(
        "RetroPanel.TFrame",
        background=RETRO_COLORS["panel"],
        borderwidth=3,
        relief="ridge",
    )
    style.configure(
        "RetroSubPanel.TFrame",
        background=RETRO_COLORS["panel"],
        borderwidth=0,
        relief="flat",
    )

    style.configure(
        "RetroHeading.TLabel",
        background=RETRO_COLORS["bg"],
        foreground=RETRO_COLORS["accent"],
        font=(font_family, 22),
    )
    style.configure(
        "RetroLabel.TLabel",
        background=RETRO_COLORS["panel"],
        foreground=RETRO_COLORS["text"],
        font=(font_family, 14),
    )
    style.configure(
        "RetroLabelEmphasis.TLabel",
        background=RETRO_COLORS["panel"],
        foreground=RETRO_COLORS["accent"],
        font=(font_family, 16),
    )

    style.configure(
        "RetroButton.TButton",
        background=RETRO_COLORS["accent"],
        foreground="black",
        padding=8,
        font=(font_family, 11),
        borderwidth=2,
        relief="raised",
    )
    style.map(
        "RetroButton.TButton",
        background=[("active", RETRO_COLORS["accent_dark"])],
        relief=[("pressed", "sunken")],
    )

    style.configure(
        "RetroCombobox.TCombobox",
        fieldbackground=RETRO_COLORS["panel"],
        background=RETRO_COLORS["panel"],
        foreground=RETRO_COLORS["text"],
        arrowcolor=RETRO_COLORS["accent"],
        borderwidth=0,
        padding=8,
        font=(font_family, 12),
    )
    style.map(
        "RetroCombobox.TCombobox",
        fieldbackground=[("readonly", RETRO_COLORS["panel"])],
        selectbackground=[("readonly", RETRO_COLORS["panel"])],
        foreground=[("readonly", RETRO_COLORS["text"])],
    )
