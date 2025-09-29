"""Tkinterを用いたオセロ強化学習デモアプリのエントリーポイント。"""

import tkinter as tk
from tkinter import ttk

from app.ui.app_root import AppRoot
from app.utils.style import apply_global_style


def main() -> None:
    """アプリケーションを起動する。"""
    # ルートウィンドウを初期化
    root = tk.Tk()
    root.title("Othello RL Demo")
    root.geometry("960x720")
    root.minsize(800, 600)

    # ルートアプリケーションを生成
    apply_global_style(root)

    AppRoot(root)

    # メインループを開始
    root.mainloop()


if __name__ == "__main__":
    main()
