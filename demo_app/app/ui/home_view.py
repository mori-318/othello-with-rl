"""ホーム画面のTkinterビュー。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List

from app.utils.style import (
    RETRO_COLORS,
    style_button,
    style_heading,
    style_label,
    style_panel,
)

AGENT_OPTIONS: List[str] = ["random", "dqn", "mcts_nn"]


class HomeView(tk.Frame):
    """ホーム画面を表示するTkinterフレーム。"""

    def __init__(
        self,
        master: tk.Misc,
        on_start_agent_vs_human: Callable[[str], None],
        on_start_agent_vs_agent: Callable[[str, str], None],
    ) -> None:
        """ホーム画面を初期化する。

        Args:
            master (tk.Misc): 親ウィジェット。
            on_start_agent_vs_human (Callable[[str], None]): 人間対エージェント開始時のコールバック。
            on_start_agent_vs_agent (Callable[[str, str], None]): エージェント同士対戦開始時のコールバック。
        """
        super().__init__(master, background=RETRO_COLORS["bg"])

        # コールバックを保持
        self._on_start_agent_vs_human = on_start_agent_vs_human
        self._on_start_agent_vs_agent = on_start_agent_vs_agent

        # UI構築
        self._build_view()

    def _build_view(self) -> None:
        """ビューを構築する。"""
        # ルートレイアウトの設定
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # タイトルラベル
        title_label = ttk.Label(self, text="OTHELLO RL DEMO")
        title_label.grid(row=0, column=0, pady=(20, 15))
        style_heading(title_label)

        # コンテンツフレーム
        content = ttk.Frame(self)
        content.grid(row=1, column=0, sticky=tk.NSEW)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        # 人間 vs エージェントセクション
        human_frame = ttk.Frame(content, padding=12)
        human_frame.grid(row=0, column=0, padx=15, pady=8, sticky=tk.NSEW)
        human_frame.columnconfigure(0, weight=1)
        style_panel(human_frame)

        human_title = ttk.Label(human_frame, text="PLAYER VS AGENT")
        human_title.grid(row=0, column=0, sticky=tk.W)
        style_label(human_title, emphasis=True)

        human_label = ttk.Label(human_frame, text="エージェントを選択")
        human_label.grid(row=1, column=0, pady=(8, 4), sticky=tk.W)
        style_label(human_label)

        self._human_agent_var = tk.StringVar(value=AGENT_OPTIONS[0])
        human_option = ttk.Combobox(
            human_frame,
            textvariable=self._human_agent_var,
            values=AGENT_OPTIONS,
            state="readonly",
        )
        human_option.grid(row=2, column=0, pady=4, sticky=tk.EW)

        human_button = ttk.Button(
            human_frame,
            text="START",
            command=self._handle_start_agent_vs_human,
        )
        human_button.grid(row=3, column=0, pady=(10, 0), sticky=tk.E)
        style_button(human_button)

        # エージェント vs エージェントセクション
        agent_frame = ttk.Frame(content, padding=12)
        agent_frame.grid(row=0, column=1, padx=15, pady=8, sticky=tk.NSEW)
        agent_frame.columnconfigure(0, weight=1)
        agent_frame.columnconfigure(1, weight=1)
        style_panel(agent_frame)

        agent_title = ttk.Label(agent_frame, text="AGENT VS AGENT")
        agent_title.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        style_label(agent_title, emphasis=True)

        black_label = ttk.Label(agent_frame, text="黒番エージェント")
        black_label.grid(row=1, column=0, pady=(8, 4), sticky=tk.W)
        style_label(black_label)

        self._black_agent_var = tk.StringVar(value=AGENT_OPTIONS[0])
        black_option = ttk.Combobox(
            agent_frame,
            textvariable=self._black_agent_var,
            values=AGENT_OPTIONS,
            state="readonly",
        )
        black_option.grid(row=2, column=0, pady=4, sticky=tk.EW)

        white_label = ttk.Label(agent_frame, text="白番エージェント")
        white_label.grid(row=1, column=1, pady=(8, 4), sticky=tk.W)
        style_label(white_label)

        self._white_agent_var = tk.StringVar(value=AGENT_OPTIONS[1])
        white_option = ttk.Combobox(
            agent_frame,
            textvariable=self._white_agent_var,
            values=AGENT_OPTIONS,
            state="readonly",
        )
        white_option.grid(row=2, column=1, pady=4, sticky=tk.EW)

        agent_button = ttk.Button(
            agent_frame,
            text="BATTLE",
            command=self._handle_start_agent_vs_agent,
        )
        agent_button.grid(row=3, column=1, pady=(10, 0), sticky=tk.E)
        style_button(agent_button)

    def _handle_start_agent_vs_human(self) -> None:
        """人間対エージェントを開始する。"""
        agent_name = self._human_agent_var.get()
        self._on_start_agent_vs_human(agent_name)

    def _handle_start_agent_vs_agent(self) -> None:
        """エージェント同士の対戦を開始する。"""
        black_agent = self._black_agent_var.get()
        white_agent = self._white_agent_var.get()
        self._on_start_agent_vs_agent(black_agent, white_agent)
