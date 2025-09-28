"""Tkinterアプリ全体を管理するルートクラス。"""

from __future__ import annotations

import tkinter as tk
from typing import Optional

from app.ui.home_view import HomeView
from app.ui.agent_vs_human_view import AgentVsHumanView
from app.ui.agent_vs_agent_view import AgentVsAgentView


class AppRoot:
    """Tkinterアプリ全体を管理するクラス。"""

    def __init__(self, master: tk.Tk) -> None:
        """ルートウィンドウを初期化する。

        Args:
            master (tk.Tk): Tkinterのルートウィンドウ。
        """
        # ルートウィンドウ保持
        self.master = master

        # コンテンツ領域のフレームを構築
        self.container = tk.Frame(master)
        self.container.pack(fill=tk.BOTH, expand=True)

        # 現在表示しているビューの参照
        self.current_view: Optional[tk.Frame] = None

        # 初期画面としてホームを表示
        self.show_home()

    def clear_view(self) -> None:
        """現在のビューを破棄する。"""
        # 既存ビューが存在する場合は破棄
        if self.current_view is not None:
            self.current_view.destroy()
            self.current_view = None

    def show_home(self) -> None:
        """ホーム画面を表示する。"""
        # 既存ビューをクリア
        self.clear_view()

        # ホームビューを生成して配置
        view = HomeView(
            master=self.container,
            on_start_agent_vs_human=self.show_agent_vs_human,
            on_start_agent_vs_agent=self.show_agent_vs_agent,
        )
        view.pack(fill=tk.BOTH, expand=True)
        self.current_view = view

    def show_agent_vs_human(self, agent_name: str) -> None:
        """人間対エージェント画面を表示する。

        Args:
            agent_name (str): 使用するエージェント名。
        """
        # 既存ビューをクリア
        self.clear_view()

        # 人間対エージェントビューを生成
        view = AgentVsHumanView(
            master=self.container,
            agent_name=agent_name,
            on_back=self.show_home,
        )
        view.pack(fill=tk.BOTH, expand=True)
        self.current_view = view

    def show_agent_vs_agent(self, black_agent: str, white_agent: str) -> None:
        """エージェント同士の対戦画面を表示する。

        Args:
            black_agent (str): 黒番エージェント名。
            white_agent (str): 白番エージェント名。
        """
        # 既存ビューをクリア
        self.clear_view()

        # エージェント同士対戦ビューを生成
        view = AgentVsAgentView(
            master=self.container,
            black_agent=black_agent,
            white_agent=white_agent,
            on_back=self.show_home,
        )
        view.pack(fill=tk.BOTH, expand=True)
        self.current_view = view
