"""人間対エージェント対戦ビュー。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from app.controllers.agent_loader import create_agent
from app.controllers.game_controller import GameController
from app.game.renderer import BoardCanvas
from app.utils.style import (
    RETRO_COLORS,
    style_button,
    style_label,
    style_panel,
)


class AgentVsHumanView(tk.Frame):
    """人間とエージェントの対戦を行うビュー。"""

    def __init__(
        self,
        master: tk.Misc,
        agent_name: str,
        on_back: Callable[[], None],
    ) -> None:
        """ビューを初期化する。

        Args:
            master (tk.Misc): 親ウィジェット。
            agent_name (str): 対戦するエージェント名。
            on_back (Callable[[], None]): ホーム画面に戻る際のコールバック。
        """
        super().__init__(master, background=RETRO_COLORS["bg"])

        # コールバックとエージェント情報を保持
        self._on_back = on_back
        self._agent = create_agent(agent_name)
        self._controller = GameController()
        self._animation_after: Optional[str] = None

        # UIとゲーム状態を初期化
        self._build_view()
        self._update_status()
        self._play_start_animation()

    def _build_view(self) -> None:
        """UI部品を構築する。"""
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # 左側: 盤面領域
        board_frame = ttk.Frame(self, padding=12)
        board_frame.grid(row=0, column=0, padx=15, pady=12, sticky=tk.NSEW)
        board_frame.columnconfigure(0, weight=1)
        board_frame.rowconfigure(1, weight=1)
        style_panel(board_frame)

        # ターン表示
        self._turn_label = ttk.Label(board_frame, text="TURN", anchor=tk.CENTER)
        self._turn_label.grid(row=0, column=0, pady=(0, 15))
        style_label(self._turn_label, emphasis=True)

        self._board_canvas = BoardCanvas(board_frame)
        self._board_canvas.grid(row=1, column=0, sticky=tk.NSEW)
        self._board_canvas.bind("<Button-1>", self._on_board_click)

        # 右側: 情報表示
        info_frame = ttk.Frame(self, padding=12)
        info_frame.grid(row=0, column=1, padx=15, pady=12, sticky=tk.NSEW)
        info_frame.columnconfigure(0, weight=1)
        style_panel(info_frame)

        status_title = ttk.Label(info_frame, text="STATUS")
        status_title.grid(row=0, column=0, sticky=tk.W)
        style_label(status_title, emphasis=True)

        self._status_label = ttk.Label(info_frame, text="", wraplength=280, justify=tk.LEFT)
        self._status_label.grid(row=1, column=0, pady=(8, 6), sticky=tk.W)
        style_label(self._status_label)

        self._score_label = ttk.Label(info_frame, text="")
        self._score_label.grid(row=2, column=0, pady=(0, 12), sticky=tk.W)
        style_label(self._score_label)

        button_frame = ttk.Frame(info_frame)
        button_frame.grid(row=3, column=0, sticky=tk.EW)
        style_panel(button_frame)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        back_button = ttk.Button(button_frame, text="BACK", command=self._on_back)
        back_button.grid(row=0, column=0, padx=3, sticky=tk.EW)
        style_button(back_button)

        reset_button = ttk.Button(button_frame, text="RESET", command=self._reset_game)
        reset_button.grid(row=0, column=1, padx=3, sticky=tk.EW)
        style_button(reset_button)

        # 初期盤面を描画
        self._draw_board()

    def _reset_game(self) -> None:
        """ゲームを初期化する。"""
        self._controller.reset()
        self._draw_board()
        self._update_status()
        self._play_start_animation()

    def _on_board_click(self, event: tk.Event) -> None:
        """盤面クリック時の処理。"""
        if self._controller.is_finished():
            return

        row, col = self._board_canvas.event_to_cell(event)
        if not self._board_canvas.within_board(row, col):
            return

        action = row * 8 + col
        if action not in self._controller.legal_actions():
            return

        # 人間の手を適用
        self._controller.step(action)

        # 終局チェック
        if not self._controller.is_finished():
            # エージェント手番で行動
            agent_action = self._agent.select_action(self._controller.env)
            self._controller.step(agent_action)
        else:
            self._show_finish_effect()

        self._draw_board()
        self._update_status()

    def _draw_board(self) -> None:
        """現在の盤面を描画する。"""
        board = self._controller.get_board()
        highlights = self._controller.legal_positions()
        self._board_canvas.draw_board(board, highlights)

    def _update_status(self) -> None:
        """ステータス表示を更新する。"""
        board_state = self._controller.get_board_state()
        black, white, empty = board_state.stone_counts()
        player = self._controller.current_player()

        status_lines = [f"黒石: {black}", f"白石: {white}", f"空き: {empty}"]
        if self._controller.is_finished():
            winner = self._controller.winner()
            if winner == 1:
                result = "黒の勝利"
            elif winner == -1:
                result = "白の勝利"
            else:
                result = "引き分け"
            status_lines.append(f"結果: {result}")
        else:
            turn_text = "黒番" if player == 1 else "白番"
            status_lines.append(f"手番: {turn_text}")

        self._status_label.config(text="\n".join(status_lines))
        self._turn_label.config(
            text="YOUR TURN" if player == 1 else "AGENT TURN",
            foreground=RETRO_COLORS["accent"],
            font=("Press Start 2P", 24),
        )
        self._score_label.config(text=f"エージェント: {self._agent.name}")

    def _play_start_animation(self) -> None:
        """ゲーム開始時の簡易アニメーションを実行する。"""
        if self._animation_after is not None:
            self.after_cancel(self._animation_after)
            self._animation_after = None

        overlay = tk.Label(
            self._board_canvas,
            text="READY",
            font=("Press Start 2P", 28),
            fg=RETRO_COLORS["accent"],
            bg=RETRO_COLORS["bg"],
        )
        overlay.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        def step(message: str, delay: int) -> None:
            overlay.config(text=message)
            overlay.after(delay, overlay.place_forget)

        overlay.after(600, lambda: step("FIGHT!", 600))
        self._animation_after = overlay.after(1200, overlay.destroy)

    def _show_finish_effect(self) -> None:
        """勝敗に応じたエフェクトを表示する。"""
        winner = self._controller.winner()
        if winner == 0:
            message = "DRAW"
            color = RETRO_COLORS["subtext"]
        elif winner == 1:
            message = "YOU WIN"
            color = RETRO_COLORS["success"]
        else:
            message = "YOU LOSE"
            color = RETRO_COLORS["alert"]

        overlay = tk.Label(
            self._board_canvas,
            text=message,
            font=("Press Start 2P", 28),
            fg=color,
            bg=RETRO_COLORS["bg"],
        )
        overlay.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        def blink(count: int = 4) -> None:
            if count <= 0:
                overlay.destroy()
                return
            current = overlay.cget("foreground")
            overlay.config(foreground=RETRO_COLORS["bg"] if current == color else color)
            overlay.after(200, lambda: blink(count - 1))

        blink()
