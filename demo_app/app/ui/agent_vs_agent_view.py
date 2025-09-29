"""エージェント同士の自動対戦ビュー。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

from app.controllers.agent_loader import create_agent
from app.controllers.game_controller import GameController
from app.game.renderer import BoardCanvas
from app.utils.style import (
    RETRO_COLORS,
    style_button,
    style_label,
    style_panel,
)

TURN_INTERVAL_MS = 500


class AgentVsAgentView(tk.Frame):
    """エージェント同士の対局を表示するビュー。"""

    def __init__(
        self,
        master: tk.Misc,
        black_agent: str,
        white_agent: str,
        on_back: Callable[[], None],
    ) -> None:
        """ビューを初期化する。

        Args:
            master (tk.Misc): 親ウィジェット。
            black_agent (str): 黒番エージェント名。
            white_agent (str): 白番エージェント名。
            on_back (Callable[[], None]): ホームに戻るコールバック。
        """
        super().__init__(master, background=RETRO_COLORS["bg"])

        # コールバックとエージェント情報を保持
        self._on_back = on_back
        self._controller = GameController()
        self._agents: Dict[int, str] = {
            1: black_agent,
            -1: white_agent,
        }
        self._agent_instances = {
            1: create_agent(black_agent),
            -1: create_agent(white_agent),
        }
        self._after_id: Optional[str] = None
        self._running = False
        self._animation_after: Optional[str] = None

        # UI構築
        self._build_view()
        self._update_status()
        self._play_start_animation()

    def _build_view(self) -> None:
        """UI部品を構築する。"""
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        # 盤面フレーム
        board_frame = ttk.Frame(self, padding=12)
        board_frame.grid(row=0, column=0, padx=15, pady=12, sticky=tk.NSEW)
        board_frame.columnconfigure(0, weight=1)
        board_frame.rowconfigure(1, weight=1)
        style_panel(board_frame)

        self._turn_label = ttk.Label(board_frame, text="TURN", anchor=tk.CENTER)
        self._turn_label.grid(row=0, column=0, pady=(0, 15))
        style_label(self._turn_label, emphasis=True)

        self._board_canvas = BoardCanvas(board_frame)
        self._board_canvas.grid(row=1, column=0, sticky=tk.NSEW)

        # 情報・操作フレーム
        info_frame = ttk.Frame(self, padding=12)
        info_frame.grid(row=0, column=1, padx=15, pady=12, sticky=tk.NSEW)
        info_frame.columnconfigure(0, weight=1)
        style_panel(info_frame)

        agent_info_text = (
            f"黒: {self._agents[1]}\n"
            f"白: {self._agents[-1]}"
        )
        self._agents_label = ttk.Label(info_frame, text=agent_info_text, justify=tk.LEFT)
        self._agents_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        style_label(self._agents_label, emphasis=True)

        self._status_label = ttk.Label(info_frame, text="", wraplength=260)
        self._status_label.grid(row=1, column=0, pady=(0, 10), sticky=tk.W)
        style_label(self._status_label)

        button_frame = ttk.Frame(info_frame)
        button_frame.grid(row=2, column=0, sticky=tk.EW)
        style_panel(button_frame)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        self._start_button = ttk.Button(button_frame, text="START", command=self._start_game)
        self._start_button.grid(row=0, column=0, padx=2, sticky=tk.EW)
        style_button(self._start_button)

        self._stop_button = ttk.Button(button_frame, text="STOP", command=self._stop_game, state=tk.DISABLED)
        self._stop_button.grid(row=1, column=0, padx=2, pady=3, sticky=tk.EW)
        style_button(self._stop_button)

        back_button = ttk.Button(button_frame, text="BACK", command=self._handle_back)
        back_button.grid(row=0, column=1, columnspan=2, padx=2, sticky=tk.EW)
        style_button(back_button)

        # 初期盤面描画
        self._draw_board()

    def _handle_back(self) -> None:
        """ホームに戻る処理を行う。"""
        self._stop_game()
        self._on_back()

    def _start_game(self) -> None:
        """対局を開始する。"""
        if self._running:
            return
        # ゲームをリセット
        self._controller.reset()
        self._running = True
        self._start_button.config(state=tk.DISABLED)
        self._stop_button.config(state=tk.NORMAL)
        self._draw_board()
        self._update_status()
        self._play_start_animation()
        self._schedule_next_turn()

    def _stop_game(self) -> None:
        """対局を停止する。"""
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._animation_after is not None:
            self.after_cancel(self._animation_after)
            self._animation_after = None
        self._running = False
        self._start_button.config(state=tk.NORMAL)
        self._stop_button.config(state=tk.DISABLED)

    def _schedule_next_turn(self) -> None:
        """次のターン処理を予約する。"""
        if not self._running:
            return
        self._after_id = self.after(TURN_INTERVAL_MS, self._play_turn)

    def _play_turn(self) -> None:
        """1ターン分の行動を実行する。"""
        if not self._running:
            return
        if self._controller.is_finished():
            self._stop_game()
            self._update_status()
            self._show_finish_effect()
            return

        player = self._controller.current_player()
        agent = self._agent_instances[player]
        action = agent.select_action(self._controller.env)
        self._controller.step(action)

        self._draw_board()
        self._update_status()
        self._schedule_next_turn()

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

        lines = [
            f"黒石: {black}",
            f"白石: {white}",
            f"空き: {empty}",
        ]
        if self._controller.is_finished():
            winner = self._controller.winner()
            if winner == 1:
                lines.append("結果: 黒の勝利")
            elif winner == -1:
                lines.append("結果: 白の勝利")
            else:
                lines.append("結果: 引き分け")
        else:
            turn_text = "黒番" if player == 1 else "白番"
            lines.append(f"手番: {turn_text}")
        self._status_label.config(text="\n".join(lines))
        turn_text = "黒番" if player == 1 else "白番"
        self._turn_label.config(
            text=turn_text,
            foreground=RETRO_COLORS["accent"],
            font=("Press Start 2P", 24),
        )

    def destroy(self) -> None:  # type: ignore[override]
        """ウィジェット破棄時にタイマーを停止する。"""
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._animation_after is not None:
            self.after_cancel(self._animation_after)
            self._animation_after = None
        super().destroy()

    def _play_start_animation(self) -> None:
        """ゲーム開始時のエフェクトを表示する。"""
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

        overlay.after(600, lambda: step("BATTLE", 600))
        self._animation_after = overlay.after(1200, overlay.destroy)

    def _show_finish_effect(self) -> None:
        """勝敗に応じたメッセージを表示する。"""
        winner = self._controller.winner()
        if winner == 0:
            message = "DRAW"
            color = RETRO_COLORS["subtext"]
        elif winner == 1:
            message = f"{self._agents[1].upper()} WINS"
            color = RETRO_COLORS["success"]
        else:
            message = f"{self._agents[-1].upper()} WINS"
            color = RETRO_COLORS["alert"]

        overlay = tk.Label(
            self._board_canvas,
            text=message,
            font=("Press Start 2P", 24),
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
