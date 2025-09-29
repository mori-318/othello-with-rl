"""Tkinterキャンバスを用いたオセロ盤面描画ヘルパー。"""

from __future__ import annotations

import tkinter as tk
from typing import Iterable, Optional, Tuple

import numpy as np

GREEN_COLOR = "#2E8B57"
GRID_COLOR = "#004B23"
BLACK_STONE_COLOR = "#222222"
WHITE_STONE_COLOR = "#F4F4F4"
HIGHLIGHT_COLOR = "#FFD60A"
BOARD_SIZE = 8
CELL_SIZE = 64
CANVAS_SIZE = CELL_SIZE * BOARD_SIZE


class BoardCanvas(tk.Canvas):
    """オセロ盤を描画するためのカスタムキャンバス。"""

    def __init__(self, master: tk.Misc) -> None:
        """キャンバスを初期化する。

        Args:
            master (tk.Misc): 親ウィジェット。
        """
        super().__init__(
            master,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background=GREEN_COLOR,
            highlightthickness=0,
        )
        # 盤面の初期描画
        self.draw_board(np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int))

    def draw_board(
        self,
        board: np.ndarray,
        highlights: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> None:
        """盤面を描画する。

        Args:
            board (np.ndarray): 8x8の盤面配列。
            highlights (Optional[Iterable[Tuple[int, int]]]): ハイライトする座標群。
        """
        # 既存の描画をクリア
        self.delete("all")

        # グリッド線を描画
        for i in range(BOARD_SIZE + 1):
            coord = i * CELL_SIZE
            self.create_line(0, coord, CANVAS_SIZE, coord, fill=GRID_COLOR)
            self.create_line(coord, 0, coord, CANVAS_SIZE, fill=GRID_COLOR)

        # ハイライトセルを描画
        if highlights:
            for row, col in highlights:
                x0 = col * CELL_SIZE
                y0 = row * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    outline=HIGHLIGHT_COLOR,
                    width=3,
                )

        # 石を描画
        padding = CELL_SIZE * 0.15
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                value = int(board[row, col])
                if value == 0:
                    continue
                x0 = col * CELL_SIZE + padding
                y0 = row * CELL_SIZE + padding
                x1 = (col + 1) * CELL_SIZE - padding
                y1 = (row + 1) * CELL_SIZE - padding
                fill_color = BLACK_STONE_COLOR if value == 1 else WHITE_STONE_COLOR
                self.create_oval(x0, y0, x1, y1, fill=fill_color, outline="black")

    def event_to_cell(self, event: tk.Event) -> Tuple[int, int]:
        """イベント座標を盤面セルに変換する。

        Args:
            event (tk.Event): マウスイベント。

        Returns:
            Tuple[int, int]: 盤面上の行列インデックス。
        """
        row = int(event.y // CELL_SIZE)
        col = int(event.x // CELL_SIZE)
        return row, col

    def within_board(self, row: int, col: int) -> bool:
        """指定セルが盤面内か判定する。

        Args:
            row (int): 行インデックス。
            col (int): 列インデックス。

        Returns:
            bool: 盤面内ならTrue。
        """
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
