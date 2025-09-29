"""オセロ盤面状態の補助機能を提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BoardState:
    """盤面と石数情報を保持するデータクラス。"""

    board: np.ndarray

    @classmethod
    def from_array(cls, board: np.ndarray) -> "BoardState":
        """NumPy配列から `BoardState` を生成する。

        Args:
            board (np.ndarray): 盤面を表す8x8のNumPy配列。

        Returns:
            BoardState: 生成された盤面状態。
        """
        # 盤面をコピーして外部変更の影響を受けないようにする
        board_copy = np.array(board, copy=True)
        return cls(board=board_copy)

    def stone_counts(self) -> Tuple[int, int, int]:
        """石数を計算する。

        Returns:
            Tuple[int, int, int]: 黒石数、白石数、空マス数。
        """
        # 石数を集計
        black = int(np.sum(self.board == 1))
        white = int(np.sum(self.board == -1))
        empty = int(np.sum(self.board == 0))
        return black, white, empty
