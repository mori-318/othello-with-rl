"""ゲーム進行を管理するコントローラーモジュール。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.game.board import BoardState
from app.game.othello_env import OthelloEnv


class GameController:
    """`OthelloEnv` をラップして盤面情報を提供するクラス。"""

    def __init__(self) -> None:
        """コントローラーを初期化する。"""
        # 環境を生成
        self.env = OthelloEnv()
        # 初期化した盤面を保持
        self.reset()

    def reset(self) -> np.ndarray:
        """環境を初期状態に戻す。

        Returns:
            np.ndarray: 現在の盤面(8x8)。
        """
        # 環境をリセット
        self.env.reset()
        # 盤面を返す
        return self.get_board()

    def get_board(self) -> np.ndarray:
        """現在の盤面を取得する。

        Returns:
            np.ndarray: 盤面配列。
        """
        # 盤面をコピーして返す
        return np.array(self.env.game.board, copy=True)

    def get_board_state(self) -> BoardState:
        """盤面の集計情報を取得する。

        Returns:
            BoardState: 盤面の石数情報。
        """
        # BoardStateに変換
        return BoardState.from_array(self.env.game.board)

    def current_player(self) -> int:
        """現在の手番プレイヤーを取得する。

        Returns:
            int: 黒(+1)または白(-1)。
        """
        # 手番を返す
        return self.env.game.player

    def legal_actions(self) -> List[int]:
        """現在の合法手リストを取得する。

        Returns:
            List[int]: 行動インデックスのリスト。
        """
        # 環境から合法手を取得
        return self.env.legal_actions()

    def legal_positions(self) -> List[Tuple[int, int]]:
        """合法手を盤面座標のリストとして取得する。

        Returns:
            List[Tuple[int, int]]: (row, col)の座標リスト。
        """
        # 合法手を座標リストに変換
        positions: List[Tuple[int, int]] = []
        for action in self.legal_actions():
            if action == 64:
                continue
            row, col = divmod(action, 8)
            positions.append((row, col))
        return positions

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """指定行動で環境を1手進める。

        Args:
            action (int): 0-63の盤面位置または64(パス)。

        Returns:
            Tuple[np.ndarray, float, bool, dict]: 次状態・報酬・終了フラグ・追加情報。
        """
        # 行動を適用
        return self.env.step(action)

    def is_finished(self) -> bool:
        """ゲーム終了かを判定する。

        Returns:
            bool: 終了していればTrue。
        """
        # 環境から終局を確認
        return self.env.game.is_terminal()

    def winner(self) -> int:
        """勝者を取得する。

        Returns:
            int: 黒(+1)、白(-1)、引き分け(0)。
        """
        # 勝者を返す
        return self.env.game.winner()
