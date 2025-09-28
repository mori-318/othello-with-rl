from typing import List, Tuple

import numpy as np


# 定数
EMPTY, BLACK, WHITE = 0, 1, -1
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class OthelloGame:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.board[3, 3] = self.board[4, 4] = WHITE
        self.board[3, 4] = self.board[4, 3] = BLACK
        self.player = BLACK

    @staticmethod
    def opponent(player: int) -> int:
        """
        現在のプレイヤーの相手を返す
        Args:
            player(int): 現在のプレイヤー
        Returns:
            int: 相手のプレイヤー
        Examples:
            >>> OthelloGame.opponent(1)
            -1
            >>> OthelloGame.opponent(-1)
            1
        """
        return -player

    def clone(self):
        """盤面をコピーして新しいインスタンスを返す"""
        g = OthelloGame()
        g.board = self.board.copy()
        g.player = self.player
        return g

    def is_inside(self, r, c):
        """
        盤面の範囲内かを返す
        Args:
            r (int): 行
            c (int): 列
        Returns:
            bool: 盤面の範囲内か
        """
        return 0 <= r < 8 and 0 <= c < 8

    def legal_moves(self, player=None) -> List[Tuple[int, int]]:
        """
        合法手を返す
        Args:
            player (int, optional): プレイヤーの色.指定しない場合は現在のプレイヤーを返す
        Returns:
            List[Tuple[int, int]]: 合法手のリスト
        """
        if player is None:
            player = self.player
        empties = np.argwhere(self.board == EMPTY)
        moves: List[Tuple[int, int]] = []
        for r, c in empties:
            r_int = int(r)
            c_int = int(c)
            if self._would_flip(r_int, c_int, player):
                moves.append((r_int, c_int))
        return moves

    def _would_flip(self, r, c, player) -> bool:
        """
        石をひっくり返せるかを返す

        Args:
            r (int): 行
            c (int): 列
            player (int): プレイヤーの色

        Returns:
            bool: 石をひっくり返せるか
        """
        if self.board[r, c] != EMPTY:  # 空白でない場合は石をひっくり返せない
            return False

        opponent = self.opponent(player)
        for dr, dc in DIRECTIONS:
            rr = r + dr
            cc = c + dc
            if not self.is_inside(rr, cc) or self.board[rr, cc] != opponent:
                continue
            # 相手の石が連続する限り進む
            while self.is_inside(rr, cc) and self.board[rr, cc] == opponent:
                rr += dr
                cc += dc
            if self.is_inside(rr, cc) and self.board[rr, cc] == player:
                return True
        return False

    def play(self, r, c, player=None):
        """
        石を置く & 石をひっくり返す

        Args:
            r (int): 行
            c (int): 列
            player (int, optional): プレイヤーの色.指定しない場合は現在のプレイヤーを返す
        """
        if player is None:
            player = self.player
        assert self.board[r, c] == EMPTY  # 空白でない場合は石を置くことができない

        # 着手地点に石を配置
        self.board[r, c] = player

        flipped = []
        for dr, dc in DIRECTIONS:
            line = []
            rr = r + dr
            cc = c + dc
            while self.is_inside(rr, cc) and self.board[rr, cc] == self.opponent(player):  # 盤面の範囲内か & 相手の石が置かれている場所か
                line.append((rr, cc))  # 石をひっくり返す場所を追加
                rr += dr
                cc += dc
            if line and self.is_inside(rr, cc) and self.board[rr, cc] == player:  # 石をひっくり返す場所が存在 & 盤面の範囲内か & 自分の石をみたか
                flipped.extend(line)  # 石をひっくり返す場所を追加
        if not flipped:  # 石をひっくり返す場所が存在しない場合は不正な手
            # 着手地点を元に戻す
            self.board[r, c] = EMPTY
            raise ValueError("Illegal move")
        for rr, cc in flipped:  # 石をひっくり返す
            self.board[rr, cc] = player
        self.player = self.opponent(player)
        # 現在のプレイヤーが合法手がない場合は相手の番
        if not self.legal_moves(self.player):
            self.player = self.opponent(self.player)

    def is_terminal(self) -> bool:
        """
        終局かどうかを判定する

        Returns:
            bool: 終局かどうか
        """
        if self.legal_moves(BLACK):
            return False
        if self.legal_moves(WHITE):
            return False
        return True

    def game_score(self) -> int:
        """
        黒(+1)・白(-1)の石数差を返す

        Returns:
            int: 黒(+1)・白(-1)の石数差
        """
        return int(self.board.sum())

    def winner(self) -> int:
        """
        勝者を返す（黒:+1, 白:-1, 引き分け:0）

        Returns:
            int: 勝者
        """
        score = self.game_score()
        if score > 0:
            return BLACK
        if score < 0:
            return WHITE
        return 0
