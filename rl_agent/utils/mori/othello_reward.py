from typing import Optional

import numpy as np

from utils.mori.othello_game import OthelloGame


class OthelloReward:
    """
    終局時の石数差を [-1, 1] に正規化して返すシンプルな報酬関数。

    中間ステップでは常に 0 を返し、ゲーム終了時のみ報酬を計算する。
    視点を固定しない場合は、遷移前環境の手番を基準に石差を算出する。
    """

    def __init__(self, player: Optional[int] = None) -> None:
        """石差を評価する際の視点プレイヤーを設定する。

            player: 視点を固定したいプレイヤー（+1 / -1）。`None` なら遷移前状態の手番視点を使用。
        """

        self.player = None if player is None else int(player)

    def get_reward(
        self,
        env: OthelloGame,
        perspective: Optional[int] = None,
    ) -> float:
        """
        終局時の石差を [-1, 1] に正規化した報酬として返す。
        終局だが、置かれた石の数が64個に満たない場合は、勝ちの場合は1、負けの場合は-1を返す。
        終局でなければ報酬を返さない(一応, ０を返す)

        Args:
            env: 評価対象の終局盤面を保持する環境。
            perspective: 報酬を評価するプレイヤー視点（+1 / -1）。
                指定しない場合は `__init__` で設定されたプレイヤー、
                それも未設定なら `env.player` を用いる。

        Returns:
            終局であれば石差を 64 で正規化した値。ゲーム続行中であれば 0.0。
        """
        viewpoint = (
            int(perspective)
            if perspective is not None
            else self.player
            if self.player is not None
            else env.player
        )

        # 終局だが、置かれた石の数が64個に満たない場合は、勝ちの場合は1、負けの場合は-1を返す
        num_stones = int(np.count_nonzero(env.board))
        if env.is_terminal() and num_stones < 64:
            return 1.0 if env.board.sum() * viewpoint > 0 else -1.0

        # 終局でなければ報酬を返さない(一応, ０を返す)
        if not env.is_terminal():
            return 0.0

        score = int(env.board.sum())
        diff = score * viewpoint
        return float(diff) / 64.0

    def __call__(self, prev_env: OthelloGame, next_env: OthelloGame) -> float:
        perspective = (
            self.player
            if self.player is not None
            else prev_env.player
        )
        return self.get_reward(next_env, perspective)


class ShapedReward:
    def __init__(self, player: Optional[int] = None, eta: float = 0.05) -> None:
        """
        石差を評価する際の視点プレイヤーを設定する。

        Args:
            player: 視点を固定したいプレイヤー（+1 / -1）。`None` なら遷移前状態の手番視点を使用。
        """
        self.player = None if player is None else int(player)
        # shaping 全体のスケール。終局報酬(±1)を主役に保つため小さめ推奨
        self.eta = float(eta)

    def __call__(self, prev_env: OthelloGame, next_env: OthelloGame) -> float:
        """フェーズに応じて重みをスケジューリングし、終局報酬に微小な shaping を足す。

        - 序盤: モビリティ重視
        - 中盤〜終盤: 角/安定石/石差の比重を増やす
        """
        perspective = self.player if self.player is not None else prev_env.player

        # ベース: 終局時のみ非ゼロ（±1 または diff/64）。プロジェクト方針に一致
        base = OthelloReward(player=perspective)(prev_env, next_env)

        # ゲームフェーズ推定（空きマス数で線形に 0→1）
        n_empty = int(np.count_nonzero(next_env.board == 0))
        phase = 1.0 - (n_empty / 64.0)  # 0:序盤, 1:終盤
        phase = float(np.clip(phase, 0.0, 1.0))

        # 特徴量計算（次状態・視点固定）
        mobility = self._mobility(next_env, perspective)                 # [-1,1] 目安
        corner   = self._corner_score(next_env, perspective)             # [-1,1]
        stable   = self._stable_estimate(next_env, perspective)          # [-1,1] 近似
        frontier = self._frontier_score(next_env, perspective)           # 自石少→プラスにしたいので符号に注意
        diff     = float(next_env.board.sum() * perspective) / 64.0      # [-1,1]

        # 重みスケジュール
        w_mobility = 1.0 - phase                 # 序盤で高く、終盤で下げる
        w_corner   = phase                       # 終盤で高く
        w_stable   = phase                       # 終盤で高く
        w_frontier = phase * 0.5                 # 終盤寄りでやや加味
        w_diff     = phase ** 1.5                # 終盤で強く

        # フロンティアは「少ないほど良い」ため、自分フロンティアが少ないとプラスになるよう符号を反転
        shaping_raw = (
            w_mobility * mobility +
            w_corner   * corner +
            w_stable   * stable +
            w_frontier * (-frontier) +
            w_diff     * diff
        )
        shaping = self.eta * float(shaping_raw)
        return float(base + shaping)

    # --------- 特徴量計算（簡易実装） ---------
    def _mobility(self, env: OthelloGame, player: int) -> float:
        g = env.clone()
        my_moves = len(g.legal_moves(player))
        opp_moves = len(g.legal_moves(-player))
        # 正規化: 最大合法手は理論上 60 程度だが小さくスケール
        return float(my_moves - opp_moves) / 16.0

    def _corner_score(self, env: OthelloGame, player: int) -> float:
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        b = env.board
        my = 0
        opp = 0
        for r, c in corners:
            v = int(b[r, c])
            if v == player:
                my += 1
            elif v == -player:
                opp += 1
        return float(my - opp) / 4.0

    def _stable_estimate(self, env: OthelloGame, player: int) -> float:
        """簡易安定石推定: 角から同色が連続するエッジ石をカウント（近似）。"""
        b = env.board
        my = 0
        opp = 0
        # 上辺
        my += self._edge_run(b[0, :], player)
        opp += self._edge_run(b[0, :], -player)
        # 下辺
        my += self._edge_run(b[7, :], player)
        opp += self._edge_run(b[7, :], -player)
        # 左辺
        my += self._edge_run(b[:, 0], player)
        opp += self._edge_run(b[:, 0], -player)
        # 右辺
        my += self._edge_run(b[:, 7], player)
        opp += self._edge_run(b[:, 7], -player)
        # 粗い近似なので 0-1 に収めるため 16 で正規化（各辺最大連鎖 8 を想定）
        return float(my - opp) / 16.0

    def _edge_run(self, line: np.ndarray, color: int) -> int:
        # 角から内側へ同色が連続している長さ（両端）
        cnt = 0
        # 左端から
        for v in line:
            if v == color:
                cnt += 1
            else:
                break
        # 右端から
        cnt_r = 0
        for v in line[::-1]:
            if v == color:
                cnt_r += 1
            else:
                break
        return cnt + cnt_r

    def _frontier_score(self, env: OthelloGame, player: int) -> float:
        """フロンティア石（隣接に空きがある自石）を数える。"""
        b = env.board
        empties = np.argwhere(b == 0)
        # 空きに隣接する座標集合
        adj = set()
        for r, c in empties:
            r = int(r); c = int(c)
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                rr = r + dr; cc = c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    adj.add((rr, cc))
        # 自分/相手のフロンティア数
        my_f = 0
        opp_f = 0
        for rr, cc in adj:
            v = int(b[rr, cc])
            if v == player:
                my_f += 1
            elif v == -player:
                opp_f += 1
        # 多いほど悪いので正規化して返す（シグナルの大きさは小さく）
        return float(my_f - opp_f) / 16.0