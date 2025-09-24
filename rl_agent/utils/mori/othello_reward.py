from typing import Optional

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

        if not env.is_terminal():
            return 0.0

        score = int(env.board.sum())
        diff = score * viewpoint
        return float(diff) / 64.0

    __call__ = get_reward