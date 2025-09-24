import numpy as np
from utils.mori.othello_game import OthelloGame

class OthelloEnv:
    """
    Othello (Reversi) 環境の Gym 風クラス。

    このクラスは強化学習のための環境インターフェースを提供します。
    盤面の初期化、合法手の生成、行動適用、報酬計算、終局判定を行います。

    Attributes:
        game (OthelloGame): ゲーム状態を保持する OthelloGame インスタンス。
        player (int): 現在のプレイヤー (BLACK=1, WHITE=-1)。
    """

    def __init__(self):
        """ コンストラクタ """
        self.game = OthelloGame()
        self.player = self.game.player

    def reset(self):
        """
        環境を初期状態にリセットし、観測を返す
        Returns:
            observation (np.ndarray): 観測 (2, 8, 8, float32)
                - [0] 盤面: -1(白), 0(空), 1(黒)
                - [1] 手番: 盤面と同形状で、全要素が現手番(BLACK=1/WHITE=-1)
        """
        self.game = OthelloGame()
        self.player = self.game.player
        return self.get_state()

    def step(self, action, reward_fn=None):
        """
        １ステップ環境を進める
        Args:
            action (int): 0~64の整数（64はパス）
            reward_fn (Reward, Optional): 報酬関数のインスタンス
                指定された場合、行動適用後に報酬を計算
        Returns:
            tuple:
                - observation (np.ndarray): 次の観測 (2, 8, 8, float32)
                - reward (float): 報酬
                - done (bool): ゲーム終了フラグ
                - info (dict): 追加情報
        """
        done = False
        reward = 0.0

        prev_game = self.game.clone()

        try:
            if action == 64:  # パスの場合
                # パスして相手番
                self.game.player = self.game.opponent(self.game.player)
            else:
                r, c = divmod(action, 8)
                self.game.play(r, c, self.game.player)
        except ValueError:
            # 不正手なら、即終了 & 負の報酬
            done = True
            reward = -10.0
            return self.get_state(), reward, done, {}

        # 報酬を計算
        if reward_fn is not None:
            reward = reward_fn(prev_game, self.game)

        done = self.game.is_terminal()
        self.player = self.game.player

        return self.get_state(), reward, done, {}

    def get_state(self):
        """
        現在の観測を numpy 配列で取得
        Returns:
            observation (np.ndarray): (2, 8, 8, float32)
                - [0] 盤面: -1(白), 0(空), 1(黒)
                - [1] 手番: 盤面と同形状で、全要素が現手番(BLACK=1/WHITE=-1)
        """
        board = self.game.board.astype(np.float32)
        player_plane = np.full(board.shape, float(self.game.player), dtype=np.float32)
        return np.stack((board, player_plane), axis=0)

    def legal_actions(self):
        """
        現在のプレイヤーの合法手を整数インデックスのリストとして返す
        Returns:
            actions (list): 合法手の整数インデックスのリスト
        """
        moves = self.game.legal_moves(self.player)
        if not moves:
            return [64]
        return [r * 8 + c for r, c in moves]

    def render(self):
        """
        盤面をコンソールに出力する
        """
        board_np = np.array(self.game.board)
        print(board_np)