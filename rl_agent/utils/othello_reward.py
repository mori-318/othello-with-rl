from typing import Tuple, Optional
from utils.othello_game import OthelloGame, EMPTY, BLACK, WHITE


class OthelloPotential:
    """
    オセロ用のポテンシャル関数 φ(s) を計算するクラス。
    - 目的: Potential-Based Reward Shaping (PBRS) 用の φ(s) を提供
    - 重要: φ(s) は「視点一貫（手番＝自分視点）」で計算することを想定

    特徴量（差分は常に 自分(me) - 相手(opp) ）:
        - corners: 4隅の占有差
        - mobility: 合法手数差
        - frontier: フロンティア石差（空きマスに隣接する石）

    各差分はラプラス平滑化付きの比率 `_ratio(a, b) = ((a+β) - (b+β)) / ((a+β) + (b+β))` に変換。
    これにより a+b が小さい局面でもノイズを抑える。

    φ(s) の値域は概ね [-1, 1] 付近になるように重みを設定しておくのが望ましい。
    """

    def __init__(
        self,
        w_corner: float = 1.0,
        w_mob: float = 0.7,
        w_frontier: float = -0.4,
        beta: float = 1.0,
        dynamic_phase_weighting: bool = True,
    ) -> None:
        """
        Args:
            w_corner: コーナー項の基礎係数
            w_mob: 可動性（合法手数）項の基礎係数
            w_frontier: フロンティア項の基礎係数（通常は負）
            beta: ラプラス平滑化の強さ（1〜2 推奨）
            dynamic_phase_weighting: 局面の進行度（空きマス数）で係数をスケーリングするか
        """
        self.w_corner = float(w_corner)
        self.w_mob = float(w_mob)
        self.w_frontier = float(w_frontier)
        self.beta = float(beta)
        self.dynamic_phase_weighting = bool(dynamic_phase_weighting)

    # ====== 公開API ======

    def phi(self, env: OthelloGame, player: Optional[int] = None) -> float:
        """
        φ(s) を返す。手番（player 引数が指定されていればそのプレイヤー、
        指定がなければ env.player）視点で計算する。
        観測が常に「自分視点」に正規化されている環境なら、それと一致する。
        """
        me = env.player if player is None else int(player)
        opp = env.opponent(me)

        # 差分特徴（比率）を計算
        corner_term = self._ratio(*self._count_corner(env, me, opp))
        mob_term = self._ratio(len(env.legal_moves(me)), len(env.legal_moves(opp)))
        front_term = self._ratio(*self._count_frontier(env, me, opp))

        # 係数（局面進行に応じた調整を入れると安定しやすい）
        w_corner, w_mob, w_frontier = self._phase_weights(env)

        return (
            w_corner * corner_term
            + w_mob * mob_term
            + w_frontier * front_term
        )

    def terminal_reward(self, env: OthelloGame, player: Optional[int] = None) -> float:
        """
        手番（player 引数が指定されていればそのプレイヤー、指定がなければ
        env.player）視点の終局報酬を返す。
        勝ち=+1, 負け=-1, 引き分け=0。非終局では 0。
        """
        if not env.is_terminal():
            return 0.0
        me = env.player if player is None else int(player)
        opp = env.opponent(me)
        pc, oc = self._count_stones(env, me, opp)
        return 1.0 if pc > oc else -1.0 if pc < oc else 0.0

    # ====== 内部ユーティリティ ======

    def _ratio(self, a: int, b: int) -> float:
        # ラプラス平滑化で a+b が小さいときのノイズを低減
        ap = a + self.beta
        bp = b + self.beta
        denom = ap + bp
        if denom <= 0:
            return 0.0
        return (ap - bp) / denom

    def _count_stones(self, env: OthelloGame, me: int, opp: int) -> Tuple[int, int]:
        pc = oc = 0
        for r in range(8):
            for c in range(8):
                v = env.board[r][c]
                if v == me:
                    pc += 1
                elif v == opp:
                    oc += 1
        return pc, oc

    def _count_corner(self, env: OthelloGame, me: int, opp: int) -> Tuple[int, int]:
        corners = ((0, 0), (0, 7), (7, 0), (7, 7))
        pc = oc = 0
        for r, c in corners:
            v = env.board[r][c]
            if v == me:
                pc += 1
            elif v == opp:
                oc += 1
        return pc, oc

    def _count_frontier(self, env: OthelloGame, me: int, opp: int) -> Tuple[int, int]:
        """
        フロンティア石 = 空きマスに隣接する石（8近傍のいずれかが EMPTY）
        """
        pc = oc = 0
        for r in range(8):
            for c in range(8):
                cell = env.board[r][c]
                if cell == EMPTY:
                    continue
                frontier = any(
                    0 <= r + dr < 8
                    and 0 <= c + dc < 8
                    and env.board[r + dr][c + dc] == EMPTY
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if dr or dc
                )
                if frontier:
                    if cell == me:
                        pc += 1
                    elif cell == opp:
                        oc += 1
        return pc, oc

    def _phase_weights(self, env: OthelloGame) -> Tuple[float, float, float]:
        """
        空きマス数に応じて係数を補間する（任意）。序中盤は mobility 比重、
        終盤は corner/フロンティア（安定度）比重をやや上げる。
        """
        if not self.dynamic_phase_weighting:
            return (self.w_corner, self.w_mob, self.w_frontier)

        # 空きマス数（0=終局）。終盤ほど t→1、序盤ほど t→0 にしたいので 1 - (空き/64)
        empties = 64 - self._occupied_count(env)
        # 進行度 p: 0(序盤) → 1(終盤)
        p = min(1.0, max(0.0, 1.0 - empties / 64.0))

        # 緩やかな補間（好みに応じ調整可）
        # mobility は序盤寄り（p が小さいとき強め）、終盤で少し弱める
        w_mob = self.w_mob * (1.0 - 0.4 * p)
        # corner は終盤に向けてやや重く
        w_corner = self.w_corner * (1.0 + 0.3 * p)
        # frontier（通常は負）は終盤にやや強め（確定石/外側の安定に近い意味合い）
        w_frontier = self.w_frontier * (1.0 + 0.3 * p)

        return (w_corner, w_mob, w_frontier)

    def _occupied_count(self, env: OthelloGame) -> int:
        cnt = 0
        for r in range(8):
            for c in range(8):
                if env.board[r][c] != EMPTY:
                    cnt += 1
        return cnt


class ShapedReward:
    """
    Potential-Based Reward Shaping（PBRS）による整形報酬。
      r = r_terminal(s') + eta * (gamma * Φ(s') - Φ(s))

    実装上のポイント:
        - φ は「手番視点」で一貫して計算（player 引数で視点固定可）
        - next_env が終局のときは shaping を 0 にして、終局報酬の純度を保つ
        - PPO/GAE で使う gamma と同じ値を使用すること
    """

    def __init__(
        self,
        player: Optional[int] = None,
        gamma: float = 0.99,
        eta: float = 0.1,
        clip_shaping: Optional[float] = None,
        potential: Optional[OthelloPotential] = None,
    ) -> None:
        """
        Args:
            player: φ の視点を固定したい場合に指定するプレイヤー（+1/-1）。
            gamma: 割引率（学習側と一致させる）
            eta: シェーピングの強さ（0.05〜0.2 目安）
            clip_shaping: 途中の shaping 値を [-clip, clip] にクリップ（None なら無効）
            potential: 既製の OthelloPotential を差し込む場合
        """
        self.player = None if player is None else int(player)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.clip_shaping = float(clip_shaping) if clip_shaping is not None else None
        self.potential = potential if potential is not None else OthelloPotential()

    def get_reward(self, prev_env: OthelloGame, next_env: OthelloGame) -> float:
        perspective = self.player if self.player is not None else prev_env.player

        # 終局報酬は next_env（遷移後の状態）基準・手番視点で定義
        r_term = self.potential.terminal_reward(next_env, player=perspective)

        # 終局ではシェーピングを入れない（φ(s_terminal)=0 と等価）
        if next_env.is_terminal():
            return r_term

        phi_prev = self.potential.phi(prev_env, player=perspective)
        phi_next = self.potential.phi(next_env, player=perspective)

        shaped = self.gamma * phi_next - phi_prev

        if self.clip_shaping is not None:
            c = self.clip_shaping
            if shaped > c:
                shaped = c
            elif shaped < -c:
                shaped = -c

        return r_term + self.eta * shaped

    # env.step のコールバック形式に合わせて関数呼び出し可能に
    __call__ = get_reward