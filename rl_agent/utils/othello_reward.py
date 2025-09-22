from typing import List, Tuple
from utils.othello_game import OthelloGame, EMPTY, BLACK, WHITE


class OthelloPotential:
    def __init__(
        self,
        player: int,
        w_corner: float = 1.0,
        w_mob: float = 0.7,
        w_frontier: float = -0.4
    ):
        self.player = player
        self.w_corner = w_corner
        self.w_mob = w_mob
        self.w_frontier = w_frontier

    def phi(self, env: OthelloGame) -> float:
        corner_term = self._ratio(*self._count_corner(env))
        mob_term = self._ratio(
            len(env.legal_moves(self.player)),
            len(env.legal_moves(env.opponent(self.player)))
        )
        front_term = self._ratio(*self._count_frontier(env))
        return self.w_corner*corner_term + self.w_mob*mob_term + self.w_frontier*front_term

    def terminal_reward(self, env: OthelloGame) -> float:
        if not env.is_terminal():
            return 0.0
        pc, oc = self._count_stones(env)
        return 1.0 if pc > oc else -1.0 if pc < oc else 0.0

    # --- helpers ---
    def _ratio(self, a: int, b: int) -> float:
        s = a + b
        return (a - b) / max(1, s)

    def _count_stones(self, env: OthelloGame) -> Tuple[int, int]:
        pc = oc = 0
        opp = env.opponent(self.player)
        for r in range(8):
            for c in range(8):
                if env.board[r][c] == self.player: pc += 1
                elif env.board[r][c] == opp: oc += 1
        return pc, oc

    def _count_corner(self, env: OthelloGame) -> Tuple[int, int]:
        corners = [(0,0),(0,7),(7,0),(7,7)]
        pc = oc = 0
        opp = env.opponent(self.player)
        for r,c in corners:
            if env.board[r][c] == self.player: pc += 1
            elif env.board[r][c] == opp: oc += 1
        return pc, oc

    def _count_frontier(self, env: OthelloGame) -> Tuple[int, int]:
        pc = oc = 0
        opp = env.opponent(self.player)
        for r in range(8):
            for c in range(8):
                cell = env.board[r][c]
                if cell == EMPTY:
                    continue
                frontier = any(
                    0 <= r+dr < 8 and 0 <= c+dc < 8 and env.board[r+dr][c+dc] == EMPTY
                    for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc
                )
                if frontier:
                    if cell == self.player: pc += 1
                    elif cell == opp: oc += 1
        return pc, oc

class ShapedReward:
    """
    状態価値を用いた報酬を算出
    r = r_terminal(s') + eta * (gamma * Φ(s') - Φ(s))
    """
    def __init__(self, player: int, eta: float = 0.1, gamma: float = 0.99):
        self.player = player
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.potential = OthelloPotential(player)

    def get_reward(self, prev_env: OthelloGame, next_env: OthelloGame) -> float:
        r_term = self.potential.terminal_reward(next_env)
        phi_prev = self.potential.phi(prev_env)
        phi_next = self.potential.phi(next_env)
        shaped = self.gamma * phi_next - phi_prev
        return r_term + self.eta * shaped

    # 関数的に呼べるようにする（env.step の reward_fn(prev, next) 形式に対応）
    __call__ = get_reward