# %% [markdown]
# ## DQNでエージェントを構築（修正版：2ch入力 + ポテンシャル型シェーピング）

# %%
import os
import random
import math
from typing import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import copy
from collections import deque
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchinfo

import ipytest
ipytest.autoconfig()

# %% [markdown]
# ## オセロ環境

# %% [markdown]
# ### オセロゲーム

# %%
EMPTY, BLACK, WHITE = 0, 1, -1
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

class OthelloGame:
    def __init__(self):
        self.board = [[EMPTY for _ in range(8)] for _ in range(8)]
        self.board[3][3] = self.board[4][4] = WHITE
        self.board[3][4] = self.board[4][3] = BLACK
        self.player = BLACK

    @staticmethod
    def opponent(player: int) -> int:
        return -player

    def clone(self):
        g = OthelloGame()
        g.board = [row[:] for row in self.board]
        g.player = self.player
        return g

    def inside(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def legal_moves(self, player=None) -> List[Tuple[int, int]]:
        if player is None:
            player = self.player
        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r][c] != EMPTY:
                    continue
                if self._would_flip(r, c, player):
                    moves.append((r, c))
        return moves

    def _would_flip(self, r, c, player) -> bool:
        if self.board[r][c] != EMPTY:
            return False
        for dr, dc in DIRECTIONS:
            rr, cc = r + dr, c + dc
            seen_opp = False
            while self.inside(rr, cc) and self.board[rr][cc] == self.opponent(player):
                seen_opp = True
                rr += dr; cc += dc
            if seen_opp and self.inside(rr, cc) and self.board[rr][cc] == player:
                return True
        return False

    def play(self, r, c, player=None):
        if player is None:
            player = self.player
        assert self.board[r][c] == EMPTY
        flipped = []
        for dr, dc in DIRECTIONS:
            line = []
            rr, cc = r + dr, c + dc
            while self.inside(rr,cc) and self.board[rr][cc] == self.opponent(player):
                line.append((rr,cc))
                rr += dr; cc += dc
            if line and self.inside(rr,cc) and self.board[rr][cc] == player:
                flipped.extend(line)
        if not flipped:
            raise ValueError("Illegal move")
        self.board[r][c] = player
        for rr,cc in flipped:
            self.board[rr][cc] = player
        self.player = self.opponent(player)
        if not self.legal_moves(self.player):
            self.player = self.opponent(self.player)

    def is_terminal(self) -> bool:
        if self.legal_moves(BLACK): return False
        if self.legal_moves(WHITE): return False
        return True

    def game_score(self) -> int:
        s = 0
        for r in range(8):
            for c in range(8):
                s += self.board[r][c]
        return s

    def winner(self) -> int:
        s = self.game_score()
        return BLACK if s > 0 else WHITE if s < 0 else 0

# %% [markdown]
# ### 報酬（ポテンシャル型シェーピング）

# %%
class OthelloPotential:
    """
    Φ(s): 角・モビリティ・フロンティアを差分正規化で線形結合
    """
    def __init__(self, player: int,
                 w_corner: float = 1.0,
                 w_mob: float = 0.7,
                 w_frontier: float = -0.4):
        self.player = player
        self.w_corner = w_corner
        self.w_mob = w_mob
        self.w_frontier = w_frontier

    def phi(self, env: OthelloGame) -> float:
        corner_term = self._ratio(*self._count_corner(env))
        mob_term = self._ratio(len(env.legal_moves(self.player)),
                               len(env.legal_moves(env.opponent(self.player))))
        front_term = self._ratio(*self._count_frontier(env))
        return self.w_corner*corner_term + self.w_mob*mob_term + self.w_frontier*front_term

    def terminal_reward(self, env: OthelloGame) -> float:
        if not env.is_terminal():
            return 0.0
        pc, oc = self._count_stones(env)
        return 1.0 if pc > oc else -1.0 if pc < oc else 0.0

    # --- helpers (player 視点) ---
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

# %% [markdown]
# ### Gym風API（報酬は prev/next を渡して評価）

# %%
class OthelloEnv:
    def __init__(self):
        self.game = OthelloGame()
        self.player = self.game.player

    def reset(self):
        self.game = OthelloGame()
        self.player = self.game.player
        return self.get_state()

    def step(self, action, reward_fn=None):
        done = False
        reward = 0.0

        prev_game = self.game.clone()
        try:
            if action == 64:  # パス
                self.game.player = self.game.opponent(self.game.player)
            else:
                r, c = divmod(action, 8)
                self.game.play(r, c, self.game.player)
        except ValueError:
            done = True
            reward = -10.0
            return self.get_state(), reward, done, {}

        if reward_fn is not None:
            # prev -> next でシェーピング
            reward = reward_fn(prev_game, self.game)

        done = self.game.is_terminal()
        self.player = self.game.player
        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.array(self.game.board, dtype=np.float32)

    def legal_actions(self):
        moves = self.game.legal_moves(self.player)
        if not moves:
            return [64]
        return [r * 8 + c for r, c in moves]

    def render(self):
        board_np = np.array(self.game.board)
        print(board_np)

# %% [markdown]
# ## DQNのネットワーク定義（入力2ch: 盤面 + 手番）

# %%
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=3, padding=1, groups=in_channels, bias=False,
        )
        self.pw = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, bias=False,
        )
        self.gn = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=2,  # ← 2ch（盤面+手番）
                out_channels=8,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.SiLU(),
        )
        self.block1 = DSConv(in_channels=8, out_channels=8)
        self.block2 = DSConv(in_channels=8, out_channels=8)

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8, out_features=64),
            nn.SiLU(),
        )
        self.head = nn.Linear(in_features=64, out_features=65)  # 65番目はパス

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = self.fc(x)
        x = self.head(x)
        return x

# %%
# アーキテクチャのテスト
dqn = DQN()
dummy_board = torch.zeros((1, 1, 8, 8))
dummy_player = torch.ones((1, 1, 8, 8))  # 手番 +1
dummy_input = torch.cat([dummy_board, dummy_player], dim=1)  # (1,2,8,8)
print(dqn(dummy_input).shape)
torchinfo.summary(dqn, (1, 2, 8, 8))

# %% [markdown]
# ## DQNの学習

# %%
class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# %%
class TrainDoubleDQN:
    """
    Double DQN 学習（2ch入力 & シェーピング報酬）
    """
    def __init__(
        self,
        dqn,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        init_memory_size: int = 5000,
        memory_size: int = 50000,
        target_update_freq: int = 1000,
        tau: float = 0.0,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 120,
        train_freq: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 30000,
        seed: int = 42,
        device: Optional[torch.device] = None,
        ReplayBufferCls=None,
        shaping_eta: float = 0.1,
    ):
        assert ReplayBufferCls is not None, "ReplayBufferCls を渡してください"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self.dqn = dqn.to(self.device)
        self.target_dqn = copy.deepcopy(dqn).to(self.device)
        self.target_dqn.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = gamma
        self.batch_size = batch_size
        self.init_memory_size = init_memory_size
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes = num_episodes
        self.train_freq = max(1, int(train_freq))
        self.gradient_steps = max(1, int(gradient_steps))
        self.learning_starts = int(learning_starts)

        self.tau = float(tau)
        self.target_update_freq = int(target_update_freq)
        self._num_updates = 0
        self._num_env_steps = 0

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))

        self.replay_buffer = ReplayBufferCls(memory_size)

        self.shaping_eta = float(shaping_eta)

        self._init_replay_buffer()

        self.rewards: List[float] = []
        self.losses: List[float] = []

    def train(self) -> Dict[str, List[float]]:
        pbar = tqdm(total=self.num_episodes, desc="Train Double DQN")
        for ep in range(self.num_episodes):
            ep_reward = self._run_episode(ep)
            self.rewards.append(ep_reward)
            last_loss = self.losses[-1] if self.losses else float("nan")
            pbar.set_postfix_str(
                f"EpR: {ep_reward:.2f}  Loss: {last_loss:.3f}  ε: {self._epsilon_by_step(self._num_env_steps):.3f}"
            )
            pbar.update(1)
        pbar.close()
        return {"rewards": self.rewards, "losses": self.losses}

    def _run_episode(self, episode_idx: int) -> float:
        env = OthelloEnv()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < self.max_steps_per_episode:
            epsilon = self._epsilon_by_step(self._num_env_steps)
            player = env.player
            state = env.get_state()
            action = self._select_action_by_epsilon_greedy(env, epsilon)

            # 1 ステップ進める：ShapedReward は prev/next を使う
            shaped = ShapedReward(player, eta=self.shaping_eta, gamma=self.gamma)
            next_state, reward, done, _ = env.step(action, reward_fn=shaped.get_reward)
            next_player = env.player

            next_legal_actions = self._legal_actions_from_board(next_state, next_player)
            self._store_transition(
                board=state,
                action=action,
                reward=reward,
                next_board=next_state,
                done=done,
                player=player,
                next_player=next_player,
                next_legal_actions=next_legal_actions,
            )

            if (self._num_env_steps % self.train_freq == 0) and \
               (len(self.replay_buffer) >= max(self.batch_size, self.learning_starts)):
                for _ in range(self.gradient_steps):
                    loss = self._update_dqn_double()
                    if not math.isnan(loss):
                        self.losses.append(loss)

            self._maybe_update_target()

            total_reward += float(reward)
            steps += 1
            self._num_env_steps += 1

        return float(total_reward)

    def _update_dqn_double(self) -> float:
        batch = self.replay_buffer.sample(self.batch_size)

        # (B,2,8,8) に整形（盤面+手番）
        board = torch.stack([self._to_input(b['board'], b['player']) for b in batch]).to(self.device)
        next_board = torch.stack([self._to_input(b['next_board'], b['next_player']) for b in batch]).to(self.device)

        action = torch.tensor([b['action'] for b in batch], dtype=torch.int64, device=self.device)
        reward = torch.tensor([b['reward'] for b in batch], dtype=torch.float32, device=self.device)
        done = torch.tensor([b['done'] for b in batch], dtype=torch.float32, device=self.device)
        next_legal_actions_list = [b['next_legal_actions'] for b in batch]

        # Q(s,a)
        q_all = self.dqn(board)                                 # (B,65)
        q_sa = q_all.gather(1, action.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            # マスク（-1e9 で安定化）
            next_masks = self._build_masks_from_indices(next_legal_actions_list, fill_value=-1e9)
            q_next_online = self.dqn(next_board) + next_masks
            next_actions_online = q_next_online.argmax(dim=1)  # (B,)

            q_next_target = self.target_dqn(next_board)
            next_q = q_next_target.gather(1, next_actions_online.unsqueeze(1)).squeeze(1)

            target = reward + self.gamma * next_q * (1.0 - done)

        loss = self.loss_fn(q_sa, target)
        if torch.isnan(loss):
            return float("nan")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._num_updates += 1
        return float(loss.item())

    def _select_action_by_epsilon_greedy(self, env: OthelloEnv, epsilon: float) -> int:
        legal_actions = env.legal_actions()
        if random.random() < epsilon:
            return random.choice(legal_actions)
        return self._select_action_by_greedy(env)

    def _select_action_by_greedy(self, env: OthelloEnv) -> int:
        legal_actions = env.legal_actions()
        board_tensor = self._to_input(env.get_state(), env.player).unsqueeze(0).to(self.device)  # (1,2,8,8)
        with torch.no_grad():
            q_all = self.dqn(board_tensor).squeeze(0)  # (65,)
            mask = torch.full((65,), -1e9, device=self.device)
            for a in legal_actions:
                mask[a] = 0.0
            q_masked = q_all + mask
            action = int(q_masked.argmax().item())
        return action

    def _init_replay_buffer(self):
        target = min(self.init_memory_size, self.replay_buffer.memory_size)
        added = 0
        pbar = tqdm(total=target, desc='Init replay buffer')
        while added < target:
            env = OthelloEnv()
            done = False
            while not done and added < target:
                player = env.player
                state = env.get_state()
                legal_actions = env.legal_actions()
                action = random.choice(legal_actions)
                shaped = ShapedReward(player, eta=self.shaping_eta, gamma=self.gamma)
                next_state, reward, done, _ = env.step(action, reward_fn=shaped.get_reward)
                next_player = env.player
                next_legal_actions = self._legal_actions_from_board(next_state, next_player)

                self._store_transition(
                    board=state,
                    action=action,
                    reward=reward,
                    next_board=next_state,
                    done=done,
                    player=player,
                    next_player=next_player,
                    next_legal_actions=next_legal_actions,
                )
                added += 1
                pbar.update(1)
        pbar.close()

    def _maybe_update_target(self):
        if self.tau and self.tau > 0.0:
            with torch.no_grad():
                for tp, p in zip(self.target_dqn.parameters(), self.dqn.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        else:
            if (self._num_updates % max(1, self.target_update_freq)) == 0 and self._num_updates > 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())

    def _epsilon_by_step(self, step: int) -> float:
        if step >= self.epsilon_decay_steps:
            return self.epsilon_end
        span = self.epsilon_start - self.epsilon_end
        return self.epsilon_start - span * (step / self.epsilon_decay_steps)

    def _build_masks_from_indices(self, batch_next_legal_actions, fill_value: float = -1e9):
        B = len(batch_next_legal_actions)
        masks = torch.full((B, 65), float(fill_value), device=self.device)
        for i, acts in enumerate(batch_next_legal_actions):
            for a in acts:
                masks[i, a] = 0.0
        return masks

    # --- 入力整形: (2,8,8) = [盤面, 手番プレーン] ---
    def _to_input(self, board_like, player_scalar: int) -> torch.Tensor:
        t = torch.as_tensor(board_like, dtype=torch.float32)
        if t.dim() == 2 and t.shape == (8,8):
            t = t.unsqueeze(0)  # (1,8,8)
        elif t.dim() == 3 and t.shape == (1,8,8):
            pass
        else:
            t = t.reshape(1,8,8)
        player_plane = torch.full_like(t, float(player_scalar))  # (1,8,8) 全 +1/-1
        return torch.cat([t, player_plane], dim=0)  # (2,8,8)

    def _legal_actions_from_board(self, board_np: np.ndarray, player: int) -> List[int]:
        g = OthelloGame()
        g.board = board_np.astype(np.int8).tolist()
        g.player = player
        moves = g.legal_moves(player)
        if not moves:
            return [64]
        return [r * 8 + c for r, c in moves]

    def _store_transition(
        self,
        board,
        action: int,
        reward: float,
        next_board,
        done: bool,
        player: int,
        next_player: int,
        next_legal_actions: List[int],
    ):
        transition = {
            "board": np.array(board, dtype=np.float32),      # 生盤面を保持
            "action": int(action),
            "reward": float(reward),
            "next_board": np.array(next_board, dtype=np.float32),
            "done": bool(done),
            "player": int(player),
            "next_player": int(next_player),
            "next_legal_actions": next_legal_actions,
        }
        self.replay_buffer.append(transition)

# %%
# デバイスの設定
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

num_episodes = 10000
gamma = 0.99
lr = 5e-4
target_update_freq = 2000
batch_size = 128

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 20000

# ラン可能な最小実行ブロック（必要ならコメントアウト解除して使用）
if __name__ == "__main__":
    dqn = DQN()
    trainer = TrainDoubleDQN(
        dqn,
        device=device,
        num_episodes=num_episodes,
        gamma=gamma,
        lr=lr,
        target_update_freq=target_update_freq,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        ReplayBufferCls=ReplayBuffer,
        max_steps_per_episode=120,  # 64〜150程度で調整可
        shaping_eta=0.1,            # 0.05〜0.2の範囲で調整
    )

    res = trainer.train()
    rewards = res["rewards"]

    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
