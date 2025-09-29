import os
import math
import copy
import random
import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ローカル環境/報酬
from utils.mori.othello_env import OthelloEnv
from utils.mori.othello_game import OthelloGame
from utils.mori.othello_reward import OthelloReward


# -----------------------------
# DQN モデル
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch: int, bn_eps: float = 1e-5, zero_init: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch, eps=bn_eps)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch, eps=bn_eps)
        if zero_init:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class DQN(nn.Module):
    def __init__(
        self,
        in_ch: int = 2,
        width: int = 32,
        num_res_blocks: int = 3,
        bn_eps: float = 1e-5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.GroupNorm(1, width, eps=bn_eps),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(width, bn_eps=bn_eps, zero_init=False) for _ in range(num_res_blocks)]
        )
        # 65アクション（0..63: 盤上, 64: パス）
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=2, kernel_size=1, bias=False),  # (B,2,8,8)
            nn.ReLU(inplace=True),
        )
        self.logits_fc = nn.Linear(2 * 8 * 8, 65)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.res_blocks(h)
        h = self.policy_head(h)
        h = h.view(h.size(0), -1)
        logits = self.logits_fc(h)  # (B,65)
        return logits


# -----------------------------
# リプレイバッファ
# -----------------------------
class ReplayBuffer:
    def __init__(self, memory_size: int):
        self.memory_size = int(memory_size)
        self.memory = deque(maxlen=self.memory_size)

    def append(self, transition: Dict):
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# -----------------------------
# シェーピング報酬
# -----------------------------
class ShapedReward:
    """
    OthelloReward（終局報酬）に、ボードポテンシャル差分の形状化を加える。

    phi(board, player) = (sum(board) * player) / 64.0
    r = base_terminal_reward + eta * (phi(next) - phi(prev))
    """
    def __init__(self, perspective: int, eta: float = 0.1):
        self.perspective = int(perspective)  # 遷移前状態の手番視点
        self.eta = float(eta)
        self.base = OthelloReward(player=self.perspective)

    def __call__(self, prev_env: OthelloGame, next_env: OthelloGame) -> float:
        # ベース（終局時のみ非ゼロ）
        base_r = self.base(prev_env, next_env)
        # ポテンシャル差分
        prev_phi = float(prev_env.board.sum()) * self.perspective / 64.0
        next_phi = float(next_env.board.sum()) * self.perspective / 64.0
        shaping = self.eta * (next_phi - prev_phi)
        return float(base_r + shaping)


# -----------------------------
# Double DQN 学習器
# -----------------------------
class TrainDoubleDQN:
    def __init__(
        self,
        dqn: nn.Module,
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
        ReplayBufferCls=ReplayBuffer,
        shaping_eta: float = 0.1,
        rolling_window: int = 100,
        save_best_path: Optional[str] = None,
    ):
        assert ReplayBufferCls is not None, "ReplayBufferCls を渡してください"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # 再現性
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ネットワーク
        self.dqn = dqn.to(self.device)
        self.target_dqn = copy.deepcopy(dqn).to(self.device)
        self.target_dqn.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # HP
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.init_memory_size = int(init_memory_size)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.num_episodes = int(num_episodes)
        self.train_freq = max(1, int(train_freq))
        self.gradient_steps = max(1, int(gradient_steps))
        self.learning_starts = int(learning_starts)

        self.tau = float(tau)
        self.target_update_freq = int(target_update_freq)
        self._num_updates = 0
        self._num_env_steps = 0

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))

        self.replay_buffer = ReplayBufferCls(int(memory_size))
        self.shaping_eta = float(shaping_eta)

        self.rewards: List[float] = []
        self.losses: List[float] = []
        self.rolling_window = int(rolling_window)
        self.best_score = -float("inf")
        self.best_state_dict = copy.deepcopy(self.dqn.state_dict())
        self.save_best_path = save_best_path

        # 初期リプレイ収集
        self._init_replay_buffer()

    # --------- 公開 API ---------
    def train(self, return_best_model: bool = False):
        pbar = tqdm(total=self.num_episodes, desc="Train Double DQN")
        for ep in range(self.num_episodes):
            ep_reward = self._run_episode(ep)
            self.rewards.append(ep_reward)

            # ローリング平均
            if len(self.rewards) >= self.rolling_window:
                rolling_avg = float(np.mean(self.rewards[-self.rolling_window:]))
            else:
                rolling_avg = float(np.mean(self.rewards))

            # ベスト更新
            if rolling_avg > self.best_score:
                self.best_score = rolling_avg
                self.best_state_dict = copy.deepcopy(self.dqn.state_dict())
                if self.save_best_path is not None:
                    torch.save(self.best_state_dict, self.save_best_path)

            last_loss = self.losses[-1] if self.losses else float("nan")
            pbar.set_postfix_str(
                f"EpR: {ep_reward:.2f} Roll@{self.rolling_window}: {rolling_avg:.2f} Best: {self.best_score:.2f} "
                f"Loss: {last_loss:.3f} ε: {self._epsilon_by_step(self._num_env_steps):.3f}"
            )
            pbar.update(1)
        pbar.close()

        metrics = {"rewards": self.rewards, "losses": self.losses}
        if return_best_model:
            best_model = self._clone_model_with_state(self.best_state_dict)
            return metrics, best_model
        return metrics

    def get_best_model(self) -> nn.Module:
        return self._clone_model_with_state(self.best_state_dict)

    # --------- 内部処理 ---------
    def _clone_model_with_state(self, state_dict) -> nn.Module:
        model = copy.deepcopy(self.dqn).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _run_episode(self, episode_idx: int) -> float:
        env = OthelloEnv()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < self.max_steps_per_episode:
            # ε-greedy 選択
            epsilon = self._epsilon_by_step(self._num_env_steps)
            player = env.player
            state = env.get_state()
            action = self._select_action_by_epsilon_greedy(env, epsilon)

            # シェーピング付き報酬
            reward_fn = ShapedReward(player, eta=self.shaping_eta)
            next_state, reward, done, _ = env.step(action, reward_fn=reward_fn)
            next_player = env.player

            next_legal_actions = self._legal_actions_from_board(next_state[0], next_player)
            self._store_transition(
                board=state[0],
                action=action,
                reward=reward,
                next_board=next_state[0],
                done=done,
                player=player,
                next_player=next_player,
                next_legal_actions=next_legal_actions,
            )

            # 学習トリガ
            if (self._num_env_steps % self.train_freq == 0) and (len(self.replay_buffer) >= max(self.batch_size, self.learning_starts)):
                for _ in range(self.gradient_steps):
                    loss = self._update_dqn_double()
                    if not math.isnan(loss):
                        self.losses.append(loss)

            # ターゲット更新
            self._maybe_update_target()

            total_reward += float(reward)
            steps += 1
            self._num_env_steps += 1

        return float(total_reward)

    def _update_dqn_double(self) -> float:
        batch = self.replay_buffer.sample(self.batch_size)

        # (B,2,8,8)
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
            # 非合法手マスク
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
                reward_fn = ShapedReward(player, eta=self.shaping_eta)
                next_state, reward, done, _ = env.step(action, reward_fn=reward_fn)
                next_player = env.player
                next_legal_actions = self._legal_actions_from_board(next_state[0], next_player)

                self._store_transition(
                    board=state[0],
                    action=action,
                    reward=reward,
                    next_board=next_state[0],
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

    def _build_masks_from_indices(self, batch_next_legal_actions: List[List[int]], fill_value: float = -1e9):
        B = len(batch_next_legal_actions)
        masks = torch.full((B, 65), float(fill_value), device=self.device)
        for i, acts in enumerate(batch_next_legal_actions):
            for a in acts:
                masks[i, a] = 0.0
        return masks

    def _to_input(self, board_like, player_scalar: int) -> torch.Tensor:
        t = torch.as_tensor(board_like, dtype=torch.float32)
        if t.dim() == 3 and t.shape == (2, 8, 8):
            t = t[0:1]
        elif t.dim() == 2 and t.shape == (8, 8):
            t = t.unsqueeze(0)
        elif t.dim() == 3 and t.shape == (1, 8, 8):
            pass
        else:
            t = t.reshape(1, 8, 8)
        player_plane = torch.full_like(t, float(player_scalar))
        return torch.cat([t, player_plane], dim=0)

    def _legal_actions_from_board(self, board_np: np.ndarray, player: int) -> List[int]:
        # OthelloGame を新規に作成し、numpy 配列を設定
        g = OthelloGame()
        # board_np は float32（-1,0,1）なので int8 に変換
        if board_np.ndim == 3 and board_np.shape == (2, 8, 8):
            board_np = board_np[0]
        g.board = np.asarray(board_np, dtype=np.int8)
        g.player = int(player)
        moves = g.legal_moves(g.player)
        return [64] if not moves else [r * 8 + c for r, c in moves]

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
        legal_actions = self._legal_actions_from_board(np.array(board, dtype=np.float32), int(player))
        transition = {
            "board": np.array(board, dtype=np.float32),
            "action": int(action),
            "reward": float(reward),
            "next_board": np.array(next_board, dtype=np.float32),
            "done": bool(done),
            "player": int(player),
            "next_player": int(next_player),
            "legal_actions": legal_actions,
            "next_legal_actions": next_legal_actions,
        }
        self.replay_buffer.append(transition)


# -----------------------------
# 便利関数: 軽量設定での学習実行
# -----------------------------

def train_quick(
    episodes: int = 50,
    init_memory: int = 1000,
    batch_size: int = 128,
    save_dir: str = "model_weights/ddqn",
    device: Optional[str] = None,
):
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dqn = DQN().to(dev)
    trainer = TrainDoubleDQN(
        dqn=dqn,
        device=dev,
        ReplayBufferCls=ReplayBuffer,
        num_episodes=episodes,
        batch_size=batch_size,
        gamma=0.99,
        lr=1e-3,
        target_update_freq=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
        max_steps_per_episode=120,
        shaping_eta=0.1,
        init_memory_size=init_memory,
        memory_size=max(init_memory, 5000),
        learning_starts=min(init_memory, 1000),
        rolling_window=20,
        save_best_path=None,
    )

    metrics, best_model = trainer.train(return_best_model=True)

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"best_{ts}.pth")
    torch.save(best_model.state_dict(), save_path)
    return metrics, save_path, trainer


if __name__ == "__main__":
    metrics, save_path, trainer = train_quick()
    print(f"Best model saved to {save_path}")
