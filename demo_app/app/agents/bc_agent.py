from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.agents.base_agent import BaseAgent
from app.game.othello_env import OthelloEnv


logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, ch: int, bn_eps: float = 1e-5, zero_init: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch, eps=bn_eps)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch, eps=bn_eps)
        if zero_init:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class CriticNet(nn.Module):
    """`train_dqn.ipynb` で使用した価値推定ネットワーク。"""

    def __init__(
        self,
        in_ch: int = 2,
        width: int = 32,
        num_res_blocks: int = 3,
        bn_eps: float = 1e-5,
        head_hidden_size: int = 32,
        use_gap: bool = True,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, width, eps=bn_eps),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(width, bn_eps=bn_eps, zero_init=True) for _ in range(num_res_blocks)]
        )

        self.value_conv = nn.Conv2d(width, 1, kernel_size=1, bias=False)
        self.value_norm = nn.LayerNorm((1, 8, 8))
        self.use_gap = use_gap
        self.value_fc1 = nn.Linear(1 if use_gap else 8 * 8, head_hidden_size)
        self.value_fc2 = nn.Linear(head_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.res_blocks(h)
        h = self.value_conv(h)
        h = self.value_norm(h)
        h = F.relu(h)
        if self.use_gap:
            h = h.mean(dim=(2, 3))
        else:
            h = h.view(h.size(0), -1)
        h = F.relu(self.value_fc1(h))
        return torch.tanh(self.value_fc2(h))


class BCAgent(BaseAgent):
    """事前学習済み CriticNet を用いた評価ベース行動エージェント。"""

    def __init__(
        self,
        name: str = "bc",
        model_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        )

        self.model = CriticNet().to(self.device)
        self.model.eval()

        ckpt_path = self._resolve_checkpoint(model_dir=model_dir, model_path=model_path)
        logger.info("Loading BC critic weights from %s", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "ロードした重みがCriticNetと互換性がありません。"
                " CriticNetで学習した重み(.pth)を指定してください。"
                f" 問題のファイル: {ckpt_path}"
            ) from exc

        for param in self.model.parameters():
            param.requires_grad_(False)

    def select_action(self, env: OthelloEnv) -> int:
        legal_actions = env.legal_actions()
        best_value = -float("inf")
        best_action = legal_actions[0]

        for action in legal_actions:
            value = self._simulate_and_evaluate(env, action)
            logger.debug("action=%d value=%.4f", action, value)
            if value > best_value:
                best_value = value
                best_action = action

        logger.info("Selected action %d with value %.4f", best_action, best_value)
        return best_action

    def _simulate_and_evaluate(self, env: OthelloEnv, action: int) -> float:
        sim_env = self._clone_env(env)
        try:
            sim_env.step(action)
        except Exception:
            return -float("inf")

        state = sim_env.get_state()
        tensor = torch.from_numpy(state).to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = self.model(tensor).item()
        return float(value)

    def _clone_env(self, env: OthelloEnv) -> OthelloEnv:
        clone = OthelloEnv()
        clone.game = env.game.clone()
        clone.player = env.player
        return clone

    def _resolve_checkpoint(
        self,
        *,
        model_dir: Optional[str],
        model_path: Optional[str],
    ) -> Path:
        if model_path is not None:
            path = Path(model_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"指定されたBCモデルが見つかりません: {path}")
            return path

        search_dir = (
            Path(model_dir).expanduser().resolve()
            if model_dir is not None
            else Path(__file__).resolve().parent / "model_weights"
        )
        if not search_dir.exists():
            raise FileNotFoundError(
                "BCモデルディレクトリが存在しません。"
                f" 以下に学習済みcritic(.pth)を配置してください: {search_dir}"
            )

        checkpoints = sorted(search_dir.glob("bc*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                "BCの学習済み重みが見つかりません。"
                f" 以下のディレクトリに bc*.pth を配置してください: {search_dir}"
            )
        return checkpoints[-1]


def configure_logging(level: int = logging.INFO) -> None:
    handler_exists = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not handler_exists:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

