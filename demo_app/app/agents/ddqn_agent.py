from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.agents.base_agent import BaseAgent
from app.game.othello_env import OthelloEnv


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


class DQN(nn.Module):
    def __init__(
        self,
        in_ch: int = 2,
        width: int = 32,
        num_res_blocks: int = 3,
        bn_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.GroupNorm(1, width, eps=bn_eps),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(width, bn_eps=bn_eps, zero_init=False) for _ in range(num_res_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(width, 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.logits_fc = nn.Linear(2 * 8 * 8, 65)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.res_blocks(h)
        h = self.policy_head(h)
        h = h.view(h.size(0), -1)
        return self.logits_fc(h)


DeviceLike = Union[str, torch.device]


class DDQNAgent(BaseAgent):
    """学習済みDouble DQNを用いた推論エージェント。"""

    def __init__(
        self,
        name: str = "ddqn",
        model_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        device: Optional[DeviceLike] = None,
    ) -> None:
        super().__init__(name=name)

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        else:
            self.device = torch.device(device)

        self.model = DQN().to(self.device)
        self.model.eval()

        ckpt_path = self._resolve_checkpoint_path(model_path=model_path, model_dir=model_dir)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        for param in self.model.parameters():
            param.requires_grad_(False)

        self._model_path = str(ckpt_path)

    def select_action(self, env: OthelloEnv) -> int:
        legal_actions = env.legal_actions()
        state = env.get_state()  # (2,8,8)
        board = torch.from_numpy(state).to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(board).squeeze(0)  # (65,)
        mask = torch.full((65,), -1e9, device=self.device)
        for a in legal_actions:
            mask[a] = 0.0
        action = int(torch.argmax(q_values + mask).item())
        return action

    def _resolve_checkpoint_path(
        self, *, model_path: Optional[str], model_dir: Optional[str]
    ) -> Path:
        if model_path is not None:
            ckpt_path = Path(model_path).expanduser().resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"指定されたDDQNモデルが見つかりません: {ckpt_path}")
            return ckpt_path

        search_dir = (
            Path(model_dir).expanduser().resolve()
            if model_dir is not None
            else Path(__file__).resolve().parent / "model_weights"
        )

        if not search_dir.exists():
            raise FileNotFoundError(
                "DDQNモデルディレクトリが存在しません。"
                f" 以下に学習済みモデル(.pth)を配置してください: {search_dir}"
            )

        ckpt = find_latest_checkpoint(search_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "DDQNモデルが見つかりません。"
                f" 以下のディレクトリに学習済みモデル(.pth)を配置してください: {search_dir}"
            )
        return ckpt

def find_latest_checkpoint(search_dir: Path) -> Optional[Path]:
    """指定ディレクトリ内で最新の重みファイルを返す。"""
    checkpoints = sorted(search_dir.glob("*.pth"))
    return checkpoints[-1] if checkpoints else None

def create_ddqn_agent(
    name: str = "ddqn",
    model_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[DeviceLike] = None,
) -> DDQNAgent:
    if model_path is None:
        model_dir = Path(model_dir).expanduser().resolve() if model_dir is not None else Path(__file__).resolve().parent / "model_weights"
        ckpt = find_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f"DDQNモデルが見つかりません。以下のディレクトリに学習済みモデル(.pth)を配置してください: {model_dir}"
            )
        model_path = str(ckpt)
    return DDQNAgent(
        name=name,
        model_path=model_path,
        device=device,
    )