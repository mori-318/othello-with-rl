"""ランダムエージェントの実装モジュール。"""

from __future__ import annotations

import random

from app.agents.base_agent import BaseAgent
from app.game.othello_env import OthelloEnv


class RandomAgent(BaseAgent):
    """ランダムに合法手を選択するエージェント。"""

    def __init__(self, name: str = "random") -> None:
        """ランダムエージェントを初期化する。

        Args:
            name (str): エージェント名。
        """
        super().__init__(name=name)

    def select_action(self, env: OthelloEnv) -> int:
        """合法手をランダムに選択する。

        Args:
            env (OthelloEnv): 現在のオセロ環境。

        Returns:
            int: 選択した行動インデックス。
        """
        legal_actions = env.legal_actions()
        return random.choice(legal_actions)
