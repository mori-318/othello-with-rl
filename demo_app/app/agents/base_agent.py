"""エージェント実装の抽象基底クラスを提供するモジュール。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.game.othello_env import OthelloEnv


class BaseAgent(ABC):
    """強化学習エージェントの共通インターフェース。"""

    def __init__(self, name: str) -> None:
        """エージェントを初期化する。

        Args:
            name (str): エージェント名。
        """
        # エージェント名を保持
        self._name = name

    @property
    def name(self) -> str:
        """エージェント名を取得する。

        Returns:
            str: エージェント名。
        """
        return self._name

    @abstractmethod
    def select_action(self, env: OthelloEnv) -> int:
        """とるべき行動を決定する。

        Args:
            env (OthelloEnv): 現在のゲーム環境。

        Returns:
            int: 選択した行動のインデックス。0-63の盤面位置または64(パス)。
        """
        raise NotImplementedError
