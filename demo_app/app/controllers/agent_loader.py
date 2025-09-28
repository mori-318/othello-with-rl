"""デモ用エージェントを生成するローダーモジュール。"""

from __future__ import annotations

from typing import Dict, Type

from app.agents.base_agent import BaseAgent
from app.agents.random_agent import RandomAgent


def available_agents() -> Dict[str, Type[BaseAgent]]:
    """利用可能なエージェントクラスを取得する。

    Returns:
        Dict[str, Type[BaseAgent]]: 名前とエージェントクラスのマッピング。
    """
    # ダミーとしてランダムエージェントのみ登録
    return {
        "random": RandomAgent,
        "dqn": RandomAgent,
        "bc": RandomAgent,
        "mcts_nn": RandomAgent,
    }


def create_agent(name: str) -> BaseAgent:
    """エージェントインスタンスを生成する。

    Args:
        name (str): エージェント名。

    Returns:
        BaseAgent: 生成されたエージェント。

    Raises:
        ValueError: 未対応のエージェント名が指定された場合。
    """
    # 利用可能エージェント辞書を取得
    agents = available_agents()
    if name not in agents:
        raise ValueError(f"Unsupported agent: {name}")
    # エージェントを生成
    agent_class = agents[name]
    return agent_class(name=name)
