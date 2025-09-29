"""デモ用エージェントを生成するローダーモジュール。"""

from __future__ import annotations

import os
from typing import Dict, Type

from app.agents.base_agent import BaseAgent
from app.agents.random_agent import RandomAgent
from app.agents.ddqn_agent import DDQNAgent, create_ddqn_agent


def available_agents() -> Dict[str, Type[BaseAgent]]:
    """利用可能なエージェントクラスを取得する。

    Returns:
        Dict[str, Type[BaseAgent]]: 名前とエージェントクラスのマッピング。
    """
    return {
        "random": RandomAgent,
        "dqn": RandomAgent,
        "mcts_nn": RandomAgent,
        "ddqn": DDQNAgent,
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
    agent_class = agents[name]

    if agent_class is DDQNAgent:
        model_dir = os.environ.get("DDQN_MODEL_DIR")
        model_path = os.environ.get("DDQN_MODEL_PATH")
        return create_ddqn_agent(name="ddqn", model_dir=model_dir, model_path=model_path)

    return agent_class(name=name)
