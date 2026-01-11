# agents/__init__.py
"""
Module agents - Strategies et entrainement RL pour le PDP

Ce module contient:
- Strategies Baseline (Level, Chase, Fixed Moderate)
- Agents Tabulaires (Monte Carlo, Q-Learning, SARSA)
- Agents Deep RL (DQN, A2C, SAC)
- Integration Stable-Baselines3 (PPO)
- Registre centralise des agents
"""

# ===== Strategies Baseline =====
from .baseline_strategies import (
    BaselineStrategy,
    LevelStrategy,
    ChaseStrategy,
    FixedModerateStrategy,
    BASELINE_STRATEGIES
)

# ===== Agents Tabulaires =====
from .tabular_agents import (
    ActionDiscretizer,
    StateDiscretizer,
    BaseTabularAgent,
    MonteCarloAgent,
    QLearningAgent,
    SARSAAgent,
    TABULAR_AGENTS
)

# ===== Agents Deep RL =====
from .dqn_agent import (
    ReplayBuffer,
    QNetwork,
    DQNAgent
)

from .a2c_agent import (
    ActorCriticNetwork,
    A2CAgent
)

from .sac_agent import (
    GaussianPolicy,
    TwinQNetwork,
    SACAgent
)

# ===== PPO (Stable-Baselines3) =====
from .ppo_trainer import PPOTrainer

# ===== Utilities =====
from .rl_utils import (
    SaveOnBestTrainingRewardCallback,
    EarlyStoppingCallback
)

# ===== Registre des agents =====
from .agent_registry import (
    AgentConfig,
    AgentRegistry,
    register_all_agents
)


__all__ = [
    # Strategies Baseline
    'BaselineStrategy',
    'LevelStrategy',
    'ChaseStrategy',
    'FixedModerateStrategy',
    'BASELINE_STRATEGIES',
    
    # Agents Tabulaires
    'ActionDiscretizer',
    'StateDiscretizer',
    'BaseTabularAgent',
    'MonteCarloAgent',
    'QLearningAgent',
    'SARSAAgent',
    'TABULAR_AGENTS',
    
    # Deep RL
    'ReplayBuffer',
    'QNetwork',
    'DQNAgent',
    'ActorCriticNetwork',
    'A2CAgent',
    'GaussianPolicy',
    'TwinQNetwork',
    'SACAgent',
    
    # PPO
    'PPOTrainer',
    
    # Utilities
    'SaveOnBestTrainingRewardCallback',
    'EarlyStoppingCallback',
    
    # Registry
    'AgentConfig',
    'AgentRegistry',
    'register_all_agents',
]
