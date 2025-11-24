# config/training_configs.py
 
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class PPOTrainingConfig:
    """Configuration pour l'entraînement PPO - HYPERPARAMÈTRES OPTIMISÉS"""
    # Hyperparamètres PPO - AJUSTÉS pour meilleure convergence
    learning_rate: float = 3e-4
    n_steps: int = 2048              # Nombre de steps avant mise à jour
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99              # Facteur de discount
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01           # Augmenté pour plus d'exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # Gradient clipping
    
    # Architecture du réseau - PLUS PROFOND
    policy_arch: List[int] = field(default_factory=lambda: [256, 256, 128])
    
    # Entraînement
    total_timesteps: int = 100000
    log_interval: int = 10
    save_interval: int = 10000
    
    # Chemins
    model_save_path: str = "./models/"
    tensorboard_log_path: str = "./logs/tensorboard/"