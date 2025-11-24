# config/environment_configs.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .base_config import BaseConfig

@dataclass
class PDPEnvironmentConfig(BaseConfig):
    """Configuration spécifique à l'environnement"""
    # Normalisation
    normalize_observations: bool = True
    normalize_rewards: bool = True  # Géré par VecNormalize
    clip_actions: bool = True
    
    # Reward shaping - POIDS RÉVISÉS pour un meilleur équilibre
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'production_cost': 0.4,      # Augmenté (coût principal)
        'inventory_cost': 0.3,       # Augmenté (important)
        'shortage_cost': 0.5,        # Augmenté (critique)
        'service_bonus': 1.0         # Augmenté (objectif principal)
    })
    
    # Observation
    demand_lookahead: int = 6
    include_capacity_info: bool = True
    include_period_info: bool = True
    
    # Contraintes
    allow_backorders: bool = True
    max_backorder_periods: int = 3
    
    # Génération de demande
    demand_file_path: Optional[str] = None
