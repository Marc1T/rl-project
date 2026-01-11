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
    
    # Reward shaping - POIDS NORMALISÉS (somme = 1 pour les coûts, bonus séparé)
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'production_cost': 0.25,     # 25% du poids des coûts
        'inventory_cost': 0.25,      # 25% du poids des coûts  
        'shortage_cost': 0.50,       # 50% du poids des coûts (critique)
        'service_bonus': 0.5         # Bonus séparé (0-0.5 range)
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
    demand_intensity: str = "high"  # "low", "medium", "high", "extreme"
