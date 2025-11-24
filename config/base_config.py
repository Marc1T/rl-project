# config/base_config.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class BaseConfig:
    """Configuration de base pour tous les environnements PDP"""
    # Général
    env_id: str = "StrategicPDP-v1"
    n_products: int = 1
    horizon: int = 12
    seed: int = 42
    
    # Capacités de production
    regular_capacity: List[float] = field(default_factory=lambda: [100.0])
    overtime_capacity: List[float] = field(default_factory=lambda: [30.0])
    subcontracting_capacity: List[float] = field(default_factory=lambda: [50.0])
    
    # Coûts
    regular_cost: List[float] = field(default_factory=lambda: [10.0])
    overtime_cost: List[float] = field(default_factory=lambda: [15.0])
    subcontracting_cost: List[float] = field(default_factory=lambda: [20.0])
    holding_cost: List[float] = field(default_factory=lambda: [2.0])
    shortage_cost: List[float] = field(default_factory=lambda: [100.0])  # Élevé pour éviter les ruptures
    
    # État initial
    initial_stock: List[float] = field(default_factory=lambda: [50.0])
    min_stock: List[float] = field(default_factory=lambda: [0.0])
    max_stock: List[float] = field(default_factory=lambda: [500.0])
    
    # Stratégie
    service_level_target: float = 0.95
    max_overtime_ratio: float = 0.3
    max_subcontracting_ratio: float = 0.4
    
    def validate(self):
        """Valide la configuration"""
        assert self.n_products > 0, "n_products doit être > 0"
        assert len(self.regular_capacity) == self.n_products, "Tailles incohérentes"
