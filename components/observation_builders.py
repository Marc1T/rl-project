# components/observation_builders.py

import numpy as np
from typing import Dict
from config.environment_configs import PDPEnvironmentConfig

class ObservationBuilder:
    """Construit les observations à partir de l'état courant"""
    
    def __init__(self, config: PDPEnvironmentConfig):
        self.config = config
    
    def build(self, current_stock: np.ndarray, current_period: int, demands: np.ndarray) -> Dict:
        """Construit l'observation complète"""
        observation = {
            'current_stock': current_stock.copy(),
            'future_demands': self._get_future_demands(current_period, demands),
        }
        
        if self.config.include_period_info:
            observation['current_period'] = np.array([current_period], dtype=np.float32)
            
        return observation
    
    def _get_future_demands(self, current_period: int, demands: np.ndarray) -> np.ndarray:
        """Extrait les demandes futures"""
        lookahead = self.config.demand_lookahead
        end_period = min(current_period + lookahead, self.config.horizon)
        
        future_demands = demands[:, current_period:end_period]
        
        # Complète avec des zéros si nécessaire
        if future_demands.shape[1] < lookahead:
            padding = np.zeros((self.config.n_products, lookahead - future_demands.shape[1]))
            future_demands = np.hstack([future_demands, padding])
            
        return future_demands.astype(np.float32)
