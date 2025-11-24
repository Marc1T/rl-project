# components/normalizers.py

import numpy as np
from typing import Dict
from config.environment_configs import PDPEnvironmentConfig

class ObservationNormalizer:
    """Normalise les observations pour la stabilité de l'entraînement"""
    
    def __init__(self, config: PDPEnvironmentConfig):
        self.config = config
        self._compute_normalization_factors()
    
    def _compute_normalization_factors(self):
        """Calcule les facteurs de normalisation"""
        # Pour le stock: on considère qu'il peut être négatif (backorders)
        # On normalise par rapport à la plage [-max_stock/2, max_stock]
        self.stock_range = np.array(self.config.max_stock, dtype=np.float32) * 1.5
        self.stock_offset = np.array(self.config.max_stock, dtype=np.float32) * 0.5
        
        # Pour les demandes: basé sur la capacité maximale
        self.demand_factor = np.max(self.config.regular_capacity) * 2.0
        
    def normalize(self, observation: Dict) -> Dict:
        """
        Normalise une observation pour obtenir des valeurs dans [-1, 1] ou [0, 1]
        """
        normalized = {}
        
        for key, value in observation.items():
            if key == 'current_stock':
                # Normalisation du stock: [-max_stock/2, max_stock] -> [-1, 1]
                normalized[key] = (value + self.stock_offset) / self.stock_range * 2.0 - 1.0
                # Clip pour éviter les valeurs hors limites
                normalized[key] = np.clip(normalized[key], -1.0, 1.0)
                
            elif key == 'future_demands':
                # Normalisation des demandes: [0, inf] -> [0, 1]
                normalized[key] = np.clip(value / self.demand_factor, 0.0, 1.0)
                
            elif key == 'current_period':
                # Normalisation de la période: [0, horizon] -> [0, 1]
                normalized[key] = value / self.config.horizon
                
            else:
                # Autres champs: copie directe
                normalized[key] = value
                
        return normalized
    
    def denormalize_stock(self, normalized_stock: np.ndarray) -> np.ndarray:
        """Dénormalise le stock (utile pour le debugging)"""
        return (normalized_stock + 1.0) / 2.0 * self.stock_range - self.stock_offset
