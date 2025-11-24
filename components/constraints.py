# components/constraints.py

import numpy as np
from typing import Dict, List
from config.base_config import BaseConfig

class ConstraintsManager:
    """
    Gère les règles métier et les contraintes de capacité qui affectent
    la production et l'état de l'environnement.
    """
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.regular_capacity = np.array(config.regular_capacity)
        self.overtime_capacity = np.array(config.overtime_capacity)
        self.subcontracting_capacity = np.array(config.subcontracting_capacity)

    def get_available_capacity(self, period: int) -> Dict[str, np.ndarray]:
        """
        Retourne la capacité disponible pour la période donnée, en appliquant
        les contraintes spécifiques à la période (ex: mois chômés).
        """
        
        # Exemple de contrainte: Capacité réduite de 50% au 7ème mois (vacances)
        if period == 6: # Période 6 (7ème mois)
            return {
                'regular': self.regular_capacity * 0.5,
                'overtime': self.overtime_capacity * 0.5,
                'subcontracting': self.subcontracting_capacity
            }
        
        # Exemple de contrainte: Heures supplémentaires interdites au dernier mois
        if period == self.config.horizon - 1:
            return {
                'regular': self.regular_capacity,
                'overtime': np.zeros_like(self.overtime_capacity),
                'subcontracting': self.subcontracting_capacity
            }
            
        return {
            'regular': self.regular_capacity,
            'overtime': self.overtime_capacity,
            'subcontracting': self.subcontracting_capacity
        }

    def validate_and_constrain_action(self, action: Dict[str, np.ndarray], period: int) -> Dict[str, np.ndarray]:
        """
        Applique les contraintes de capacité et les règles métier à l'action proposée.
        """
        
        available_capacity = self.get_available_capacity(period)
        constrained_action = {}
        
        # L'action est supposée être en quantités réelles (non normalisées)
        for key in ['regular', 'overtime', 'subcontracting']:
            # L'action ne doit pas dépasser la capacité disponible pour cette période
            constrained_action[key] = np.clip(
                action[key],
                0, 
                available_capacity[key]
            )
            
        return constrained_action
