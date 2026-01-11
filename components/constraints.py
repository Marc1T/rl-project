# components/constraints.py

import numpy as np
from typing import Dict, List
from config.base_config import BaseConfig

class ConstraintsManager:
    """
    Gère les règles métier et les contraintes de capacité qui affectent
    la production et l'état de l'environnement.
    
    Les contraintes sont maintenant CONFIGURABLES via BaseConfig:
    - reduced_capacity_periods: Dict des périodes avec capacité réduite
    - no_overtime_periods: Liste des périodes sans heures supplémentaires
    - no_overtime_last_period: Bool pour désactiver overtime à la dernière période
    """
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.regular_capacity = np.array(config.regular_capacity)
        self.overtime_capacity = np.array(config.overtime_capacity)
        self.subcontracting_capacity = np.array(config.subcontracting_capacity)
        
        # Charger les contraintes depuis la config (avec valeurs par défaut)
        self.reduced_capacity_periods = getattr(config, 'reduced_capacity_periods', {6: 0.5})
        self.no_overtime_periods = getattr(config, 'no_overtime_periods', [])
        self.no_overtime_last_period = getattr(config, 'no_overtime_last_period', True)

    def get_available_capacity(self, period: int) -> Dict[str, np.ndarray]:
        """
        Retourne la capacité disponible pour la période donnée, en appliquant
        les contraintes spécifiques à la période depuis la configuration.
        """
        
        # Capacités de base
        regular_cap = self.regular_capacity.copy()
        overtime_cap = self.overtime_capacity.copy()
        subcontracting_cap = self.subcontracting_capacity.copy()
        
        # Contrainte: Capacité réduite pour certaines périodes (configurable)
        if period in self.reduced_capacity_periods:
            reduction_ratio = self.reduced_capacity_periods[period]
            regular_cap = regular_cap * reduction_ratio
            overtime_cap = overtime_cap * reduction_ratio
        
        # Contrainte: Heures supplémentaires interdites pour certaines périodes
        if period in self.no_overtime_periods:
            overtime_cap = np.zeros_like(overtime_cap)
        
        # Contrainte: Heures supplémentaires interdites à la dernière période
        if self.no_overtime_last_period and period == self.config.horizon - 1:
            overtime_cap = np.zeros_like(overtime_cap)
            
        return {
            'regular': regular_cap,
            'overtime': overtime_cap,
            'subcontracting': subcontracting_cap
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
