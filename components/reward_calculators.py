# components/reward_calculators.py

import numpy as np
from typing import Dict
from config.environment_configs import PDPEnvironmentConfig
from .cost_calculators import CostCalculator

class RewardCalculator:
    """Calcule les rewards en utilisant CostCalculator et la pondération des rewards."""
    
    def __init__(self, config: PDPEnvironmentConfig):
        self.config = config
        self.cost_calculator = CostCalculator(config)
        
        # Facteur d'échelle pour normaliser les rewards
        # Basé sur les coûts typiques attendus
        max_capacity = np.sum(config.regular_capacity)
        self.cost_scale = max_capacity * np.mean(config.regular_cost)
        
    def compute(self, action: Dict, stock_before: np.ndarray, 
                current_stock: np.ndarray, current_demand: np.ndarray,
                current_period: int) -> float:
        """
        Calcule le reward complet.
        
        Args:
            action: Action prise (quantités réelles)
            stock_before: Stock avant production
            current_stock: Stock après production et demande
            current_demand: Demande de la période
            current_period: Période courante
        """
        
        # 1. Calcul des coûts
        production_cost = self.cost_calculator.compute_production_cost(action)
        inventory_cost = self.cost_calculator.compute_inventory_cost(current_stock)
        shortage_cost = self.cost_calculator.compute_shortage_cost(current_stock)
        
        # 2. Calcul du bonus de service (basé sur la demande satisfaite)
        service_bonus = self._compute_service_bonus(stock_before, action, current_demand)
        
        # 3. Reward pondéré et normalisé
        # On divise par cost_scale pour avoir des rewards dans une plage raisonnable
        total_cost = (
            production_cost * self.config.reward_weights['production_cost'] +
            inventory_cost * self.config.reward_weights['inventory_cost'] +
            shortage_cost * self.config.reward_weights['shortage_cost']
        )
        
        # Normalisation pour avoir des rewards de l'ordre de -1 à +1
        normalized_cost = total_cost / (self.cost_scale + 1e-6)
        
        # Le bonus est déjà normalisé (0-1), on le scale pour qu'il ait un impact similaire
        scaled_bonus = service_bonus * self.config.reward_weights['service_bonus'] * 2
        
        reward = -normalized_cost + scaled_bonus
        
        return float(reward)
    
    def _compute_service_bonus(self, stock_before: np.ndarray, 
                               action: Dict, 
                               demand: np.ndarray) -> float:
        """
        Calcule le bonus pour niveau de service.
        """
        
        # Production totale
        total_production = (
            action['regular'] +
            action['overtime'] +
            action['subcontracting']
        )
        
        # Stock disponible après production
        stock_available = stock_before + total_production
        
        # Demande satisfaite (ce qu'on peut livrer)
        demand_satisfied = np.minimum(np.maximum(stock_available, 0), demand)
        
        # Niveau de service
        total_demand = np.sum(demand)
        if total_demand > 0:
            service_level = np.sum(demand_satisfied) / total_demand
        else:
            service_level = 1.0
        
        # Bonus progressif en fonction du niveau de service
        if service_level >= self.config.service_level_target:
            # Excellent service
            return 1.0
        elif service_level >= 0.90:
            # Bon service
            return 0.7
        elif service_level >= 0.80:
            # Service acceptable
            return 0.4
        else:
            # Service insuffisant
            return 0.0
