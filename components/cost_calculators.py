# components/cost_calculators.py

import numpy as np
from typing import Dict, List
from config.base_config import BaseConfig

class CostCalculator:
    """
    Calcule les différents coûts associés à une décision de production et à l'état du stock.
    Utilisé par RewardCalculator et pour le reporting.
    """
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.regular_cost = np.array(config.regular_cost)
        self.overtime_cost = np.array(config.overtime_cost)
        self.subcontracting_cost = np.array(config.subcontracting_cost)
        self.holding_cost = np.array(config.holding_cost)
        self.shortage_cost = np.array(config.shortage_cost)

    def compute_production_cost(self, action: Dict[str, np.ndarray]) -> float:
        """Calcule le coût total de production pour l'étape courante."""
        regular_cost = np.sum(action['regular'] * self.regular_cost)
        overtime_cost = np.sum(action['overtime'] * self.overtime_cost)
        subcontracting_cost = np.sum(action['subcontracting'] * self.subcontracting_cost)
        
        return float(regular_cost + overtime_cost + subcontracting_cost)

    def compute_inventory_cost(self, current_stock: np.ndarray) -> float:
        """Calcule le coût de stockage (holding cost) pour le stock positif."""
        positive_stock = np.maximum(current_stock, 0)
        return float(np.sum(positive_stock * self.holding_cost))

    def compute_shortage_cost(self, current_stock: np.ndarray) -> float:
        """
        Calcule le coût de rupture (shortage cost) pour le stock négatif (backorder).
        Le stock négatif représente la quantité en rupture.
        """
        negative_stock = np.maximum(-current_stock, 0)
        return float(np.sum(negative_stock * self.shortage_cost))

    def compute_total_step_cost(self, action: Dict[str, np.ndarray], current_stock: np.ndarray) -> float:
        """Calcule le coût total pour un pas de temps."""
        prod_cost = self.compute_production_cost(action)
        inv_cost = self.compute_inventory_cost(current_stock)
        short_cost = self.compute_shortage_cost(current_stock)
        
        return prod_cost + inv_cost + short_cost

    def compute_all_costs(self, action: Dict[str, np.ndarray], current_stock: np.ndarray) -> Dict[str, float]:
        """Retourne tous les coûts calculés sous forme de dictionnaire."""
        return {
            'production_cost': self.compute_production_cost(action),
            'inventory_cost': self.compute_inventory_cost(current_stock),
            'shortage_cost': self.compute_shortage_cost(current_stock),
        }
