# agents/baseline_strategies.py

import numpy as np
from typing import Dict, Tuple
from environments.base_pdp_env import BasePDPEnvironment
from config.environment_configs import PDPEnvironmentConfig

class BaselineStrategy:
    """Classe de base pour les stratégies de production classiques."""
    
    def __init__(self, env: BasePDPEnvironment):
        self.env = env
        self.config = env.config
        self.regular_cap = np.array(self.config.regular_capacity)
        self.overtime_cap = np.array(self.config.overtime_capacity)
        self.subcontracting_cap = np.array(self.config.subcontracting_capacity)
        
    def get_action(self, observation: Dict) -> Dict[str, np.ndarray]:
        """
        Calcule l'action (ratios 0-1) à partir de l'observation.
        Doit être implémenté par les sous-classes.
        """
        raise NotImplementedError

    def run_episode(self) -> Tuple[float, Dict]:
        """Exécute un épisode complet avec la stratégie."""
        obs, info = self.env.reset()
        done = False
        total_reward = 0
        metrics = []
        
        while not done:
            action_ratios = self.get_action(obs)
            
            # Conversion des ratios en action aplatie pour l'environnement
            action_flat = np.concatenate([action_ratios[k] for k in ['regular', 'overtime', 'subcontracting']])
            
            obs, reward, terminated, truncated, info = self.env.step(action_flat)
            done = terminated or truncated
            total_reward += reward
            
            # Stockage des métriques
            metrics.append({
                'period': info['period'],
                'reward': reward,
                'total_production': info['total_production'],
                'inventory_level': info['inventory_level'].copy(),
                'demand_fulfillment': info['demand_fulfillment'],
                'costs': info['costs']
            })
            
        return total_reward, {'metrics': metrics, 'final_info': info}

class LevelStrategy(BaselineStrategy):
    """Stratégie Nivelée: Production régulière et constante."""
    
    def get_action(self, observation: Dict) -> Dict[str, np.ndarray]:
        """
        Produit à un niveau constant (ex: 80% de la capacité régulière)
        et utilise les autres capacités seulement si le stock est très bas.
        """
        
        # Production régulière à 80% de la capacité max
        regular_ratio = np.full(self.config.n_products, 0.8)
        
        # Utilisation des autres capacités (overtime, subcontracting) seulement si le stock est bas
        current_stock = observation['current_stock']
        
        # Seuil de stock bas (ex: 10% du stock max)
        low_stock_threshold = np.array(self.config.max_stock) * 0.1
        
        # Si le stock est inférieur au seuil, on utilise 50% des capacités supplémentaires
        overtime_ratio = np.where(current_stock < low_stock_threshold, 0.5, 0.0)
        subcontracting_ratio = np.where(current_stock < low_stock_threshold, 0.5, 0.0)
        
        return {
            'regular': regular_ratio,
            'overtime': overtime_ratio,
            'subcontracting': subcontracting_ratio
        }

class ChaseStrategy(BaselineStrategy):
    """Stratégie Synchrone (Chase): Ajuste la production à la demande future."""
    
    def get_action(self, observation: Dict) -> Dict[str, np.ndarray]:
        """
        Calcule la production nécessaire pour couvrir la demande du prochain pas de temps
        et ajuste les capacités en conséquence.
        """
        
        current_stock = observation['current_stock']
        # Demande du prochain pas de temps (première colonne de future_demands)
        next_demand = observation['future_demands'][:, 0]
        
        # Production nécessaire pour couvrir la demande et maintenir le stock à zéro
        required_production = next_demand - current_stock
        required_production = np.maximum(required_production, 0) # Ne pas produire si stock > demande
        
        # Allocation de la production nécessaire aux différentes capacités
        
        # 1. Capacité régulière
        regular_prod = np.minimum(required_production, self.regular_cap)
        remaining_prod = required_production - regular_prod
        
        # 2. Heures supplémentaires
        overtime_prod = np.minimum(remaining_prod, self.overtime_cap)
        remaining_prod -= overtime_prod
        
        # 3. Sous-traitance
        subcontracting_prod = np.minimum(remaining_prod, self.subcontracting_cap)
        
        # Conversion en ratios (l'environnement s'occupera de la validation finale)
        regular_ratio = regular_prod / (self.regular_cap + 1e-6)
        overtime_ratio = overtime_prod / (self.overtime_cap + 1e-6)
        subcontracting_ratio = subcontracting_prod / (self.subcontracting_cap + 1e-6)
        
        return {
            'regular': regular_ratio,
            'overtime': overtime_ratio,
            'subcontracting': subcontracting_ratio
        }

class FixedModerateStrategy(BaselineStrategy):
    """Stratégie Modérée Fixe: Utilise un mix fixe de capacités."""
    
    def get_action(self, observation: Dict) -> Dict[str, np.ndarray]:
        """
        Utilise un ratio fixe de chaque capacité, indépendamment de la demande ou du stock.
        Ceci est une stratégie de base pour la comparaison.
        """
        
        # Ratios fixes (exemple: 70% régulier, 10% heures sup, 5% sous-traitance)
        regular_ratio = np.full(self.config.n_products, 0.7)
        overtime_ratio = np.full(self.config.n_products, 0.1)
        subcontracting_ratio = np.full(self.config.n_products, 0.05)
        
        return {
            'regular': regular_ratio,
            'overtime': overtime_ratio,
            'subcontracting': subcontracting_ratio
        }

# Registre des stratégies pour l'évaluation
BASELINE_STRATEGIES = {
    'level': LevelStrategy,
    'chase': ChaseStrategy,
    'fixed_moderate': FixedModerateStrategy
}
