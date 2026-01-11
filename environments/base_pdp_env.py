# environments/base_pdp_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from config.environment_configs import PDPEnvironmentConfig
from components.normalizers import ObservationNormalizer
from components.reward_calculators import RewardCalculator
from components.action_validators import ActionValidator
from components.observation_builders import ObservationBuilder
from components.demand_generators import DemandGenerator
from components.cost_calculators import CostCalculator

class BasePDPEnvironment(gym.Env):
    """Environnement PDP de base entièrement configurable"""
    
    def __init__(self, config: PDPEnvironmentConfig):
        super().__init__()
        self.config = config
        self._setup_spaces()
        self._setup_components()
        self.demands = None
        
    def _setup_spaces(self):
        """Configure les espaces d'observation et d'action"""
        self.observation_space = self._create_observation_space()
        
        # Action normalisée entre 0 et 1
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0,
            shape=(3 * self.config.n_products,),
            dtype=np.float32
        )
    
    def _setup_components(self):
        """Initialise les composants modulaires.
        
        Note: ObservationNormalizer est desactive par defaut car VecNormalize
        dans ppo_trainer.py gere la normalisation adaptative des observations.
        Ceci evite une double normalisation qui fausserait les valeurs.
        """
        # IMPORTANT: Ne pas utiliser ObservationNormalizer si VecNormalize est utilise
        # La normalisation est geree par VecNormalize dans le trainer
        self.normalizer = None  # Desactive pour eviter double normalisation
        self.reward_calculator = RewardCalculator(self.config)
        self.action_validator = ActionValidator(self.config)
        self.obs_builder = ObservationBuilder(self.config)
        # Obtenir l'intensité de demande depuis la config (defaut: high)
        demand_intensity = getattr(self.config, 'demand_intensity', 'high')
        self.demand_generator = DemandGenerator(self.config, demand_intensity=demand_intensity)
        self.cost_calculator = CostCalculator(self.config)
    
    def _create_observation_space(self) -> spaces.Dict:
        """Crée l'espace d'observation basé sur la configuration"""
        # Limites plus réalistes pour le stock
        stock_low = -np.array(self.config.max_stock, dtype=np.float32) * 0.5  # Permet backorders
        stock_high = np.array(self.config.max_stock, dtype=np.float32)
        
        obs_dict = {
            'current_stock': spaces.Box(
                low=stock_low,
                high=stock_high,
                dtype=np.float32
            ),
            'future_demands': spaces.Box(
                low=0, 
                high=np.max(self.config.regular_capacity) * 3,  # Borne réaliste
                shape=(self.config.n_products, self.config.demand_lookahead),
                dtype=np.float32
            )
        }
        
        if self.config.include_period_info:
            obs_dict['current_period'] = spaces.Box(
                low=0, high=self.config.horizon, shape=(1,), dtype=np.float32
            )
            
        return spaces.Dict(obs_dict)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Réinitialise l'environnement"""
        super().reset(seed=seed)
        
        self.current_period = 0
        self.current_stock = np.array(self.config.initial_stock, dtype=np.float32)
        
        # Génération des demandes
        if self.config.demand_file_path:
            self.demands = self.demand_generator.import_real_demands(self.config.demand_file_path)
        else:
            self.demands = self.demand_generator.generate_synthetic_demands(seed=seed)
        
        raw_obs = self.obs_builder.build(
            current_stock=self.current_stock,
            current_period=self.current_period,
            demands=self.demands
        )
        
        observation = self.normalizer.normalize(raw_obs) if self.normalizer else raw_obs
        info = self._get_initial_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Exécute un pas de temps"""
        
        # Sauvegarde du stock avant action pour le calcul du reward
        stock_before = self.current_stock.copy()
        
        # 1. Conversion et validation de l'action
        action_ratios = self._flattened_action_to_dict(action)
        valid_action = self.action_validator.validate_and_constrain(action_ratios, self.current_period)
        
        # 2. Demande courante
        current_demand = self.demands[:, self.current_period]
        
        # 3. Mise à jour de l'état
        info_step = self._update_state(valid_action, current_demand)
        
        # 4. Calcul du reward AVEC les bonnes informations
        reward = self.reward_calculator.compute(
            action=valid_action,
            stock_before=stock_before,
            current_stock=self.current_stock,
            current_demand=current_demand,
            current_period=self.current_period
        )
        
        # 5. Incrémentation de la période
        self.current_period += 1
        
        # 6. Vérification de la fin d'épisode
        terminated = self.current_period >= self.config.horizon
        truncated = False
        
        # 7. Observation suivante
        observation = self._get_observation()
        
        # 8. Informations supplémentaires
        info = self._get_info(valid_action, reward, info_step, current_demand)
        
        return observation, reward, terminated, truncated, info
    
    def _flattened_action_to_dict(self, flattened_action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convertit l'action aplatie en dictionnaire structuré de ratios (0-1)"""
        n = self.config.n_products
        return {
            'regular': flattened_action[0:n],
            'overtime': flattened_action[n:2*n],
            'subcontracting': flattened_action[2*n:3*n]
        }
    
    def _update_state(self, action: Dict, current_demand: np.ndarray) -> Dict:
        """Met à jour l'état du système - VERSION CORRIGÉE"""
        
        # Stock avant production
        stock_before_prod = self.current_stock.copy()
        
        # Production totale
        total_production = (
            action['regular'] + 
            action['overtime'] + 
            action['subcontracting']
        )
        
        # Stock après production
        stock_after_prod = stock_before_prod + total_production
        
        # Demande satisfaite (ce qui peut être livré)
        demand_satisfied = np.minimum(np.maximum(stock_after_prod, 0), current_demand)
        
        # On soustrait la demande réelle, pas la demande satisfaite
        self.current_stock = stock_after_prod - current_demand
        
        # Application des limites de stock si pas de backorders
        if not self.config.allow_backorders:
            self.current_stock = np.maximum(self.current_stock, 0)
        
        # Métriques pour l'info
        return {
            'stock_before_prod': stock_before_prod,
            'stock_after_prod': stock_after_prod,
            'current_demand': current_demand,
            'demand_satisfied': demand_satisfied,
            'total_production': total_production,
            'shortage': np.maximum(current_demand - stock_after_prod, 0)
        }
    
    def _get_observation(self) -> Dict:
        """Construit l'observation"""
        raw_obs = self.obs_builder.build(
            current_stock=self.current_stock,
            current_period=self.current_period,
            demands=self.demands
        )
        return self.normalizer.normalize(raw_obs) if self.normalizer else raw_obs
    
    def _calculate_service_level(self, demand_satisfied: np.ndarray, current_demand: np.ndarray) -> float:
        """Calcule le niveau de service (ratio de demande satisfaite)"""
        total_demand = np.sum(current_demand)
        total_satisfied = np.sum(demand_satisfied)
        
        return float(total_satisfied / total_demand) if total_demand > 0 else 1.0
    
    def _get_initial_info(self) -> Dict:
        """Retourne les informations initiales"""
        return {
            'initial_stock': self.current_stock.copy(),
            'demands': self.demands.copy()
        }
    
    def _get_info(self, action: Dict, reward: float, info_step: Dict, current_demand: np.ndarray) -> Dict:
        """Retourne les informations de diagnostic"""
        
        # Calcul des coûts pour le reporting
        costs = self.cost_calculator.compute_all_costs(action, self.current_stock)
        
        return {
            'period': self.current_period - 1,
            'reward': reward,
            'total_production': np.sum(info_step['total_production']),
            'inventory_level': self.current_stock.copy(),
            'demand_fulfillment': self._calculate_service_level(
                info_step['demand_satisfied'], 
                current_demand
            ),
            'costs': costs,
            'raw_metrics': info_step
        }
    
    def render(self):
        """Méthode de rendu (non implémentée)"""
        pass

    def close(self):
        """Méthode de fermeture (non implémentée)"""
        pass
