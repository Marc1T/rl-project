# environments/strategic_pdp_env.py

import numpy as np
from typing import Dict, Tuple, Optional
from .base_pdp_env import BasePDPEnvironment
from config.environment_configs import PDPEnvironmentConfig

class StrategicPDPEnvironment(BasePDPEnvironment):
    """
    Environnement PDP avec stratégie modérée avancée.
    Ajoute des contraintes et des rewards pour favoriser la stabilité de la production.
    """
    
    def __init__(self, config: PDPEnvironmentConfig):
        super().__init__(config)
        # Stocke l'action précédente en quantités réelles
        self.previous_action = {
            'regular': np.zeros(config.n_products, dtype=np.float32),
            'overtime': np.zeros(config.n_products, dtype=np.float32),
            'subcontracting': np.zeros(config.n_products, dtype=np.float32)
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Réinitialise l'environnement et l'action précédente."""
        obs, info = super().reset(seed=seed, options=options)
        self.previous_action = {
            'regular': np.zeros(self.config.n_products, dtype=np.float32),
            'overtime': np.zeros(self.config.n_products, dtype=np.float32),
            'subcontracting': np.zeros(self.config.n_products, dtype=np.float32)
        }
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Override avec gestion stratégique avancée"""
        
        # Sauvegarde du stock avant action
        stock_before = self.current_stock.copy()
        
        # 1. Conversion et validation de l'action
        action_ratios = self._flattened_action_to_dict(action)
        valid_action = self.action_validator.validate_and_constrain(action_ratios, self.current_period)
        
        # 2. Application des contraintes stratégiques (variation de production)
        strategic_action = self._apply_strategic_constraints(valid_action)
        
        # 3. Demande courante
        current_demand = self.demands[:, self.current_period]
        
        # 4. Mise à jour de l'état
        info_step = self._update_state(strategic_action, current_demand)
        
        # 5. Calcul du reward
        reward = self.reward_calculator.compute(
            action=strategic_action,
            stock_before=stock_before,
            current_stock=self.current_stock,
            current_demand=current_demand,
            current_period=self.current_period
        )
        
        # 6. Ajustement du reward basé sur la stratégie
        strategic_reward = self._adjust_reward_strategic(reward, strategic_action)
        
        # 7. Incrémentation de la période
        self.current_period += 1
        
        # 8. Vérification de la fin d'épisode
        terminated = self.current_period >= self.config.horizon
        truncated = False
        
        # 9. Observation suivante
        observation = self._get_observation()
        
        # 10. Informations supplémentaires
        info = self._get_info(strategic_action, strategic_reward, info_step, current_demand)
        
        # Mise à jour de l'action précédente pour le prochain pas de temps
        for key in self.previous_action:
            self.previous_action[key] = strategic_action[key].copy()
        
        return observation, strategic_reward, terminated, truncated, info
    
    def _apply_strategic_constraints(self, action: Dict) -> Dict:
        """Applique des contraintes stratégiques supplémentaires (ex: variation de production)"""
        constrained = {}
        
        # Limiter la variation brusque de production (max 30% de variation)
        max_variation_ratio = 0.3
        
        for action_type in ['regular', 'overtime', 'subcontracting']:
            
            # Capacité maximale pour ce type d'action
            if action_type == 'regular':
                max_cap = np.array(self.config.regular_capacity, dtype=np.float32)
            elif action_type == 'overtime':
                max_cap = np.array(self.config.overtime_capacity, dtype=np.float32)
            else:
                max_cap = np.array(self.config.subcontracting_capacity, dtype=np.float32)
                
            # Variation maximale autorisée en quantité
            max_variation_qty = max_cap * max_variation_ratio
            
            # Application de la contrainte
            constrained[action_type] = np.clip(
                action[action_type],
                self.previous_action[action_type] - max_variation_qty,
                self.previous_action[action_type] + max_variation_qty
            )
            
            # S'assurer que l'action reste dans les limites de capacité
            constrained[action_type] = np.clip(constrained[action_type], 0, max_cap)
            
        return constrained
    
    def _adjust_reward_strategic(self, base_reward: float, action: Dict) -> float:
        """
        Ajuste le reward pour favoriser les comportements stratégiques.
        VERSION CORRIGÉE avec normalisation appropriée.
        """
        strategic_bonus = 0.0
        
        # 1. Bonus pour stabilité de production
        current_total_prod = np.sum([action[k] for k in action])
        previous_total_prod = np.sum([self.previous_action[k] for k in self.previous_action])
        
        if previous_total_prod > 0:
            variation_ratio = np.abs(current_total_prod - previous_total_prod) / previous_total_prod
            
            if variation_ratio < 0.1:  # Faible variation
                strategic_bonus += 0.1
            elif variation_ratio > 0.5:  # Forte variation (pénalité)
                strategic_bonus -= 0.1
        
        # 2. Bonus pour utilisation équilibrée des capacités
        regular_cap = np.sum(self.config.regular_capacity)
        if regular_cap > 0:
            regular_util = np.sum(action['regular']) / regular_cap
            
            if 0.7 <= regular_util <= 0.95:  # Utilisation optimale
                strategic_bonus += 0.05
        
        # 3. Pénalité pour utilisation excessive de sous-traitance
        total_prod = current_total_prod
        if total_prod > 0:
            subcontracting_ratio = np.sum(action['subcontracting']) / total_prod
            if subcontracting_ratio > 0.3:  # Plus de 30% de sous-traitance
                strategic_bonus -= 0.1
        
        return base_reward + strategic_bonus
