# components/action_validators.py

import numpy as np
from typing import Dict
from config.environment_configs import PDPEnvironmentConfig
from .constraints import ConstraintsManager

class ActionValidator:
    """
    Valide et contraint les actions.
    Le rôle principal est maintenant de convertir l'action normalisée en quantité réelle
    et d'appliquer les contraintes de capacité via ConstraintsManager.
    """
    
    def __init__(self, config: PDPEnvironmentConfig):
        self.config = config
        self.constraints_manager = ConstraintsManager(config)
        
    def validate_and_constrain(self, action: Dict, current_period: int) -> Dict:
        """
        Convertit l'action normalisée (0-1) en quantité réelle et applique les contraintes.
        L'action est un dictionnaire de quantités normalisées.
        """
        
        # 1. Conversion de l'action normalisée (0-1) en quantités réelles maximales
        # L'action est supposée être un dictionnaire de valeurs entre 0 et 1.
        
        # On utilise les capacités maximales pour dénormaliser
        max_regular = np.array(self.config.regular_capacity)
        max_overtime = np.array(self.config.overtime_capacity)
        max_subcontracting = np.array(self.config.subcontracting_capacity)
        
        real_action = {
            'regular': action['regular'] * max_regular,
            'overtime': action['overtime'] * max_overtime,
            'subcontracting': action['subcontracting'] * max_subcontracting
        }
        
        # 2. Application des contraintes de capacité et règles métier
        constrained_action = self.constraints_manager.validate_and_constrain_action(real_action, current_period)
        
        return constrained_action
