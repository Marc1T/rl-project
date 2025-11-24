import sys
import os

# Ajout du chemin racine du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import unittest
import numpy as np
from config.environment_configs import PDPEnvironmentConfig
from components.cost_calculators import CostCalculator
from components.action_validators import ActionValidator
from components.constraints import ConstraintsManager

class TestComponents(unittest.TestCase):
    
    def setUp(self):
        # Configuration de base pour les tests
        self.config = PDPEnvironmentConfig(
            n_products=1,
            horizon=12,
            regular_capacity=[100.0],
            overtime_capacity=[30.0],
            subcontracting_capacity=[50.0],
            regular_cost=[10.0],
            overtime_cost=[15.0],
            subcontracting_cost=[20.0],
            holding_cost=[2.0],
            shortage_cost=[100.0],
            initial_stock=[0.0],
            max_stock=[500.0]
        )
        self.cost_calculator = CostCalculator(self.config)
        self.action_validator = ActionValidator(self.config)
        self.constraints_manager = ConstraintsManager(self.config)

    def test_cost_calculator_production_cost(self):
        action = {
            'regular': np.array([50.0]),
            'overtime': np.array([10.0]),
            'subcontracting': np.array([5.0])
        }
        # Coût attendu: (50*10) + (10*15) + (5*20) = 500 + 150 + 100 = 750
        expected_cost = 750.0
        self.assertAlmostEqual(self.cost_calculator.compute_production_cost(action), expected_cost)

    def test_cost_calculator_inventory_cost(self):
        # Stock positif (holding cost)
        stock_pos = np.array([100.0])
        # Coût attendu: 100 * 2.0 = 200.0
        self.assertAlmostEqual(self.cost_calculator.compute_inventory_cost(stock_pos), 200.0)
        
        # Stock négatif (pas de holding cost)
        stock_neg = np.array([-50.0])
        self.assertAlmostEqual(self.cost_calculator.compute_inventory_cost(stock_neg), 0.0)

    def test_cost_calculator_shortage_cost(self):
        # Stock négatif (shortage cost)
        stock_neg = np.array([-50.0])
        # Coût attendu: 50 * 100.0 = 5000.0
        self.assertAlmostEqual(self.cost_calculator.compute_shortage_cost(stock_neg), 5000.0)
        
        # Stock positif (pas de shortage cost)
        stock_pos = np.array([100.0])
        self.assertAlmostEqual(self.cost_calculator.compute_shortage_cost(stock_pos), 0.0)

    def test_constraints_manager_capacity_reduction(self):
        # Période 6 (7ème mois) - Capacité régulière réduite de 50%
        period = 6
        capacity = self.constraints_manager.get_available_capacity(period)
        
        # Capacité régulière attendue: 100.0 * 0.5 = 50.0
        self.assertAlmostEqual(capacity['regular'][0], 50.0)
        
        # Période 11 (dernier mois) - Heures supplémentaires interdites
        period = 11
        capacity = self.constraints_manager.get_available_capacity(period)
        
        # Capacité heures sup attendue: 0.0
        self.assertAlmostEqual(capacity['overtime'][0], 0.0)

    def test_action_validator_constrain(self):
        # Action normalisée (ratios)
        action_ratios = {
            'regular': np.array([1.5]), # Trop haut
            'overtime': np.array([0.5]), # Correct
            'subcontracting': np.array([0.0]) # Correct
        }
        
        # Action en quantités réelles (avant contrainte)
        # regular: 1.5 * 100 = 150.0
        # overtime: 0.5 * 30 = 15.0
        
        # Contrainte appliquée (période 0)
        constrained_action = self.action_validator.validate_and_constrain(action_ratios, 0)
        
        # regular doit être plafonné à 100.0
        self.assertAlmostEqual(constrained_action['regular'][0], 100.0)
        # overtime doit être 15.0
        self.assertAlmostEqual(constrained_action['overtime'][0], 15.0)
        
        # Contrainte appliquée (période 6 - regular à 50.0)
        constrained_action_p6 = self.action_validator.validate_and_constrain(action_ratios, 6)
        
        # regular doit être plafonné à 50.0 (capacité réduite)
        self.assertAlmostEqual(constrained_action_p6['regular'][0], 50.0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
