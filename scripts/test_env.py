import sys
import os
import numpy as np

# Ajout du chemin racine du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.environment_configs import PDPEnvironmentConfig
from environments.env_registry import EnvironmentRegistry

def test_environment():
    """Test basique de l'environnement"""
    config = PDPEnvironmentConfig(
        n_products=1,
        horizon=12
    )
    
    env = EnvironmentRegistry.create('strategic', config)
    
    print("=== TEST ENVIRONNEMENT ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Reset successful - Observation keys: {list(obs.keys())}")
    print(f"Stock initial: {info['initial_stock']}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step successful - Reward: {reward:.2f}")
    print(f"Stock après step: {info['inventory_level']}")
    print(f"Niveau de service: {info['demand_fulfillment']:.3f}")
    print(f"Coûts: {info['costs']}")
    
    # Test fin d'épisode
    for _ in range(config.horizon - 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
            
    print(f"Fin d'épisode atteinte: {terminated}")
    
    print("✅ Environnement fonctionnel !")

if __name__ == "__main__":
    test_environment()
