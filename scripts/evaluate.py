# scripts/evaluate.py

import argparse
import os
import sys
import numpy as np

# Ajout du chemin racine du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from config.environment_configs import PDPEnvironmentConfig
from environments.env_registry import EnvironmentRegistry

def evaluate(model_path: str, env_type: str, n_episodes: int = 5):
    """Évaluation avec configuration cohérente"""
    
    model_path = os.path.normpath(model_path)
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    # Configuration de l'environnement (doit être la même que l'entraînement RL)
    # On suppose que le modèle a été entraîné avec la normalisation
    config = PDPEnvironmentConfig(
        n_products=1,
        horizon=12,
        normalize_observations=True # Important pour charger VecNormalize
    )
    
    print(f"🎯 Chargement du modèle: {model_path}")
    model = PPO.load(model_path)
    
    # Création de l’environnement brut
    def make_env():
        return EnvironmentRegistry.create(env_type, config)

    # Vectorisation
    env = DummyVecEnv([make_env])

    # Tenter de charger VecNormalize
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print(f"   Chargement de VecNormalize depuis: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False # On veut le reward réel pour l'évaluation
    
    results = {
        'total_rewards': [],
        'final_stocks': [],
        'service_levels': [],
        'episode_metrics': [] # Pour la visualisation
    }
    
    print(f"\n=== ÉVALUATION PROPRE SUR {n_episodes} ÉPISODES ===")
    
    for episode in range(n_episodes):
        obs = env.reset()
        # info = infos[0]
        done = False
        total_reward = 0
        episode_metrics = []
        
        print(f"\n🎬 Épisode {episode+1}:")
        # print(f"   Stock initial: {info['initial_stock']}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, infos = env.step(action)
            
            reward = rewards[0]
            info = infos[0]
            done = terminated[0]
            total_reward += reward
            
            # Stockage des métriques du pas de temps
            episode_metrics.append({
                'period': info['period'],
                'reward': info['reward'],
                'total_production': info['total_production'],
                'inventory_level': info['inventory_level'].tolist(),
                'demand_fulfillment': info['demand_fulfillment'],
                'costs': info['costs']
            })
            
            if info['period'] == 0:
                act = action[0]

                action_quantities = {
                    'regular': act[0] * config.regular_capacity[0],
                    'overtime': act[1] * config.overtime_capacity[0],
                    'subcontracting': act[2] * config.subcontracting_capacity[0]
                }
                print(f"   Premier step - Production: R{action_quantities['regular']:.1f}/O{action_quantities['overtime']:.1f}/S{action_quantities['subcontracting']:.1f}")

            
        
        results['total_rewards'].append(total_reward)
        results['final_stocks'].append(info['inventory_level'][0])
        results['service_levels'].append(info['demand_fulfillment'])
        results['episode_metrics'].append(episode_metrics)
        
        print(f"   ✅ Reward total: {total_reward:8.1f}")
        print(f"   📦 Stock final: {info['inventory_level'][0]:8.1f}")
        print(f"   🎯 Service: {info['demand_fulfillment']:6.3f}")

    # Résultats
    print(f"\n📊 PERFORMANCE MOYENNE:")
    print(f"   Reward: {np.mean(results['total_rewards']):.1f} ± {np.std(results['total_rewards']):.1f}")
    print(f"   Stock final: {np.mean(results['final_stocks']):.1f}")
    print(f"   Niveau service: {np.mean(results['service_levels']):.3f}")
    
    # Sauvegarde des métriques pour la visualisation
    import json
    log_dir = os.path.dirname(model_path)
    with open(os.path.join(log_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(results['episode_metrics'], f, indent=4)
    print(f"\n💾 Métriques d'évaluation sauvegardées dans: {log_dir}/evaluation_metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation propre")
    parser.add_argument('--model', type=str, required=True, help='Chemin vers le modèle')
    parser.add_argument('--episodes', type=int, default=5, help='Nombre d épisodes')
    parser.add_argument('--env_type', type=str, default='strategic', choices=['base', 'strategic'], help='Type d environnement à utiliser')
    
    args = parser.parse_args()
    evaluate(args.model, args.env_type, args.episodes)
