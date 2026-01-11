# scripts/evaluate_real_example.py
"""
Script pour évaluer un modèle RL sur un exemple et comparer avec les baselines
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from config.real_examples_configs import get_example_config
from environments.env_registry import EnvironmentRegistry
from agents.baseline_strategies import BASELINE_STRATEGIES

def evaluate_rl_model(model_path: str, config, env_type: str, n_episodes: int = 10):
    """Évalue un modèle RL"""
    
    def make_env():
        return EnvironmentRegistry.create(env_type, config)
    
    eval_env = DummyVecEnv([make_env])
    
    # Charger VecNormalize
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    
    model = PPO.load(model_path, env=eval_env)
    
    results = {
        'rewards': [],
        'costs': [],
        'service_levels': [],
        'stocks': []
    }
    
    for episode in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        total_cost = 0
        service_levels = []
        stocks = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            
            reward = rewards[0]
            info = infos[0]
            done = dones[0]
            
            total_reward += reward
            total_cost += sum(info['costs'].values())
            service_levels.append(info['demand_fulfillment'])
            stocks.append(info['inventory_level'][0])
        
        results['rewards'].append(total_reward)
        results['costs'].append(total_cost)
        results['service_levels'].append(np.mean(service_levels))
        results['stocks'].append(np.mean(stocks))
    
    return {
        'mean_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        'mean_cost': np.mean(results['costs']),
        'mean_service': np.mean(results['service_levels']),
        'mean_stock': np.mean(results['stocks'])
    }

def evaluate_baseline(strategy_name: str, config, env_type: str, n_episodes: int = 10):
    """Évalue une stratégie baseline"""
    
    results = {
        'rewards': [],
        'costs': [],
        'service_levels': [],
        'stocks': []
    }
    
    for episode in range(n_episodes):
        env = EnvironmentRegistry.create(env_type, config)
        StrategyClass = BASELINE_STRATEGIES[strategy_name]
        strategy = StrategyClass(env)
        
        total_reward, info = strategy.run_episode()
        metrics = info['metrics']
        
        total_cost = sum(
            m['costs']['production_cost'] + 
            m['costs']['inventory_cost'] + 
            m['costs']['shortage_cost']
            for m in metrics
        )
        
        service_levels = [m['demand_fulfillment'] for m in metrics]
        stocks = [m['inventory_level'][0] for m in metrics]
        
        results['rewards'].append(total_reward)
        results['costs'].append(total_cost)
        results['service_levels'].append(np.mean(service_levels))
        results['stocks'].append(np.mean(stocks))
    
    return {
        'mean_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        'mean_cost': np.mean(results['costs']),
        'mean_service': np.mean(results['service_levels']),
        'mean_stock': np.mean(results['stocks'])
    }

def main():
    parser = argparse.ArgumentParser(
        description="Évaluer un modèle RL sur un exemple réel"
    )
    parser.add_argument(
        '--example',
        type=str,
        required=True,
        choices=['rouleurs', 'pdp_table', 'compresseurs', 'usinage'],
        help='Exemple à utiliser'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Chemin vers le modèle RL (.zip)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Nombre d\'épisodes d\'évaluation'
    )
    parser.add_argument(
        '--env_type',
        type=str,
        default='strategic',
        choices=['base', 'strategic'],
        help='Type d\'environnement'
    )
    parser.add_argument(
        '--compare_baselines',
        action='store_true',
        help='Comparer avec les stratégies baselines'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"📊 ÉVALUATION SUR EXEMPLE: {args.example.upper()}")
    print(f"{'='*80}\n")
    
    # Charger la configuration
    config = get_example_config(args.example)
    
    print(f"📋 Configuration:")
    print(f"   Horizon: {config.horizon} périodes")
    print(f"   Produits: {config.n_products}")
    print(f"   Épisodes: {args.episodes}")
    print()
    
    # Évaluer le modèle RL
    print(f"🤖 Évaluation du modèle RL...")
    print(f"   Modèle: {args.model}")
    
    rl_results = evaluate_rl_model(args.model, config, args.env_type, args.episodes)
    
    print(f"\n✅ Résultats RL:")
    print(f"   Reward moyen: {rl_results['mean_reward']:.2f} ± {rl_results['std_reward']:.2f}")
    print(f"   Coût moyen: {rl_results['mean_cost']:.2f}")
    print(f"   Service moyen: {rl_results['mean_service']:.3f}")
    print(f"   Stock moyen: {rl_results['mean_stock']:.1f}")
    
    # Comparer avec baselines si demandé
    if args.compare_baselines:
        print(f"\n📊 Comparaison avec les baselines...")
        
        comparison_data = [{
            'Stratégie': 'RL (PPO)',
            'Reward': f"{rl_results['mean_reward']:.2f}",
            'Coût': f"{rl_results['mean_cost']:.2f}",
            'Service': f"{rl_results['mean_service']:.3f}",
            'Stock': f"{rl_results['mean_stock']:.1f}"
        }]
        
        for strategy_name in BASELINE_STRATEGIES.keys():
            print(f"   Testing {strategy_name}...")
            baseline_results = evaluate_baseline(
                strategy_name, config, args.env_type, args.episodes
            )
            
            comparison_data.append({
                'Stratégie': strategy_name,
                'Reward': f"{baseline_results['mean_reward']:.2f}",
                'Coût': f"{baseline_results['mean_cost']:.2f}",
                'Service': f"{baseline_results['mean_service']:.3f}",
                'Stock': f"{baseline_results['mean_stock']:.1f}"
            })
        
        # Afficher le tableau de comparaison
        df = pd.DataFrame(comparison_data)
        print(f"\n{'='*80}")
        print(f"📊 COMPARAISON - {args.example.upper()}")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        # Identifier le meilleur
        best_service_idx = df['Service'].astype(float).idxmax()
        best_reward_idx = df['Reward'].astype(float).idxmax()
        
        print(f"\n🏆 MEILLEURS RÉSULTATS:")
        print(f"   Meilleur Service: {df.loc[best_service_idx, 'Stratégie']} "
              f"({df.loc[best_service_idx, 'Service']})")
        print(f"   Meilleur Reward: {df.loc[best_reward_idx, 'Stratégie']} "
              f"({df.loc[best_reward_idx, 'Reward']})")
        
        # Sauvegarder
        output_dir = f"./results/{args.example}/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "evaluation_comparison.csv")
        df.to_csv(output_file, index=False)
        print(f"\n💾 Résultats sauvegardés: {output_file}")
    
    print(f"\n✅ Évaluation terminée!")

if __name__ == "__main__":
    main()
