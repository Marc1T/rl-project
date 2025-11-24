# scripts/compare_strategies.py

import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from config.environment_configs import PDPEnvironmentConfig
from environments.env_registry import EnvironmentRegistry
from agents.baseline_strategies import BASELINE_STRATEGIES

def evaluate_strategy(strategy_name: str, config: PDPEnvironmentConfig, 
                      env_type: str, n_episodes: int) -> Dict[str, Any]:
    """Évalue une stratégie (RL ou Baseline) sur plusieurs épisodes."""
    
    results = {
        'total_rewards': [],
        'final_stocks': [],
        'service_levels': [],
        'total_costs': [],
        'details': []  # Pour stockage détaillé
    }
    
    if strategy_name in BASELINE_STRATEGIES:
        # Stratégie Baseline
        env = EnvironmentRegistry.create(env_type, config)
        StrategyClass = BASELINE_STRATEGIES[strategy_name]
        strategy = StrategyClass(env)
        
        for episode in range(n_episodes):
            total_reward, info = strategy.run_episode()
            
            final_info = info['final_info']
            
            results['total_rewards'].append(total_reward)
            results['final_stocks'].append(final_info['inventory_level'][0])
            results['service_levels'].append(final_info['demand_fulfillment'])
            
            # Calcul du coût total de l'épisode
            episode_cost = sum(
                m['costs']['production_cost'] + 
                m['costs']['inventory_cost'] + 
                m['costs']['shortage_cost'] 
                for m in info['metrics']
            )
            results['total_costs'].append(episode_cost)
            
            # Détails pour analyse
            results['details'].append({
                'episode': episode,
                'metrics': info['metrics']
            })
            
    elif strategy_name.startswith('RL:'):
        # Stratégie RL
        model_path = strategy_name.split(':', 1)[1]
        
        def make_env():
            return EnvironmentRegistry.create(env_type, config)

        eval_env = DummyVecEnv([make_env])

        # Charger VecNormalize si disponible
        vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            eval_env = VecNormalize.load(vec_normalize_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        
        model = PPO.load(model_path, env=eval_env)
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            total_reward = 0
            episode_cost = 0
            episode_metrics = []
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, terminateds, infos = eval_env.step(action)
                info = infos[0]
                reward = rewards[0]
                done = terminateds[0]
                
                total_reward += reward
                
                step_cost = (
                    info['costs']['production_cost'] + 
                    info['costs']['inventory_cost'] + 
                    info['costs']['shortage_cost']
                )
                episode_cost += step_cost
                
                episode_metrics.append({
                    'period': info['period'],
                    'reward': info['reward'],
                    'total_production': info['total_production'],
                    'inventory_level': info['inventory_level'].copy(),
                    'demand_fulfillment': info['demand_fulfillment'],
                    'costs': info['costs']
                })
                
            results['total_rewards'].append(total_reward)
            results['final_stocks'].append(info['inventory_level'][0])
            results['service_levels'].append(info['demand_fulfillment'])
            results['total_costs'].append(episode_cost)
            results['details'].append({
                'episode': episode,
                'metrics': episode_metrics
            })
            
    else:
        raise ValueError(f"Stratégie inconnue: {strategy_name}")
        
    return {
        'mean_reward': np.mean(results['total_rewards']),
        'std_reward': np.std(results['total_rewards']),
        'mean_cost': np.mean(results['total_costs']),
        'std_cost': np.std(results['total_costs']),
        'mean_service': np.mean(results['service_levels']),
        'std_service': np.std(results['service_levels']),
        'mean_stock': np.mean(results['final_stocks']),
        'raw_results': results
    }

def main():
    parser = argparse.ArgumentParser(description="Comparer les stratégies PDP (RL vs Baselines)")
    parser.add_argument('--episodes', type=int, default=10, help='Nombre d\'épisodes')
    parser.add_argument('--rl_model', type=str, help='Chemin vers le modèle RL')
    parser.add_argument('--env_type', type=str, default='strategic', 
                       choices=['base', 'strategic'])
    parser.add_argument('--config', type=str, help='Chemin vers fichier de config personnalisé')
    parser.add_argument('--save_details', action='store_true', 
                       help='Sauvegarder les détails dans un CSV')
    
    args = parser.parse_args()
    
    # Configuration de l'environnement
    if args.config:
        # Charger config personnalisée (pour exemples réels)
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PDPEnvironmentConfig(**config_dict)
    else:
        # Configuration par défaut
        config = PDPEnvironmentConfig(
            n_products=1,
            horizon=12,
            normalize_observations=True
        )
    
    strategies_to_compare = list(BASELINE_STRATEGIES.keys())
    if args.rl_model:
        strategies_to_compare.append(f"RL:{args.rl_model}")
        
    print(f"🎯 COMPARAISON DE STRATÉGIES sur {args.episodes} épisodes")
    print(f"   Horizon: {config.horizon} périodes")
    print(f"   Produits: {config.n_products}")
    print()
    
    comparison_data = []
    all_results = {}
    
    for strategy_name in strategies_to_compare:
        display_name = strategy_name.split(':')[-1].split('/')[-1]
        print(f"=== Évaluation: {display_name} ===")
        
        results = evaluate_strategy(strategy_name, config, args.env_type, args.episodes)
        all_results[display_name] = results
        
        comparison_data.append({
            'Stratégie': display_name,
            'Reward Moyen': f"{results['mean_reward']:.2f}",
            'Coût Moyen': f"{results['mean_cost']:.2f}",
            'Service Level': f"{results['mean_service']:.3f}",
            'Stock Final': f"{results['mean_stock']:.1f}"
        })
        
        print(f"   ✓ Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   ✓ Coût: {results['mean_cost']:.2f} ± {results['std_cost']:.2f}")
        print(f"   ✓ Service: {results['mean_service']:.3f} ± {results['std_service']:.3f}")
        print(f"   ✓ Stock: {results['mean_stock']:.1f}")
        print()
    
    # Affichage final
    df = pd.DataFrame(comparison_data)
    print("\n📊 RÉSULTATS DE LA COMPARAISON")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Identifier le meilleur
    best_by_reward = df.loc[df['Reward Moyen'].astype(float).idxmax(), 'Stratégie']
    best_by_service = df.loc[df['Service Level'].astype(float).idxmax(), 'Stratégie']
    
    print("\n🏆 MEILLEURS RÉSULTATS:")
    print(f"   Meilleur Reward: {best_by_reward}")
    print(f"   Meilleur Service: {best_by_service}")
    
    # Sauvegarder si demandé
    if args.save_details:
        output_dir = "./results/"
        os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)
        print(f"\n💾 Résumé sauvegardé: {output_dir}comparison_summary.csv")
        
        # Sauvegarder détails pour chaque stratégie
        for strategy_name, results in all_results.items():
            details_df = pd.DataFrame([
                {
                    'episode': d['episode'],
                    'period': m['period'],
                    'production': m['total_production'],
                    'stock': m['inventory_level'][0],
                    'service': m['demand_fulfillment'],
                    'cost': sum(m['costs'].values())
                }
                for d in results['raw_results']['details']
                for m in d['metrics']
            ])
            filename = f"details_{strategy_name.replace('/', '_')}.csv"
            details_df.to_csv(os.path.join(output_dir, filename), index=False)
        
        print(f"   Détails sauvegardés dans: {output_dir}")

if __name__ == "__main__":
    main()
