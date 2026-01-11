# scripts/train_agents.py
"""
Script pour entrainer et comparer differents agents RL sur le PDP.

Usage:
    python scripts/train_agents.py --agent sac --episodes 1000
    python scripts/train_agents.py --agent all --episodes 500
    python scripts/train_agents.py --list
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

# Ajouter le chemin racine du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.environment_configs import PDPEnvironmentConfig
from environments.env_registry import EnvironmentRegistry
from agents.agent_registry import AgentRegistry, AgentConfig


def get_default_config() -> PDPEnvironmentConfig:
    """Configuration par defaut pour les tests"""
    return PDPEnvironmentConfig(
        n_products=2,
        horizon=12,
        regular_capacity=[100.0, 80.0],
        overtime_capacity=[30.0, 25.0],
        subcontracting_capacity=[50.0, 40.0],
        initial_stock=[50.0, 40.0],
        normalize_observations=False  # Les agents gerent leur propre normalisation
    )


def train_agent(
    agent_name: str,
    env_config: PDPEnvironmentConfig,
    n_episodes: int = 1000,
    seed: int = 42,
    save_dir: str = "./models/agents",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Entraine un agent et retourne les resultats.
    
    Args:
        agent_name: Nom de l'agent (du registre)
        env_config: Configuration de l'environnement
        n_episodes: Nombre d'episodes d'entrainement
        seed: Graine aleatoire
        save_dir: Repertoire de sauvegarde
        verbose: Afficher la progression
        
    Returns:
        Dictionnaire avec les resultats
    """
    print(f"\n{'='*60}")
    print(f"ENTRAINEMENT: {agent_name.upper()}")
    print(f"{'='*60}")
    
    # Creer l'environnement
    env = EnvironmentRegistry.create('strategic', env_config)
    
    # Configuration de l'agent
    agent_config = AgentConfig(
        gamma=0.99,
        learning_rate=1e-3 if agent_name in ['monte_carlo', 'q_learning', 'sarsa'] else 3e-4,
        seed=seed
    )
    
    # Creer l'agent
    agent = AgentRegistry.create(agent_name, env, agent_config)
    
    # Entrainer
    start_time = datetime.now()
    rewards = agent.train(n_episodes, verbose=verbose)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluer
    print(f"\n📊 Evaluation sur 10 episodes...")
    eval_results = agent.evaluate(n_episodes=10)
    
    # Sauvegarder
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if hasattr(agent, 'save'):
        if agent_name in ['monte_carlo', 'q_learning', 'sarsa']:
            agent.save(model_path + ".pkl")
        else:
            agent.save(model_path + ".pt")
        print(f"💾 Modele sauvegarde: {model_path}")
    
    # Resultats
    results = {
        'agent': agent_name,
        'n_episodes': n_episodes,
        'training_time_seconds': training_time,
        'final_reward_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
        'final_reward_std': np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards),
        **eval_results
    }
    
    print(f"\n✅ Resultats {agent_name}:")
    print(f"   Temps d'entrainement: {training_time:.1f}s")
    print(f"   Reward moyen (100 derniers): {results['final_reward_mean']:.2f}")
    print(f"   Reward evaluation: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Service moyen: {eval_results['mean_service']:.3f}")
    
    return results


def compare_all_agents(
    n_episodes: int = 500,
    seed: int = 42,
    save_dir: str = "./models/agents"
) -> List[Dict[str, Any]]:
    """
    Compare tous les agents disponibles.
    
    Returns:
        Liste des resultats pour chaque agent
    """
    env_config = get_default_config()
    
    all_results = []
    agents_to_test = [
        'monte_carlo', 'q_learning', 'sarsa',  # Tabulaires
        'dqn',  # Deep Q
        'a2c', 'sac'  # Actor-Critic
    ]
    
    for agent_name in agents_to_test:
        try:
            results = train_agent(
                agent_name,
                env_config,
                n_episodes=n_episodes,
                seed=seed,
                save_dir=save_dir,
                verbose=False
            )
            all_results.append(results)
        except Exception as e:
            print(f"❌ Erreur avec {agent_name}: {e}")
            all_results.append({
                'agent': agent_name,
                'error': str(e)
            })
    
    # Afficher le comparatif
    print("\n" + "=" * 80)
    print("COMPARAISON DES AGENTS")
    print("=" * 80)
    print(f"{'Agent':<15} {'Reward Eval':>12} {'Service':>10} {'Temps (s)':>10}")
    print("-" * 80)
    
    for r in all_results:
        if 'error' not in r:
            print(f"{r['agent']:<15} {r['mean_reward']:>12.2f} {r['mean_service']:>10.3f} {r['training_time_seconds']:>10.1f}")
        else:
            print(f"{r['agent']:<15} {'ERREUR':>12}")
    
    print("=" * 80)
    
    # Sauvegarder les resultats
    results_path = os.path.join(save_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📊 Resultats sauvegardes: {results_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Entrainement des agents RL pour le PDP")
    parser.add_argument('--agent', type=str, default='sac',
                        help="Nom de l'agent (ou 'all' pour tous)")
    parser.add_argument('--episodes', type=int, default=1000,
                        help="Nombre d'episodes d'entrainement")
    parser.add_argument('--seed', type=int, default=42,
                        help="Graine aleatoire")
    parser.add_argument('--save_dir', type=str, default='./models/agents',
                        help="Repertoire de sauvegarde")
    parser.add_argument('--list', action='store_true',
                        help="Lister les agents disponibles")
    
    args = parser.parse_args()
    
    if args.list:
        AgentRegistry.print_summary()
        return
    
    if args.agent == 'all':
        compare_all_agents(
            n_episodes=args.episodes,
            seed=args.seed,
            save_dir=args.save_dir
        )
    else:
        if args.agent not in AgentRegistry.list_available():
            print(f"❌ Agent '{args.agent}' non trouve.")
            print(f"   Disponibles: {AgentRegistry.list_available()}")
            return
        
        env_config = get_default_config()
        train_agent(
            args.agent,
            env_config,
            n_episodes=args.episodes,
            seed=args.seed,
            save_dir=args.save_dir
        )


if __name__ == "__main__":
    main()
