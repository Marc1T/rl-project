# scripts/test_real_examples.py
"""
Script pour tester les stratégies sur les exemples réels d'exercices du cours de gestion de production
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.real_examples_configs import get_example_config
from environments.env_registry import EnvironmentRegistry
from agents.baseline_strategies import BASELINE_STRATEGIES

def visualize_episode(metrics, example_name, strategy_name, demands):
    """Visualise les résultats d'un épisode"""
    periods = [m['period'] for m in metrics]
    stocks = [m['inventory_level'][0] for m in metrics]
    productions = [m['total_production'] for m in metrics]
    service_levels = [m['demand_fulfillment'] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{example_name.upper()} - Stratégie: {strategy_name}', fontsize=16)
    
    # Stock
    axes[0, 0].plot(periods, stocks, marker='o', linewidth=2, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Évolution du Stock')
    axes[0, 0].set_xlabel('Période')
    axes[0, 0].set_ylabel('Stock')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Production vs Demande
    axes[0, 1].plot(periods, productions, marker='s', label='Production', linewidth=2)
    axes[0, 1].plot(periods, demands[:len(periods)], marker='^', label='Demande', linewidth=2)
    axes[0, 1].set_title('Production vs Demande')
    axes[0, 1].set_xlabel('Période')
    axes[0, 1].set_ylabel('Quantité')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Niveau de Service
    axes[1, 0].bar(periods, service_levels, color='green', alpha=0.7)
    axes[1, 0].axhline(y=0.95, color='red', linestyle='--', label='Cible: 95%')
    axes[1, 0].set_title('Niveau de Service')
    axes[1, 0].set_xlabel('Période')
    axes[1, 0].set_ylabel('Service Level')
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Coûts cumulés
    costs_prod = [m['costs']['production_cost'] for m in metrics]
    costs_inv = [m['costs']['inventory_cost'] for m in metrics]
    costs_short = [m['costs']['shortage_cost'] for m in metrics]
    
    cum_prod = np.cumsum(costs_prod)
    cum_inv = np.cumsum(costs_inv)
    cum_short = np.cumsum(costs_short)
    
    axes[1, 1].plot(periods, cum_prod, marker='o', label='Production')
    axes[1, 1].plot(periods, cum_inv, marker='s', label='Stockage')
    axes[1, 1].plot(periods, cum_short, marker='^', label='Rupture')
    axes[1, 1].set_title('Coûts Cumulés')
    axes[1, 1].set_xlabel('Période')
    axes[1, 1].set_ylabel('Coût Cumulé')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_dir = f"./results/{example_name}/"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}viz_{strategy_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   📊 Visualisation sauvegardée: {filename}")
    plt.close()

def test_single_example(example_name: str, strategies: list = None, 
                       visualize: bool = True):
    """Teste un exemple avec différentes stratégies"""
    
    print(f"\n{'='*80}")
    print(f"🎯 TEST SUR EXEMPLE: {example_name.upper()}")
    print(f"{'='*80}\n")
    
    # Charger la configuration
    config = get_example_config(example_name)
    
    print(f"📋 Configuration:")
    print(f"   Horizon: {config.horizon} périodes")
    print(f"   Produits: {config.n_products}")
    print(f"   Capacité régulière: {config.regular_capacity}")
    print(f"   Heures supp: {config.overtime_capacity}")
    print(f"   Sous-traitance: {config.subcontracting_capacity}")
    print(f"   Stock initial: {config.initial_stock}")
    print()
    
    # Stratégies à tester
    if strategies is None:
        strategies = list(BASELINE_STRATEGIES.keys())
    
    # Créer l'environnement pour récupérer les demandes
    env = EnvironmentRegistry.create('strategic', config)
    obs, info = env.reset(seed=42)
    demands = info['demands'][0, :]  # Demandes pour le premier produit
    
    print(f"📈 Demandes: {demands}\n")
    
    results_summary = []
    
    for strategy_name in strategies:
        print(f"--- Stratégie: {strategy_name} ---")
        
        # Recréer l'environnement pour cette stratégie
        env = EnvironmentRegistry.create('strategic', config)
        StrategyClass = BASELINE_STRATEGIES[strategy_name]
        strategy = StrategyClass(env)
        
        # Exécuter un épisode
        total_reward, info = strategy.run_episode()
        metrics = info['metrics']
        final_info = info['final_info']
        
        # Calculer les métriques
        total_cost = sum(
            m['costs']['production_cost'] + 
            m['costs']['inventory_cost'] + 
            m['costs']['shortage_cost']
            for m in metrics
        )
        
        avg_service = np.mean([m['demand_fulfillment'] for m in metrics])
        avg_stock = np.mean([m['inventory_level'][0] for m in metrics])
        
        results_summary.append({
            'Stratégie': strategy_name,
            'Reward Total': f"{total_reward:.2f}",
            'Coût Total': f"{total_cost:.2f}",
            'Service Moyen': f"{avg_service:.3f}",
            'Stock Moyen': f"{avg_stock:.1f}",
            'Stock Final': f"{final_info['inventory_level'][0]:.1f}"
        })
        
        print(f"   ✓ Reward: {total_reward:.2f}")
        print(f"   ✓ Coût total: {total_cost:.2f}")
        print(f"   ✓ Service moyen: {avg_service:.3f}")
        print(f"   ✓ Stock final: {final_info['inventory_level'][0]:.1f}")
        print()
        
        # Visualiser
        if visualize:
            visualize_episode(metrics, example_name, strategy_name, demands)
    
    # Afficher le résumé
    df = pd.DataFrame(results_summary)
    print(f"\n📊 RÉSUMÉ DES RÉSULTATS - {example_name.upper()}")
    print("="*80)
    print(df.to_string(index=False))
    
    # Identifier le meilleur
    best_service = df.loc[df['Service Moyen'].astype(float).idxmax(), 'Stratégie']
    best_cost = df.loc[df['Coût Total'].str.replace(',', '').astype(float).idxmin(), 'Stratégie']
    
    print(f"\n🏆 MEILLEURS RÉSULTATS:")
    print(f"   Meilleur Service: {best_service}")
    print(f"   Coût le plus bas: {best_cost}")
    
    # Sauvegarder
    output_dir = f"./results/{example_name}/"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}summary.csv", index=False)
    print(f"\n💾 Résumé sauvegardé: {output_dir}summary.csv")
    
    return df

def test_all_examples(visualize: bool = True):
    """Teste tous les exemples"""
    
    examples = ['rouleurs', 'pdp_table', 'compresseurs', 'usinage']
    all_results = {}
    
    for example in examples:
        try:
            df = test_single_example(example, visualize=visualize)
            all_results[example] = df
        except Exception as e:
            print(f"❌ Erreur sur {example}: {e}")
            import traceback
            traceback.print_exc()
    
    # Résumé global
    print(f"\n{'='*80}")
    print("🎉 RÉSUMÉ GLOBAL")
    print(f"{'='*80}\n")
    
    for example, df in all_results.items():
        print(f"\n{example.upper()}:")
        best = df.iloc[0]['Stratégie']  # Simplification
        print(f"   Testés: {len(df)} stratégies")
        print(f"   Voir détails: ./results/{example}/")

def main():
    parser = argparse.ArgumentParser(
        description="Tester les stratégies sur les exemples réels"
    )
    parser.add_argument(
        '--example', 
        type=str, 
        choices=['rouleurs', 'pdp_table', 'compresseurs', 'usinage', 'all'],
        default='all',
        help='Exemple à tester'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=list(BASELINE_STRATEGIES.keys()),
        help='Stratégies à tester (défaut: toutes)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Désactiver la visualisation'
    )
    
    args = parser.parse_args()
    
    print("\n🔬 TEST SUR EXEMPLES RÉELS")
    print("="*80)
    
    if args.example == 'all':
        test_all_examples(visualize=not args.no_viz)
    else:
        test_single_example(
            args.example, 
            strategies=args.strategies,
            visualize=not args.no_viz
        )
    
    print("\n✅ Tests terminés!")

if __name__ == "__main__":
    main()
